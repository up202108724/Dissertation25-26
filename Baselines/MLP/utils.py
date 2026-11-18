import numpy as np
from typing import Tuple
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def _proximity_decay(
    dates: pd.Series,
    month: int,
    day: int,
    tau_before: float = 10.0,
    tau_after: float = 3.0,
):
    """Asymmetric exp-decay proximity to an annual fixed-date event.

    Returns a value in [0, 1]:
      - 1.0  on the event day itself.
      - exp(-days_to  / tau_before) in the days leading up to the event.
      - exp(-days_from / tau_after)  in the days after the event.

    Asymmetry reflects retail demand patterns: demand builds gradually before
    a holiday (large tau_before) and drops sharply after (small tau_after).
    Defaults: tau_before=10 (~2-week ramp), tau_after=3 (~3-day tail).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year
    this_year = pd.to_datetime(dict(year=years, month=month, day=day))
    next_year = pd.to_datetime(dict(year=years + 1, month=month, day=day))
    prev_year = pd.to_datetime(dict(year=years - 1, month=month, day=day))

    # Signed distance to nearest occurrence: negative = before event, positive = after.
    d_this = (dates - this_year).dt.days.values   # positive after this year's event
    d_next = (dates - next_year).dt.days.values   # always negative (before)
    d_prev = (dates - prev_year).dt.days.values   # always positive (after)

    # Pick the occurrence that minimises |distance|.
    candidates = np.stack([d_this, d_next, d_prev], axis=1)
    idx = np.argmin(np.abs(candidates), axis=1)
    signed = candidates[np.arange(len(candidates)), idx]   # <0 before, >0 after, 0 on day

    tau = np.where(signed <= 0, tau_before, tau_after)
    return np.exp(-np.abs(signed) / tau).astype(np.float32)


def _proximity_decay_dynamic(
    dates: pd.Series,
    holiday_map,
    holiday_name: str,
    tau_before: float = 7.0,
    tau_after: float = 3.0,
    offset_days: int = 0,
):
    """Asymmetric exp-decay proximity for a moving holiday (e.g. Thanksgiving).

    Looks up every occurrence of *holiday_name* in *holiday_map*, optionally
    shifts by *offset_days* (positive = later, used for Black Friday = +1),
    then applies the same asymmetric decay as `_proximity_decay`.

    Defaults: tau_before=7 (~1-week ramp), tau_after=3 (~3-day tail).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    event_dates = pd.to_datetime(sorted(
        d + pd.Timedelta(days=offset_days)
        for d, name in holiday_map.items()
        if name == holiday_name
    ))
    if len(event_dates) == 0:
        return np.zeros(len(dates), dtype=np.float32)

    dates_arr = dates.values.astype("datetime64[D]")
    events_arr = event_dates.values.astype("datetime64[D]")

    # For each date find the nearest event occurrence.
    signed_days = np.array([
        min(
            ((d - e) / np.timedelta64(1, "D") for e in events_arr),
            key=abs,
        )
        for d in dates_arr
    ], dtype=float)

    tau = np.where(signed_days <= 0, tau_before, tau_after)
    return np.exp(-np.abs(signed_days) / tau).astype(np.float32)
def _signed_days_to_event(dates: pd.Series, month: int, day: int, clip: int = 30):
    """Return (days_to_next, days_from_prev) for an annual fixed-date event.

    Both are non-negative integers clipped to [0, clip]. `days_to_next` is 0 on
    the event itself (and `days_from_prev` is also 0 that day).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year
    this_year = pd.to_datetime(dict(year=years, month=month, day=day))
    next_year = pd.to_datetime(dict(year=years + 1, month=month, day=day))
    prev_year = pd.to_datetime(dict(year=years - 1, month=month, day=day))

    days_to = np.where(
        this_year.values >= dates.values,
        (this_year - dates).dt.days.values,
        (next_year - dates).dt.days.values,
    )
    days_from = np.where(
        this_year.values <= dates.values,
        (dates - this_year).dt.days.values,
        (dates - prev_year).dt.days.values,
    )
    return np.clip(days_to, 0, clip).astype(int), np.clip(days_from, 0, clip).astype(int)


def generate_exogenous_features(df, exog_cols, date_col='date', target_col='value', group_cols=None):
    """
    Generates specific calendar, cyclical, and holiday exogenous features for a DataFrame
    based on the provided `exog_cols` list.
    """
    df = df.copy()
    
    if group_cols is None:
        group_cols = [c for c in ['item_id', 'store_id'] if c in df.columns]
        if not group_cols:
            group_cols = None
    
    # -----------------------------------------------------------------------------
    # FEATURE BUILDER DICTIONARY
    # -----------------------------------------------------------------------------
    def _get_holidays():
        us_holidays = holidays.US()
        return us_holidays, pd.to_datetime(sorted(us_holidays.keys()))

    _holidays_cache = None
    def get_holiday_dates():
        nonlocal _holidays_cache
        if _holidays_cache is None:
            _holidays_cache = _get_holidays()
        return _holidays_cache

    builders = {
        # BASIC CALENDAR PARTS
        "day_of_week": lambda d: d[date_col].dt.dayofweek.astype(int),
        "day_of_month": lambda d: d[date_col].dt.day.astype(int),
        "month": lambda d: d[date_col].dt.month.astype(int),
        "moy": lambda d: (d[date_col].dt.month - 1).astype(int),
        "quarter": lambda d: d[date_col].dt.quarter.astype(int),
        "doy": lambda d: (d[date_col].dt.dayofyear - 1).astype(int),
        "week_of_year": lambda d: d[date_col].dt.isocalendar().week.astype(int),
        "year": lambda d: d[date_col].dt.year.astype(int),
        
        "is_weekend": lambda d: d[date_col].dt.dayofweek.isin([5, 6]).astype(int),
        "is_monday": lambda d: (d[date_col].dt.dayofweek == 0).astype(int),
        "is_friday": lambda d: (d[date_col].dt.dayofweek == 4).astype(int),

        "is_month_start": lambda d: d[date_col].dt.is_month_start.astype(int),
        "is_month_end": lambda d: d[date_col].dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d[date_col].dt.is_quarter_start.astype(int),
        "is_quarter_end": lambda d: d[date_col].dt.is_quarter_end.astype(int),
        "week_of_month": lambda d: ((d[date_col].dt.day - 1) // 7 + 1).astype(int),

            # CYCLICAL ENCODINGS — base harmonics
        "dow_sin": lambda d: np.sin(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos": lambda d: np.cos(2 * np.pi * d[date_col].dt.dayofweek / 7),

        "dom_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),
        "dom_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),

        "wom_sin": lambda d: np.sin(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),
        "wom_cos": lambda d: np.cos(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),

        "month_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "quarter_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),
        "quarter_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),

        "woy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # CYCLICAL ENCODINGS — 2nd harmonics
        "dow_sin2": lambda d: np.sin(4 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos2": lambda d: np.cos(4 * np.pi * d[date_col].dt.dayofweek / 7),

        "month_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "woy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # EXACT HOLIDAYS
        "is_holiday": lambda d: d[date_col].isin(get_holiday_dates()[1]).astype(int),
        "is_christmas": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 25)).astype(int),
        "is_thanksgiving": lambda d: d[date_col].apply(lambda x: 1 if get_holiday_dates()[0].get(x) == "Thanksgiving Day" else 0),
        "is_black_friday": lambda d: d[date_col].isin(
            pd.to_datetime([day for day, name in get_holiday_dates()[0].items() if name == "Thanksgiving Day"]) + pd.Timedelta(days=1)
        ).astype(int),
        "is_christmas_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 24)).astype(int),
        "is_new_year_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 31)).astype(int),
        # Dec 19–25 inclusive: captures the full pre-Christmas demand ramp.
        "is_christmas_week": lambda d: (
            (d[date_col].dt.month == 12) & (d[date_col].dt.day.between(19, 25))
        ).astype(int),
        # Signed proximity to nearest Christmas, clipped at +/- 30 so the
        # feature is informative near Dec 25 and saturates the rest of the year.
        "days_to_christmas": lambda d: _signed_days_to_event(d[date_col], 12, 25, clip=30)[0],
        "days_from_christmas": lambda d: _signed_days_to_event(d[date_col], 12, 25, clip=30)[1],
        # Smooth exp-decay bump in [0, 1]: 1 on Dec 25, ~0.37 at tau days,
        # ~0 far away. No flat region, no arbitrary clip.
        "christmas_proximity": lambda d: _proximity_decay(d[date_col], 12, 25, tau_before=10.0, tau_after=3.0),
        "thanksgiving_proximity": lambda d: _proximity_decay_dynamic(
            d[date_col], _get_holidays()[0], "Thanksgiving Day", tau_before=7.0, tau_after=3.0
        ),
        "black_friday_proximity": lambda d: _proximity_decay_dynamic(
            d[date_col], _get_holidays()[0], "Thanksgiving Day", tau_before=3.0, tau_after=2.0, offset_days=1
        ),

        # PROMOTIONS
        "promo_type_FRPG": lambda d: d.get("promo_type_FRPG", pd.Series(0, index=d.index)).astype(int),
        "promo_value_FRPG": lambda d: d.get("promo_value_FRPG", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_GAS": lambda d: d.get("promo_type_GAS", pd.Series(0, index=d.index)).astype(int),
        "promo_value_GAS": lambda d: d.get("promo_value_GAS", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_BOGO": lambda d: d.get("promo_type_BOGO", pd.Series(0, index=d.index)).astype(int),
        "promo_value_BOGO": lambda d: d.get("promo_value_BOGO", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_DISC": lambda d: d.get("promo_type_DISC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_DISC": lambda d: d.get("promo_value_DISC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRC": lambda d: d.get("promo_type_CIRC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRC": lambda d: d.get("promo_value_CIRC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRE": lambda d: d.get("promo_type_CIRE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRE": lambda d: d.get("promo_value_CIRE", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CLCP": lambda d: d.get("promo_type_CLCP", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CLCP": lambda d: d.get("promo_value_CLCP", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_LFPE": lambda d: d.get("promo_type_LFPE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_LFPE": lambda d: d.get("promo_value_LFPE", pd.Series(0.0, index=d.index)).astype(float),
        
        #Trend indicators could be added here as well, but we will handle them separately in the main function to avoid data leakage
        # TREND
        "time_idx": lambda d: (d[date_col] - d[date_col].min()).dt.days.astype(int),
        "time_idx_sq": lambda d: ((d[date_col] - d[date_col].min()).dt.days.astype(float) ** 2),
        
    }

    # Generate only the requested features
    for col in exog_cols:
        if col in builders:
            df[col] = builders[col](df)
        elif col.startswith("is_pre_holiday_") or col.startswith("is_post_holiday_"):
            parts = col.split("_")
            lag = int(parts[-1])
            is_pre = "pre" in parts
            
            _, holiday_dates = get_holiday_dates()
            
            # Efficiently compute proximity using sets and exact matches (avoiding loop row-by-row)
            df[col] = 0
            for h in holiday_dates:
                target_date = h - pd.Timedelta(days=lag) if is_pre else h + pd.Timedelta(days=lag)
                df.loc[df[date_col] == target_date, col] = 1
                
        elif col.startswith("lag_"):
            lag = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].shift(lag).fillna(0)
            else:
                df[col] = df[target_col].shift(lag).fillna(0)
                
        elif col.startswith("rolling_mean_excl_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                ).fillna(0)
            else:
                df[col] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)

        elif col.startswith("rolling_mean_"):
            # NOTE: leak-safe version — excludes the current observation by shifting one step.
            # The previous implementation included y_t in the mean, which leaks the target at training time.
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                ).fillna(0)
            else:
                df[col] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)

        elif col == "is_bridge_day":
            _, holiday_dates = get_holiday_dates()
            holiday_set = set(pd.to_datetime(holiday_dates).normalize())

            d = df[date_col].dt.normalize()
            prev_is_holiday = (d - pd.Timedelta(days=1)).isin(holiday_set)
            next_is_holiday = (d + pd.Timedelta(days=1)).isin(holiday_set)
            dow = d.dt.dayofweek
            df[col] = ((prev_is_holiday & (dow == 4)) | (next_is_holiday & (dow == 0))).astype(int)

        elif col in df.columns:
            # If the column exists in the dataset natively (e.g., store_id, cat_label) and isn't a builder key, just pass
            pass
                    
        else:
            print(f"Warning: Builder for feature '{col}' not found.")

    return df


class ExogenousScaler:
    """
    Type-aware scaler for exogenous features.

    - Binary features (e.g. `is_*`, `promo_type_*`): pass-through.
    - Cyclical features (`*_sin`, `*_cos`): pass-through (already in [-1, 1]).
    - Continuous features (lags, rolling means, time_idx, day_of_week, ...):
      scaled with MinMaxScaler or StandardScaler.

    This avoids squashing binary/cyclical features and changing their semantics.
    """

    def __init__(self, continuous_strategy: str = 'minmax'):
        self.scalers = {}
        self.feature_types = {}
        self.continuous_strategy = continuous_strategy

    @staticmethod
    def _categorize_feature(col_name: str) -> str:
        if col_name.startswith("is_") or col_name.startswith("promo_type_"):
            return "binary"
        if col_name.endswith("_sin") or col_name.endswith("_cos") \
                or col_name.endswith("_sin2") or col_name.endswith("_cos2"):
            return "cyclical"
        return "continuous"

    def fit(self, df, columns):
        for col in columns:
            ftype = self._categorize_feature(col)
            self.feature_types[col] = ftype
            if ftype == "continuous":
                if self.continuous_strategy == 'minmax':
                    scaler = MinMaxScaler()
                elif self.continuous_strategy == 'standard':
                    scaler = StandardScaler()
                else:
                    raise ValueError("continuous_strategy must be 'minmax' or 'standard'")
                scaler.fit(np.asarray(df[col]).reshape(-1, 1))
                self.scalers[col] = scaler
        return self

    def transform(self, df, columns):
        return_numpy = isinstance(df, np.ndarray)
        if return_numpy:
            df = pd.DataFrame(df, columns=columns)
        else:
            df = df.copy()

        for col in columns:
            if self.feature_types.get(col, self._categorize_feature(col)) == "continuous" \
                    and col in self.scalers:
                df[col] = self.scalers[col].transform(np.asarray(df[col]).reshape(-1, 1)).ravel()
        return df.values if return_numpy else df

    def fit_transform(self, df, columns):
        return self.fit(df, columns).transform(df, columns)

    def inverse_transform(self, df, columns):
        return_numpy = isinstance(df, np.ndarray)
        if return_numpy:
            df = pd.DataFrame(df, columns=columns)
        else:
            df = df.copy()

        for col in columns:
            if col in self.scalers:
                df[col] = self.scalers[col].inverse_transform(
                    np.asarray(df[col]).reshape(-1, 1)
                ).ravel()
        return df.values if return_numpy else df


def parse_dynamic_exog_cols(exog_cols):
    """
    Identify dynamic exogenous columns that depend on the target series.
    Returns:
        lag_cols: dict {col_name: k}
        roll_cols: dict {col_name: window}  (treated as leak-safe: mean of previous `window` targets)
    Other columns are considered "static" (calendar/holiday/promo) and pass through unchanged.
    """
    lag_cols, roll_cols = {}, {}
    for c in exog_cols:
        if c.startswith("lag_"):
            try:
                lag_cols[c] = int(c.split("_")[-1])
            except ValueError:
                pass
        elif c.startswith("rolling_mean_excl_") or c.startswith("rolling_mean_"):
            try:
                roll_cols[c] = int(c.split("_")[-1])
            except ValueError:
                pass
    return lag_cols, roll_cols
