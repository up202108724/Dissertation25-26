"""
utils.py — Shared utilities for StaticGCN_LSTM.
Mirrors GNN/FixedGraphs/SimpleGNN/LSTM/utils.py (exog feature generation only).
"""

import pandas as pd
import numpy as np
import holidays


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
        "day_of_week":    lambda d: d[date_col].dt.dayofweek.astype(int),
        "day_of_month":   lambda d: d[date_col].dt.day.astype(int),
        "month":          lambda d: d[date_col].dt.month.astype(int),
        "quarter":        lambda d: d[date_col].dt.quarter.astype(int),
        "week_of_year":   lambda d: d[date_col].dt.isocalendar().week.astype(int),
        "week_of_month":  lambda d: ((d[date_col].dt.day - 1) // 7 + 1).astype(int),
        "is_weekend":     lambda d: d[date_col].dt.dayofweek.isin([5, 6]).astype(int),
        "is_monday":      lambda d: (d[date_col].dt.dayofweek == 0).astype(int),
        "is_friday":      lambda d: (d[date_col].dt.dayofweek == 4).astype(int),
        "is_month_start": lambda d: d[date_col].dt.is_month_start.astype(int),
        "is_month_end":   lambda d: d[date_col].dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d[date_col].dt.is_quarter_start.astype(int),
        "is_quarter_end":   lambda d: d[date_col].dt.is_quarter_end.astype(int),
        "dow_sin":  lambda d: np.sin(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos":  lambda d: np.cos(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "doy_sin":  lambda d: np.sin(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos":  lambda d: np.cos(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "month_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "woy_sin":  lambda d: np.sin(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos":  lambda d: np.cos(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "is_holiday": lambda d: d[date_col].isin(get_holiday_dates()[1]).astype(int),
        "is_thanksgiving": lambda d: d[date_col].apply(
            lambda x: 1 if get_holiday_dates()[0].get(x) == "Thanksgiving Day" else 0
        ),
        "is_black_friday": lambda d: d[date_col].isin(
            pd.to_datetime([day for day, name in get_holiday_dates()[0].items()
                            if name == "Thanksgiving Day"]) + pd.Timedelta(days=1)
        ).astype(int),
        "is_christmas":     lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 25)).astype(int),
        "is_christmas_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 24)).astype(int),
        "is_new_year_eve":  lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 31)).astype(int),
        "is_bridge_day": lambda d: pd.Series(0, index=d.index),
        "rolling_mean_excl_7": lambda d: (
            d.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
            ).fillna(0) if group_cols else d[target_col].shift(1).rolling(7, min_periods=1).mean().fillna(0)
        ),
    }

    for col in exog_cols:
        if col in builders:
            df[col] = builders[col](df)
        elif col.startswith("is_pre_holiday_") or col.startswith("is_post_holiday_"):
            parts   = col.split("_")
            lag     = int(parts[-1])
            is_pre  = "pre" in parts
            _, holiday_dates = get_holiday_dates()
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
                df[col] = df[target_col].shift(1).rolling(window, min_periods=1).mean().fillna(0)

    return df
