import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import holidays

def compute_metrics(y_test, y_pred):
    
    def POCID(y_test, y_pred):
        diff_original = y_test[1:] - y_test[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        is_positive = (diff_original * diff_pred) > 0
        return is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    pocid = POCID(y_test, y_pred)
    return rmse, mae, bias, score, pocid

def scale_exogenous_features(df, train_slice, categorical_cols=None, continuous_cols=None, binary_cols=None):
    df_scaled = df.copy()
    final_exog_cols = []
    
    cat_scaler = None
    cont_scaler = None
    
    # Scale categorical columns with OneHotEncoder
    if categorical_cols:
        cat_train = df.iloc[train_slice][categorical_cols].values
        cat_scaler = MinMaxScaler()  # Use MinMaxScaler to convert categories to a 0-1 range before OHE
        cat_scaler.fit(cat_train)
        
        cat_full_scaled = cat_scaler.transform(df[categorical_cols].values)
        cat_feature_names = cat_scaler.get_feature_names_out(categorical_cols)
        
        # Add new OHE columns and drop the originals
        df_cat = pd.DataFrame(cat_full_scaled, columns=cat_feature_names, index=df.index)
        df_scaled = pd.concat([df_scaled.drop(columns=categorical_cols), df_cat], axis=1)
        final_exog_cols.extend(cat_feature_names)
        
    # Scale continuous columns
    if continuous_cols:
        cont_train = df.iloc[train_slice][continuous_cols].values
        cont_scaler = MinMaxScaler()
        cont_scaler.fit(cont_train)
        
        cont_full_scaled = cont_scaler.transform(df[continuous_cols].values)
        df_scaled[continuous_cols] = cont_full_scaled
        final_exog_cols.extend(continuous_cols)
        
    # Keep binary columns as they are
    if binary_cols:
        final_exog_cols.extend(binary_cols)
        
    return df_scaled, final_exog_cols, cat_scaler, cont_scaler

def build_dynamic_exog_row(next_date, target_history, exog_cols,
                           lag_cols=None, rolling_cols=None, df_row=None):
    """
    Build one RAW exogenous row for next_date using current RAW target history.
    """
    row = {}

    if lag_cols:
        for col in lag_cols:
            lag = int(col.split('_')[1])
            row[col] = target_history[-lag] if len(target_history) >= lag else 0.0

    if rolling_cols:
        for col in rolling_cols:
            window = int(col.split('_')[2])
            if len(target_history) >= window:
                row[col] = float(np.mean(target_history[-window:]))
            else:
                row[col] = float(np.mean(target_history)) if len(target_history) > 0 else 0.0

    if df_row is not None:
        for col in exog_cols:
            if col not in row and col in df_row.index:
                row[col] = df_row[col]

    return pd.DataFrame([[row.get(col, 0.0) for col in exog_cols]], columns=exog_cols)



def transform_exog_row(row_df, categorical_cols=None, continuous_cols=None, binary_cols=None,
                       cat_scaler=None, cont_scaler=None, final_exog_cols=None):
    parts = []

    if categorical_cols:
        # Force column order
        cat_arr = cat_scaler.transform(row_df[categorical_cols].values)
        cat_names = cat_scaler.get_feature_names_out(categorical_cols)
        parts.append(pd.DataFrame(cat_arr, columns=cat_names, index=row_df.index))

    if continuous_cols:
        # Force column order precisely as fitted
        cont_arr = cont_scaler.transform(row_df[continuous_cols].values)
        parts.append(pd.DataFrame(cont_arr, columns=continuous_cols, index=row_df.index))

    if binary_cols:
        # Force column order
        parts.append(row_df[binary_cols].copy())

    out = pd.concat(parts, axis=1)

    if final_exog_cols is not None:
        # Reindex to ensure strictly ordered columns matching the training tensors
        out = out.reindex(columns=final_exog_cols, fill_value=0.0)

    return out

def generate_exogenous_features(df, exog_cols, date_col='date'):
    """
    Generates specific calendar, cyclical, and holiday exogenous features for a DataFrame
    based on the provided `exog_cols` list.
    """
    df = df.copy()
    
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
                
        elif col == "is_bridge_day":
            _, holiday_dates = get_holiday_dates()
            holiday_set = set(holiday_dates)
            
            df[col] = 0
            for idx in df.index:
                d = df.at[idx, date_col]
                prev_day = d - pd.Timedelta(days=1)
                next_day = d + pd.Timedelta(days=1)
                if ((prev_day in holiday_set and d.dayofweek == 4) or 
                    (next_day in holiday_set and d.dayofweek == 0)):
                    df.at[idx, col] = 1

        elif col in df.columns:
            # If the column exists in the dataset natively (e.g., store_id, cat_label) and isn't a builder key, just pass
            pass
                    
        else:
            print(f"Warning: Builder for feature '{col}' not found.")

    return df


