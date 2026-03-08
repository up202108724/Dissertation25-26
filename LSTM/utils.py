import numpy as np

def build_dynamic_exog_row(next_date, target_history, exog_cols, lag_cols=None, rolling_cols=None, df_row=None):
    """
    Build one exogenous row for next_date using current target history.
    target_history: latest available target values in unscaled space
    """
    row = {}

    # Target-derived features in unscaled space
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

    # Include all other static/future-known features from df_row
    if df_row is not None:
        for col in exog_cols:
            if col not in row and col in df_row.index:
                row[col] = df_row[col]

    return [row[col] for col in exog_cols]