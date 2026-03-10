
import pandas as pd


def _compute_residuals(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    value_col: str = "value",
    trend_window: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute expected values and residuals per node, then return both
    the long dataframe and the wide residual matrix.

    Residuals are defined as:
        residual = observed - expected

    Expected values are computed from:
    - a shifted rolling mean over the previous `trend_window` observations
    - with a shifted expanding mean fallback for early periods
    """
    data = df[[node_col, date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([node_col, date_col])

    min_periods = max(2, trend_window // 2)

    rolling_trend = data.groupby(node_col)[value_col].transform(
        lambda s: s.rolling(window=trend_window, min_periods=min_periods).mean().shift(1)
    )

    fallback = data.groupby(node_col)[value_col].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    data["expected"] = rolling_trend.fillna(fallback)
    data["residual"] = data[value_col] - data["expected"]

    residuals_wide = data.pivot(index=date_col, columns=node_col, values="residual")

    return data, residuals_wide

import numpy as np
import pandas as pd


def _prepare_regression_design(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    cyclical_periods: dict[str, float] | None = None,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a regression design matrix from user-selected variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Columns to include in the regression.
    categorical_cols : list[str] | None
        Subset of feature_cols to one-hot encode.
    cyclical_periods : dict[str, float] | None
        Mapping like {"doy": 365.25, "day_of_week": 7}.
        For each listed column, creates sin/cos features instead of using
        the raw column directly.
    drop_first : bool
        Whether to drop the first dummy level.

    Returns
    -------
    X : pd.DataFrame
        Numeric design matrix.
    used_feature_names : list[str]
        Final column names used in X.
    """
    categorical_cols = categorical_cols or []
    cyclical_periods = cyclical_periods or {}

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].copy()

    # Replace cyclical columns by sin/cos features
    for col, period in cyclical_periods.items():
        if col not in X.columns:
            raise ValueError(f"Cyclical column '{col}' must also be in feature_cols")
        angle = 2 * np.pi * X[col].astype(float) / float(period)
        X[f"{col}_sin"] = np.sin(angle)
        X[f"{col}_cos"] = np.cos(angle)
        X = X.drop(columns=[col])

    # One-hot encode chosen categorical columns, excluding those replaced by cyclical features
    effective_categoricals = [c for c in categorical_cols if c in X.columns]
    if effective_categoricals:
        X = pd.get_dummies(X, columns=effective_categoricals, drop_first=drop_first, dtype=float)

    # Force numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X.astype(float), X.columns.tolist()


def _ols_fit_predict_full_sample(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Fit one OLS model on all valid rows and return in-sample predictions.
    """
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)

    valid = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    preds = np.full(len(y), np.nan)

    if valid.sum() == 0:
        return preds

    X_design = np.column_stack([np.ones(valid.sum()), Xv[valid]])
    beta, *_ = np.linalg.lstsq(X_design, yv[valid], rcond=None)

    preds_valid = np.column_stack([np.ones(valid.sum()), Xv[valid]]) @ beta
    preds[valid] = preds_valid
    return preds


def _ols_fit_predict_expanding(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 28,
) -> np.ndarray:
    """
    Expanding-window OLS:
    prediction at time t is fitted using only rows < t.
    """
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)

    preds = np.full(len(y), np.nan)
    p = Xv.shape[1]

    valid_row = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)

    for t in range(len(y)):
        if not valid_row[t]:
            continue

        train_mask = valid_row[:t]
        n_train = int(train_mask.sum())

        if n_train < max(min_train_size, p + 1):
            continue

        X_train = np.column_stack([np.ones(n_train), Xv[:t][train_mask]])
        y_train = yv[:t][train_mask]

        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

        x_t = np.concatenate([[1.0], Xv[t]])
        preds[t] = x_t @ beta

    return preds


def _compute_regression_residuals(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    value_col: str = "value",
    feature_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    cyclical_periods: dict[str, float] | None = None,
    fit_mode: str = "full_sample",   # "full_sample" or "expanding"
    min_train_size: int = 28,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute regression-based expected values and residuals per item.

    Residuals are defined as:
        residual = observed - expected

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    node_col, date_col, value_col : str
        Item id, date, and target columns.
    feature_cols : list[str]
        User-selected regressors.
    categorical_cols : list[str] | None
        Subset of feature_cols to dummy encode.
    cyclical_periods : dict[str, float] | None
        Example: {"doy": 365.25, "day_of_week": 7}
    fit_mode : str
        "full_sample" for one OLS per item on the whole series.
        "expanding" for past-only predictions.
    min_train_size : int
        Minimum past observations for expanding OLS.
    drop_first : bool
        Passed to pd.get_dummies.

    Returns
    -------
    data : pd.DataFrame
        Long dataframe with expected and residual columns.
    residuals_wide : pd.DataFrame
        Wide residual matrix (index=date, columns=item_id).
    """
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one regressor")

    required = [node_col, date_col, value_col] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([node_col, date_col]).reset_index(drop=True)

    # Global design matrix so every item shares the same columns
    X_all, used_features = _prepare_regression_design(
        data,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        cyclical_periods=cyclical_periods,
        drop_first=drop_first,
    )

    data["expected"] = np.nan

    for item_id, idx in data.groupby(node_col, sort=False).groups.items():
        idx = list(idx)
        X_item = X_all.iloc[idx]
        y_item = data.iloc[idx][value_col]

        if fit_mode == "full_sample":
            preds = _ols_fit_predict_full_sample(X_item, y_item)
        elif fit_mode == "expanding":
            preds = _ols_fit_predict_expanding(
                X_item,
                y_item,
                min_train_size=min_train_size,
            )
        else:
            raise ValueError("fit_mode must be 'full_sample' or 'expanding'")

        data.loc[idx, "expected"] = preds

    data["residual"] = data[value_col] - data["expected"]

    residuals_wide = data.pivot(
        index=date_col,
        columns=node_col,
        values="residual",
    )

    return data, residuals_wide