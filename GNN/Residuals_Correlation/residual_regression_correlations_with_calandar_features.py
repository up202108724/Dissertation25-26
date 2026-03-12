
import pandas as pd
import numpy as np


def _prepare_regression_design(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    
    categorical_cols = categorical_cols or []

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].copy()

    # One-hot encode chosen categorical columns
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
    fit_mode: str = "full_sample",   # "full_sample" or "expanding"
    min_train_size: int = 28,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not feature_cols:
        raise ValueError("feature_cols must contain at least one regressor")

    required = [node_col, date_col, value_col] + feature_cols
    data = df[required].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([node_col, date_col]).reset_index(drop=True)

    # Global design matrix so every item shares the same columns
    X_all, used_features = _prepare_regression_design(
        data,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
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
def build_residual_regression_correlation_graph(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    value_col: str = "value",
    feature_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    fit_mode: str = "full_sample",
    min_train_size: int = 28,
    drop_first: bool = True,
    min_overlap: int = 10,
    corr_method: str = "pearson",
    corr_threshold: float = 0.2,
    absolute_corr: bool = True,
):
    """
    Build a residual-correlation adjacency list from daily node evolution,
    using regression residuals instead of raw values.

    See `build_residual_correlation_graph` for parameter explanations.
    """
    data, residuals_wide = _compute_regression_residuals(
        df=df,
        node_col=node_col,
        date_col=date_col,
        value_col=value_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        fit_mode=fit_mode,
        min_train_size=min_train_size,
        drop_first=drop_first,
    )

    corr_matrix = residuals_wide.corr(method=corr_method, min_periods=min_overlap)

    if absolute_corr:
        corr_matrix = corr_matrix.abs()

    adj_list = {}
    for node in corr_matrix.columns:
        neighbors = corr_matrix.index[corr_matrix[node] >= corr_threshold].tolist()
        neighbors.remove(node)  # Remove self-correlation
        adj_list[node] = neighbors

    return adj_list, corr_matrix
