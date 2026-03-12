import pandas as pd

def _compute_residuals(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    target_col: str = "value",
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
    data = df[[node_col, date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([node_col, date_col])

    min_periods = max(2, trend_window // 2)

    rolling_trend = data.groupby(node_col)[target_col].transform(
        lambda s: s.rolling(window=trend_window, min_periods=min_periods).mean().shift(1)
    )

    fallback = data.groupby(node_col)[target_col].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    data["expected"] = rolling_trend.fillna(fallback)
    data["residual"] = data[target_col] - data["expected"]

    residuals_wide = data.pivot(index=date_col, columns=node_col, values="residual")

    return data, residuals_wide


def build_residual_correlation_graph(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    target_col: str = "value",
    trend_window: int = 7,
    min_overlap: int = 10,
    corr_method: str = "pearson",
    corr_threshold: float = 0.2,
    absolute_corr: bool = True,
):
    """
    Build a residual-correlation adjacency list from daily node evolution.
    """
    data, residuals_wide = _compute_residuals(
        df=df,
        node_col=node_col,
        date_col=date_col,
        target_col=target_col,
        trend_window=trend_window,
    )

    corr_matrix = residuals_wide.corr(method=corr_method)

    not_na_matrix = residuals_wide.notna().astype(int)
    overlap = not_na_matrix.T @ not_na_matrix

    nodes = corr_matrix.columns.tolist()
    adj_list = {node: [] for node in nodes}
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            corr = corr_matrix.loc[n1, n2]
            n_common = overlap.loc[n1, n2]

            if pd.isna(corr) or n_common < min_overlap:
                continue

            keep = abs(corr) >= corr_threshold if absolute_corr else corr >= corr_threshold

            if keep:
                adj_list[n1].append(n2)
                adj_list[n2].append(n1)

    return adj_list, corr_matrix