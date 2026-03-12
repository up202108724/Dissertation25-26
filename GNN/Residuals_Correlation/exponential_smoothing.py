import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

def _compute_exponential_smoothing_residuals(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    value_col: str = "value",
    seasonal_periods: int = 7,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute residuals per item using Holt-Winters / Exponential Smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    node_col : str
        Column name identifying the time series node (e.g. item_id).
    date_col : str
        Column name for dates.
    value_col : str
        Column name for observed values.
    seasonal_periods : int
        Number of periods in a seasonal cycle (default=7 for weekly seasonality).
    trend : str | None
        Type of trend component. {"add", "mul", None}
    seasonal : str | None
        Type of seasonal component. {"add", "mul", None}

    Returns
    -------
    data : pd.DataFrame
        Long dataframe with expected and residual columns.
    residuals_wide : pd.DataFrame
        Wide residual matrix (index=date, columns=node_col).
    """
   


    required = [node_col, date_col, value_col]

    data = df[required].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([node_col, date_col]).reset_index(drop=True)

    data["expected"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item_id, idx in data.groupby(node_col, sort=False).groups.items():
            idx = list(idx)
            y_item = data.iloc[idx][value_col].to_numpy()

            # Ensure we have enough data to fit seasonal model
            if len(y_item) < 2 * seasonal_periods:
                s = pd.Series(y_item)
                expected = s.rolling(window=seasonal_periods, min_periods=1).mean().shift(1)
                expected = expected.fillna(s.expanding().mean().shift(1))
                data.loc[idx, "expected"] = expected.to_numpy()
                continue

            try:
                model = ExponentialSmoothing(
                    y_item,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                )
                res = model.fit()
                # 1-step ahead in-sample predictions
                data.loc[idx, "expected"] = res.fittedvalues
            except Exception:
                # Fallback to rolling mean if Holt-Winters fails to converge/fit
                s = pd.Series(y_item)
                expected = s.rolling(window=seasonal_periods, min_periods=1).mean().shift(1)
                expected = expected.fillna(s.expanding().mean().shift(1))
                data.loc[idx, "expected"] = expected.to_numpy()

    data["residual"] = data[value_col] - data["expected"]
    residuals_wide = data.pivot(index=date_col, columns=node_col, values="residual")

    return data, residuals_wide

import networkx as nx

def build_exponential_smoothing_correlation_graph(
    df: pd.DataFrame,
    node_col: str = "item_id",
    date_col: str = "date",
    value_col: str = "value",
    seasonal_periods: int = 7,
    trend: str | None = "add",
    seasonal: str | None = "add",
    min_overlap: int = 10,
    corr_method: str = "pearson",
    corr_threshold: float = 0.2,
    absolute_corr: bool = True,
):
    """
    Build a correlation graph based on Exponential Smoothing residuals.
    """
    data, residuals_wide = _compute_exponential_smoothing_residuals(
        df=df,
        node_col=node_col,
        date_col=date_col,
        value_col=value_col,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
    )

    corr_matrix = residuals_wide.corr(method=corr_method)

    not_na_matrix = residuals_wide.notna().astype(int)
    overlap = not_na_matrix.T @ not_na_matrix

    G = nx.Graph()

    node_stats = (
        data.groupby(node_col)
        .agg(
            n_obs=(value_col, "count"),
            mean_value=(value_col, "mean"),
            std_value=(value_col, "std"),
            mean_residual=("residual", "mean"),
            std_residual=("residual", "std"),
        )
        .reset_index()
    )

    for _, row in node_stats.iterrows():
        G.add_node(
            row[node_col],
            n_obs=int(row["n_obs"]),
            mean_value=float(row["mean_value"]) if pd.notna(row["mean_value"]) else 0.0,
            std_value=float(row["std_value"]) if pd.notna(row["std_value"]) else 0.0,
            mean_residual=float(row["mean_residual"]) if pd.notna(row["mean_residual"]) else 0.0,
            std_residual=float(row["std_residual"]) if pd.notna(row["std_residual"]) else 0.0,
        )

    nodes = corr_matrix.columns.tolist()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            corr = corr_matrix.loc[n1, n2]
            n_common = overlap.loc[n1, n2]

            if pd.isna(corr) or n_common < min_overlap:
                continue

            keep = abs(corr) >= corr_threshold if absolute_corr else corr >= corr_threshold

            if keep:
                G.add_edge(
                    n1,
                    n2,
                    weight=float(corr),
                    abs_weight=float(abs(corr)),
                    n_common=int(n_common),
                    sign="positive" if corr >= 0 else "negative",
                )

    return G, corr_matrix, residuals_wide
