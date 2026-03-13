import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import dtw
import networkx as nx

def build_standardscaled_distance_graph(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    distance_value_threshold: float = 1.0,
    k: int = None,
):
    df_pivot = (
        df.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc
        )
        .sort_index()
        .ffill()
        .fillna(0)
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_pivot),
        index=df_pivot.index,
        columns=df_pivot.columns
    )
    dist_matrix = squareform(pdist(df_scaled.T.values, metric="euclidean"))
    item_ids = df_scaled.columns.tolist()
    G = nx.Graph(name="StandardScaled_Distance_Graph")
    G.add_nodes_from(item_ids)
    n=len(item_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= distance_value_threshold:
                G.add_edge(item_ids[i], item_ids[j], weight=float(dist_matrix[i, j]))
                print(f"Added edge between {item_ids[i]} and {item_ids[j]} with DTW distance: {dist_matrix[i, j]:.4f}")
    print("Number of nodes in the DTW graph:", G.number_of_nodes())
    print("Number of edges in the DTW graph:", G.number_of_edges())

    dist_df = pd.DataFrame(dist_matrix, index=item_ids, columns=item_ids)
    

    return G, dist_df, df_scaled

### DTW-based graph construction

def build_dtw_graph(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    distance_value_threshold: float = 0.3,
    use_log1p: bool = False,
    sakoe_chiba_radius: int | None = None,
):
    # 1) Pivot: rows = dates, cols = items
    df_pivot = (
        df.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc
        )
        .sort_index()
    )

    # 2) Missing values
    # For retail, fillna(0) is usually safer than ffill for demand
    df_pivot = df_pivot.fillna(0)

    # 3) Optional variance compression
    if use_log1p:
        df_pivot = np.log1p(df_pivot)

    # 4) Standardize each item across time (z-normalization)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(df_pivot)
    df_scaled = pd.DataFrame(scaled, index=df_pivot.index, columns=df_pivot.columns)

    item_ids = df_scaled.columns.tolist()
    X = df_scaled.T.values   # shape = (n_items, T)

    n = len(item_ids)
    dist_matrix = np.zeros((n, n), dtype=float)

    G = nx.Graph(name="DTW_Graph")
    G.add_nodes_from(item_ids)

    # 5) Pairwise DTW distances
    for i in range(n):
        for j in range(i + 1, n):
            if sakoe_chiba_radius is None:
                d = dtw(X[i], X[j])
            else:
                d = dtw(X[i], X[j], global_constraint="sakoe_chiba",
                        sakoe_chiba_radius=sakoe_chiba_radius)

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            #print(f"Computed DTW distance between {item_ids[i]} and {item_ids[j]}: {d:.4f}")
            if d <= distance_value_threshold:
                G.add_edge(item_ids[i], item_ids[j], weight=float(d))
                print(f"Added edge between {item_ids[i]} and {item_ids[j]} with DTW distance: {d:.4f}")

    dist_df = pd.DataFrame(dist_matrix, index=item_ids, columns=item_ids)

    print("Number of nodes in the DTW graph:", G.number_of_nodes())
    print("Number of edges in the DTW graph:", G.number_of_edges())

    return G, dist_df, df_scaled



def build_cid_graph(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    use_log1p: bool = False,
    distance_value_threshold: float | None = None,
    self_loop: bool = False,
    eps: float = 1e-12,
):

    df_pivot = (
        df.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc,
        )
        .sort_index()
        .fillna(0)
    )
    if use_log1p:
        if (df_pivot < 0).any().any():
            raise ValueError("use_log1p=True requires non-negative target values")
        df_pivot = np.log1p(df_pivot)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_values = scaler.fit_transform(df_pivot)

    df_scaled = pd.DataFrame(
        scaled_values,
        index=df_pivot.index,
        columns=df_pivot.columns,
    )

    item_ids = df_scaled.columns.tolist()

    X = df_scaled.T.to_numpy(dtype=float)   
    n_items = X.shape[0]

    ed_condensed = pdist(X, metric="euclidean")
    ed_matrix = squareform(ed_condensed)

    ce = np.sqrt(np.sum(np.diff(X, axis=1) ** 2, axis=1))   

    ce_safe = np.maximum(ce, eps)

    ce_max = np.maximum.outer(ce_safe, ce_safe)
    ce_min = np.minimum.outer(ce_safe, ce_safe)
    cf_matrix = ce_max / ce_min

    cid_matrix = ed_matrix * cf_matrix
    np.fill_diagonal(cid_matrix, 0.0)

    cid_df = pd.DataFrame(cid_matrix, index=item_ids, columns=item_ids)

    G = nx.Graph(name="CID_Graph")
    G.add_nodes_from(item_ids)

    threshold = distance_value_threshold

    if threshold is not None:
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if cid_matrix[i, j] <= threshold:
                    G.add_edge(item_ids[i], item_ids[j], weight=float(cid_matrix[i, j]))
                    print(f"Added edge between {item_ids[i]} and {item_ids[j]} with CID distance: {cid_matrix[i, j]:.4f}")
    
    print("Number of nodes in the CID graph:", G.number_of_nodes())
    print("Number of edges in the CID graph:", G.number_of_edges())
    if self_loop:
        for node in item_ids:
            if not G.has_node(node):
                G.add_node(node)
            G.add_edge(node, node, weight=1.0)

    
    return G, cid_df, df_scaled