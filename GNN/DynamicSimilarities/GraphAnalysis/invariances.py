import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import dtw
import networkx as nx

def compute_distance_matrix_gpu(X: np.ndarray, metric: str = 'euclidean', eps: float = 1e-12) -> np.ndarray:
    """Computes pairwise distance matrix on GPU using PyTorch for massive speedups."""
    if not torch.cuda.is_available():
        print(f"CUDA not available. Falling back to scipy for {metric} distance.")
        if metric == 'euclidean':
            return squareform(pdist(X, metric="euclidean"))
        elif metric == 'hamming':
            return squareform(pdist((X > 0).astype(int), metric="hamming"))
        elif metric == 'amplitude_offset':
            # Z-normalize each time series (row) then compute Euclidean distance
            X_norm = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + eps)
            return squareform(pdist(X_norm, metric="euclidean"))
        elif metric == 'slope_consistency':
            # Min-Max normalize
            X_min = np.min(X, axis=1, keepdims=True)
            X_max = np.max(X, axis=1, keepdims=True)
            X_norm = (X - X_min) / (X_max - X_min + eps)
            # Variance of the residual difference
            return squareform(pdist(X_norm, metric=lambda u, v: np.var(u - v)))
        elif metric == 'cid':
            ed_matrix = squareform(pdist(X, metric="euclidean"))
            ce = np.sqrt(np.sum(np.diff(X, axis=1) ** 2, axis=1))
            ce_safe = np.maximum(ce, eps)
            ce_max = np.maximum.outer(ce_safe, ce_safe)
            ce_min = np.minimum.outer(ce_safe, ce_safe)
            cf_matrix = ce_max / ce_min
            cid_matrix = ed_matrix * cf_matrix
            np.fill_diagonal(cid_matrix, 0.0)
            return cid_matrix
        
    device = torch.device('cuda')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    
    if metric == 'euclidean':
        dist_matrix = torch.cdist(X_t, X_t, p=2)
        return dist_matrix.cpu().numpy()
        
    elif metric == 'hamming':
        X_bin = (X_t > 0).float()
        # Hamming distance is the proportion of non-matching values
        # We can compute mismatches via element-wise XOR or absolute differences
        diffs = torch.abs(X_bin.unsqueeze(1) - X_bin.unsqueeze(0))
        dist_matrix = torch.mean(diffs, dim=2)
        return dist_matrix.cpu().numpy()
        
    elif metric == 'amplitude_offset':
        # Z-normalize each time series (subtract mean, divide by standard deviation)
        X_mean = torch.mean(X_t, dim=1, keepdim=True)
        # using unbiased standard deviation
        X_std = torch.std(X_t, dim=1, keepdim=True) + eps
        X_norm = (X_t - X_mean) / X_std
        dist_matrix = torch.cdist(X_norm, X_norm, p=2)
        return dist_matrix.cpu().numpy()
        
    elif metric == 'slope_consistency':
        # Min-Max normalize each time series [0, 1]
        X_min = torch.min(X_t, dim=1, keepdim=True)[0]
        X_max = torch.max(X_t, dim=1, keepdim=True)[0]
        X_norm = (X_t - X_min) / (X_max - X_min + eps)
        
        # Calculate pairwise differences (residuals) and compute their variance
        diffs = X_norm.unsqueeze(1) - X_norm.unsqueeze(0)
        dist_matrix = torch.var(diffs, dim=2, unbiased=False)
        return dist_matrix.cpu().numpy()
        
    elif metric == 'cid':
        ed_matrix = torch.cdist(X_t, X_t, p=2)
        
        # Complexity estimation
        ce = torch.sqrt(torch.sum(torch.diff(X_t, dim=1) ** 2, dim=1))
        ce_safe = torch.maximum(ce, torch.tensor(eps, device=device))
        
        # Pairwise complexity correction factor
        ce_max = torch.maximum(ce_safe.unsqueeze(0), ce_safe.unsqueeze(1))
        ce_min = torch.minimum(ce_safe.unsqueeze(0), ce_safe.unsqueeze(1))
        cf_matrix = ce_max / ce_min
        
        cid_matrix = ed_matrix * cf_matrix
        cid_matrix.fill_diagonal_(0.0)
        return cid_matrix.cpu().numpy()
        
    else:
        raise ValueError(f"Metric {metric} not supported by compute_distance_matrix_gpu")

def compute_temporal_distances(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    distance_method: str = "euclidean", # 'euclidean', 'dtw', 'cid', 'hamming', 'amplitude_offset', 'slope_consistency'
    aggfunc: str = "sum",
    use_log1p: bool = False,
    sakoe_chiba_radius: int = None,
    eps: float = 1e-12,
):
    df_pivot = (
        df.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc
        )
        .sort_index()
    )
    
    if distance_method in ["euclidean", "amplitude_offset", "slope_consistency"]:
        df_pivot = df_pivot.ffill().fillna(0)
    else:
        df_pivot = df_pivot.fillna(0)
        
    if use_log1p and distance_method in ["dtw", "cid", "amplitude_offset", "slope_consistency"]:
        if (df_pivot < 0).any().any():
            raise ValueError("use_log1p=True requires non-negative target values")
        df_pivot = np.log1p(df_pivot)
        
    # Scaling
    if distance_method in ["euclidean", "dtw", "cid"]:
        scaler = StandardScaler(with_mean=True, with_std=True)
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_pivot),
            index=df_pivot.index,
            columns=df_pivot.columns
        )
        X = df_scaled.T.values
    else: # Hamming, amplitude_offset, slope_consistency (do their own normalization)
        df_scaled = df_pivot
        X = df_pivot.T.values

    item_ids = df_scaled.columns.tolist()
    n = len(item_ids)
    
    # Distance Matrix Computation
    print(f"Computing {distance_method} distance matrix...")
    if distance_method == "dtw":
        dist_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if sakoe_chiba_radius is None:
                    d = dtw(X[i], X[j])
                else:
                    d = dtw(X[i], X[j], global_constraint="sakoe_chiba",
                            sakoe_chiba_radius=sakoe_chiba_radius)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    else:
        dist_matrix = compute_distance_matrix_gpu(X, metric=distance_method, eps=eps)

    dist_df = pd.DataFrame(dist_matrix, index=item_ids, columns=item_ids)
    print(f"Finishing computing distance matrix. Shape: {dist_df.shape}")
    
    return dist_df, df_scaled

def build_graph_from_distance_matrix(
    dist_df: pd.DataFrame,
    distance_method: str = "euclidean",
    distance_value_threshold: float = 1.0,
    k: int = None,
    self_loop: bool = False,
):
    item_ids = dist_df.columns.tolist()
    n = len(item_ids)
    dist_matrix = dist_df.values

    # Graph construction
    G = nx.Graph(name=f"{distance_method.capitalize()}_Distance_Graph")
    G.add_nodes_from(item_ids)

    # Apply thresholding or KNN
    if k is not None:
        for i in range(n):
            # For distance, we want the k SMALLEST distances
            dist_row = pd.Series(dist_matrix[i, :], index=item_ids)
            dist_row[item_ids[i]] = np.nan # Ignore self
            top_k_neighbors = dist_row.nsmallest(k).dropna()
            
            for neighbor_id, dist_value in top_k_neighbors.items():
                if dist_value <= distance_value_threshold:
                    G.add_edge(item_ids[i], neighbor_id, weight=float(dist_value))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] <= distance_value_threshold:
                    G.add_edge(item_ids[i], item_ids[j], weight=float(dist_matrix[i, j]))
                
    if self_loop:
        for node in item_ids:
            if not G.has_edge(node, node):
                G.add_edge(node, node, weight=0.0)
                
    print(f"Number of nodes in the {distance_method} graph:", G.number_of_nodes())
    print(f"Number of edges in the {distance_method} graph:", G.number_of_edges())

    return G