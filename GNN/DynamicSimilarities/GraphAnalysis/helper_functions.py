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
    

def compute_correlation_matrix_gpu(df_pivot: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Computes correlation matrix on GPU using PyTorch for massive speedups."""
    if not torch.cuda.is_available():
        print(f"CUDA not available. Falling back to pandas/CPU for {method} correlation.")
        return df_pivot.corr(method=method)

    device = torch.device('cuda')
    X = torch.tensor(df_pivot.values, dtype=torch.float32, device=device)
    
    if method == "pearson":
        corr_tensor = torch.corrcoef(X.T)
        
    elif method == "spearman":
        # Fast GPU rank approximation (without tie averaging)
        _, indices = torch.sort(X, dim=0)
        ranks = torch.empty_like(X)
        ranks.scatter_(0, indices, torch.arange(1, X.shape[0]+1, dtype=torch.float32, device=device).unsqueeze(1).expand_as(X))
        corr_tensor = torch.corrcoef(ranks.T)
        
    elif method == "kendall":
        # Kendall Tau-b formula using pairwise signs on GPU
        M, N = X.shape
        if M < 2:
            corr_tensor = torch.eye(N, device=device)
        else:
            # Get all combinations of row indices i < j
            idx1, idx2 = torch.triu_indices(M, M, offset=1, device=device)
            # Find pairwise differences and directions (signs)
            signs = torch.sign(X[idx1] - X[idx2]) # Shape: [num_pairs, num_items]
            
            # S matrix (Concordant - Discordant pairs)
            # Dot product of signs between items calculates matching vs mismatching pairs 
            S = torch.mm(signs.T, signs) 
            
            # Number of non-tied pairs for each item
            non_ties = torch.sum(signs ** 2, dim=0) 
            # Kendall Tau-b denominator
            denom = torch.sqrt(torch.outer(non_ties, non_ties))
            
            corr_tensor = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
            corr_tensor.fill_diagonal_(1.0)
            
    elif method == "theil-sen":
        M, N = X.shape
        corr_tensor = torch.eye(N, device=device)
        if M >= 2:
            idx1, idx2 = torch.triu_indices(M, M, offset=1, device=device)
            # diffs: [num_pairs, N]
            diffs = X[idx1] - X[idx2]
            
            # To avoid OOM for large N and large M, we can process row by row (or item by item)
            # since we need slope = median(diffs[:, v] / diffs[:, u]) for all u, v
            slopes_matrix = torch.empty((N, N), dtype=torch.float32, device=device)
            
            for u in range(N):
                denom = diffs[:, u]
                valid = (denom != 0)
                if not valid.any():
                    slopes_matrix[u, :] = 0.0
                    continue
                
                # [num_valid_pairs, N]
                valid_diffs_v = diffs[valid]
                valid_denom = denom[valid].unsqueeze(1)
                
                slopes = (valid_diffs_v / valid_denom).median(dim=0).values
                slopes_matrix[u, :] = slopes
                
            # Make the matrix symmetric somehow, for instance, by averaging it or just taking the absolute slope proxy
            # Given similarity/correlation bounds conceptually the same, wait Theil-Sen isn't strictly bounded to [-1, 1].
            # Just return the raw slopes matrix (maybe bounded if mapped to a correlation concept) or symmetricize it.
            # To just output distances/similarity as TheilSen slope from u to v:
            corr_tensor = slopes_matrix
            corr_tensor.fill_diagonal_(1.0)

    else:
        return df_pivot.corr(method=method)

    # Convert back to pandas DataFrame matching df_pivot columns
    return pd.DataFrame(corr_tensor.cpu().numpy(), index=df_pivot.columns, columns=df_pivot.columns)

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

def compute_temporal_similarities(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    similarity_method: str = "pearson"
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
        .fillna(0)
    )

    # 2) Compute item-item similarity matrix
    print(f"Computing {similarity_method} similarity matrix...")
    sim_df = compute_correlation_matrix_gpu(df_pivot, method=similarity_method)
    print(f"Finishing computing similarity matrix. Shape: {sim_df.shape}")
    
    return sim_df, df_pivot

def build_graph_from_matrix(
    sim_df: pd.DataFrame,
    similarity_method: str = "pearson",
    similarity_threshold: float = 0.7,
    k: int = None,
    use_absolute_similarity: bool = False,
):
    item_ids = sim_df.columns.tolist()
    
    # Build graph
    G = nx.Graph(name=f"{similarity_method.capitalize()}_Similarity_Graph")
    G.add_nodes_from(item_ids)

    n = len(item_ids)

    if k is not None:
        # k-NN graph: connect each node to top-k most similar neighbors
        for i in range(n):
            sim_row = sim_df.iloc[i].copy()
            sim_row.iloc[i] = np.nan  # remove self-correlation

            if use_absolute_similarity:
                top_k_neighbors = sim_row.abs().nlargest(k).dropna()
            else:
                top_k_neighbors = sim_row.nlargest(k).dropna()

            for neighbor_id, sim_value in top_k_neighbors.items():
                j = sim_df.columns.get_loc(neighbor_id)

                edge_weight = abs(sim_value) if use_absolute_similarity else sim_value

                if use_absolute_similarity or sim_value >= similarity_threshold:
                    G.add_edge(
                        item_ids[i],
                        neighbor_id,
                        weight=float(edge_weight),
                        similarity=float(sim_value)
                    )

    else:
        # Threshold graph
        for i in range(n):
            for j in range(i + 1, n):
                sim_value = sim_df.iloc[i, j]

                if pd.isna(sim_value):
                    continue

                edge_weight = abs(sim_value) if use_absolute_similarity else sim_value

                condition = (
                    abs(sim_value) >= similarity_threshold
                    if use_absolute_similarity
                    else sim_value >= similarity_threshold
                )

                if condition:
                    G.add_edge(
                        item_ids[i],
                        item_ids[j],
                        weight=float(edge_weight),
                        similarity=float(sim_value)
                    )

    print(f"Number of nodes in the {similarity_method} graph:", G.number_of_nodes())
    print(f"Number of edges in the {similarity_method} graph:", G.number_of_edges())
    
    return G