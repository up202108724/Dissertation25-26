
import pandas as pd
import numpy as np
import networkx as nx
import torch


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

