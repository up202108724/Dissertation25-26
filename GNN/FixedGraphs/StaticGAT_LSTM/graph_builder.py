"""
graph_builder.py — Build a static global similarity graph from training data.

The graph is computed ONCE before training begins:
  • All items are nodes.
  • An edge (i, j) is added when sim(i, j) ≥ threshold.
  • Edge weight = similarity value.
  • Node features are 8 statistical summaries of each item's training series.

After construction the (global_x, edge_index, edge_weight) tensors are
moved to the target device and kept there for the entire training run.
"""

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 8   # must match compute_node_features_static output length


def compute_node_features_static(ts: np.ndarray) -> np.ndarray:
    """
    Eight statistical features computed from an item's FULL training series.

    Returns a float32 array of length NODE_FEAT_DIM (= 8):
        [mean_all, std_all, zero_ratio, slope, min_v, max_v, mean7, last_val]
    """
    ts = np.asarray(ts, dtype=np.float32)
    T  = len(ts)
    if T == 0:
        return np.zeros(NODE_FEAT_DIM, dtype=np.float32)

    mean_all   = float(np.mean(ts))
    std_all    = float(np.std(ts))
    zero_ratio = float(np.mean(ts == 0))
    slope      = float(np.polyfit(np.arange(T), ts, 1)[0]) if T > 1 else 0.0
    min_v      = float(np.min(ts))
    max_v      = float(np.max(ts))
    mean7      = float(np.mean(ts[-7:])) if T >= 7 else mean_all
    last_val   = float(ts[-1])

    return np.array([mean_all, std_all, zero_ratio, slope, min_v, max_v, mean7, last_val],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Pairwise similarity (vectorised, GPU-accelerated when available)
# ---------------------------------------------------------------------------

def _spearman_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Full N×N Spearman correlation matrix for the rows of `matrix` (N, T).
    Uses GPU if available; falls back to CPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(matrix, dtype=torch.float32, device=device)   # (N, T)

    # Rank each row
    _, idx = torch.sort(X, dim=1)
    ranks  = torch.empty_like(X)
    ranks.scatter_(
        1, idx,
        torch.arange(1, X.shape[1] + 1, dtype=torch.float32, device=device)
            .unsqueeze(0).expand_as(X),
    )

    # Pearson correlation on ranks == Spearman correlation
    ranks  = ranks - ranks.mean(dim=1, keepdim=True)
    norms  = torch.norm(ranks, dim=1, keepdim=True)
    normed = ranks / (norms + eps)
    sim    = normed @ normed.t()   # (N, N)

    return sim.cpu().numpy()


def _pearson_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Full N×N Pearson correlation matrix."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X  = torch.tensor(matrix, dtype=torch.float32, device=device)
    Xc = X - X.mean(dim=1, keepdim=True)
    norms  = torch.norm(Xc, dim=1, keepdim=True)
    normed = Xc / (norms + eps)
    sim    = normed @ normed.t()
    return sim.cpu().numpy()


def _sim_1vsall(
    target_row: np.ndarray,
    matrix: np.ndarray,
    metric: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute similarity between target_row (T,) and every row of matrix (N, T).
    Returns shape (N,).  Supports 'spearman' and 'pearson'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_row, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T)
    X      = torch.tensor(matrix,     dtype=torch.float32, device=device)               # (N, T)

    if metric == 'spearman':
        def _rank(m):
            _, idx   = torch.sort(m, dim=1)
            ranks    = torch.empty_like(m)
            ranks.scatter_(
                1, idx,
                torch.arange(1, m.shape[1] + 1, dtype=torch.float32, device=device)
                    .unsqueeze(0).expand_as(m),
            )
            return ranks
        target = _rank(target)
        X      = _rank(X)

    # Pearson / Spearman-on-ranks: normalised dot product
    tc = target - target.mean(dim=1, keepdim=True)
    Xc = X      - X.mean(dim=1, keepdim=True)
    tn = tc / (torch.norm(tc, dim=1, keepdim=True) + eps)
    Xn = Xc / (torch.norm(Xc, dim=1, keepdim=True) + eps)
    sim = (Xn @ tn.t()).squeeze(-1)   # (N,)
    return sim.cpu().numpy()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_static_graph(
    df_wide,                    # pd.DataFrame rows=items, cols=dates
    item_ids,                   # ordered list/array — defines node order
    metric: str = 'spearman',
    threshold: float = 0.82,
    train_end_idx: int = None,  # use only the first train_end_idx date-columns
) -> tuple:
    """
    Build a global PyG Data object for the similarity graph.

    Parameters
    ----------
    df_wide        : DataFrame (items × dates) with raw sales values.
    item_ids       : sequence of item IDs in the order that defines node indices.
    metric         : 'spearman' | 'pearson'
    threshold      : minimum similarity to include an edge
    train_end_idx  : if given, only columns [:train_end_idx] are used for
                     similarity computation (prevents data leakage).

    Returns
    -------
    graph_data   : PyG Data(x, edge_index, edge_weight)
                     x            – (N, 8)   static node features
                     edge_index   – (2, E)   undirected edges
                     edge_weight  – (E,)     similarity weights
    item_to_node : dict  {item_id: node_index}
    """
    item_ids = list(item_ids)
    N = len(item_ids)

    # --- 1. Extract training-period matrix --------------------------------
    if train_end_idx is not None:
        ts_matrix = df_wide.iloc[:, :train_end_idx].values.astype(np.float32)
    else:
        ts_matrix = df_wide.values.astype(np.float32)   # (N, T)

    # Mark active items (avoid zero-demand items adding noise to edges)
    active_mask = np.sum(np.abs(ts_matrix), axis=1) > 0

    # --- 2. Node features -------------------------------------------------
    x = np.stack(
        [compute_node_features_static(ts_matrix[i]) for i in range(N)],
        axis=0,
    )  # (N, 8)

    # --- 3. Pairwise similarity --------------------------------------------
    print(f"  [graph_builder] Computing {metric} similarity for {N} items "
          f"over {ts_matrix.shape[1]} training steps …")
    if metric == 'spearman':
        sim_matrix = _spearman_matrix(ts_matrix)
    elif metric == 'pearson':
        sim_matrix = _pearson_matrix(ts_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    # --- 4. Build edge list (upper triangle, then symmetrize) -------------
    src_list, dst_list, w_list = [], [], []
    for i in range(N):
        if not active_mask[i]:
            continue
        for j in range(i + 1, N):
            if not active_mask[j]:
                continue
            w = float(sim_matrix[i, j])
            if w >= threshold:
                src_list += [i, j]
                dst_list += [j, i]
                w_list   += [w, w]

    E = len(src_list)
    print(f"  [graph_builder] Edges added: {E // 2} undirected "
          f"({E} directed) at threshold={threshold}")

    if E > 0:
        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(w_list, dtype=torch.float32)
    else:
        edge_index  = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,),   dtype=torch.float32)

    graph_data = Data(
        x           = torch.tensor(x, dtype=torch.float32),
        edge_index  = edge_index,
        edge_weight = edge_weight,
    )

    item_to_node = {item_id: idx for idx, item_id in enumerate(item_ids)}
    return graph_data, item_to_node


# ---------------------------------------------------------------------------
# Ego-graph builder
# ---------------------------------------------------------------------------

def build_ego_graph(
    target_item_id,
    df_wide,                    # pd.DataFrame rows=items, cols=dates
    item_ids,                   # ordered list/array — defines row order in df_wide
    metric: str = 'spearman',
    threshold: float = 0.82,
    train_end_idx: int = None,  # use only the first train_end_idx date-columns
) -> tuple:
    """
    Build a PyG ego-graph centred on target_item_id.

    Node set  : [target] + all items with sim(target, item) >= threshold
    Edges     : target ↔ neighbor  (weight = similarity)
                neighbor ↔ neighbor if their mutual sim also >= threshold
    Node order: target is always at index 0.

    Parameters
    ----------
    target_item_id : item ID of the product being forecast
    df_wide        : DataFrame (items × dates) with raw sales values.
    item_ids       : sequence of item IDs matching the row order of df_wide.
    metric         : 'spearman' | 'pearson'
    threshold      : minimum similarity to include a node / edge
    train_end_idx  : if given, only columns [:train_end_idx] are used.

    Returns
    -------
    graph_data   : PyG Data(x, edge_index, edge_weight)
    item_to_node : dict  {item_id: node_index}  (target always maps to 0)
    """
    item_ids = list(item_ids)
    target_global_idx = item_ids.index(target_item_id)

    # --- 1. Extract training matrix ----------------------------------------
    if train_end_idx is not None:
        ts_matrix = df_wide.iloc[:, :train_end_idx].values.astype(np.float32)
    else:
        ts_matrix = df_wide.values.astype(np.float32)   # (N, T)

    active_mask = np.sum(np.abs(ts_matrix), axis=1) > 0

    # --- 2. 1-vs-all similarity: target vs every other item ----------------
    target_row = ts_matrix[target_global_idx]
    print(f"  [ego_graph] Computing {metric} similarity for target={target_item_id} "
          f"vs {len(item_ids)} items …")
    sims = _sim_1vsall(target_row, ts_matrix, metric=metric)   # (N,)

    # --- 3. Select neighbors -----------------------------------------------
    neighbor_global_idxs = [
        i for i in range(len(item_ids))
        if i != target_global_idx and active_mask[i] and sims[i] >= threshold
    ]

    # Ego-graph node order: [target, neighbor_1, …, neighbor_K]
    ego_global_idxs = [target_global_idx] + neighbor_global_idxs
    ego_item_ids    = [item_ids[i] for i in ego_global_idxs]
    item_to_node    = {item_id: local_idx for local_idx, item_id in enumerate(ego_item_ids)}
    K = len(neighbor_global_idxs)

    # --- 4. Node features for ego nodes ------------------------------------
    x = np.stack(
        [compute_node_features_static(ts_matrix[i]) for i in ego_global_idxs],
        axis=0,
    )   # (1+K, 8)

    # --- 5. Build edge list ------------------------------------------------
    src_list, dst_list, w_list = [], [], []

    # target ↔ neighbors
    for local_v, global_i in enumerate(neighbor_global_idxs, start=1):
        w = float(sims[global_i])
        src_list += [0, local_v]
        dst_list += [local_v, 0]
        w_list   += [w, w]

    # neighbor ↔ neighbor (only if they mutually exceed the threshold)
    if K > 1:
        neighbor_matrix = ts_matrix[neighbor_global_idxs]   # (K, T)
        if metric == 'spearman':
            sub_sim = _spearman_matrix(neighbor_matrix)
        else:
            sub_sim = _pearson_matrix(neighbor_matrix)

        for a in range(K):
            for b in range(a + 1, K):
                w = float(sub_sim[a, b])
                if w >= threshold:
                    la, lb = a + 1, b + 1   # +1 because target occupies index 0
                    src_list += [la, lb]
                    dst_list += [lb, la]
                    w_list   += [w, w]

    E = len(src_list)
    print(f"  [ego_graph] {1 + K} nodes | {E // 2} undirected edges "
          f"({K} neighbors) at threshold={threshold}")

    if E > 0:
        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(w_list, dtype=torch.float32)
    else:
        edge_index  = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,),   dtype=torch.float32)

    graph_data = Data(
        x           = torch.tensor(x, dtype=torch.float32),
        edge_index  = edge_index,
        edge_weight = edge_weight,
    )
    return graph_data, item_to_node
