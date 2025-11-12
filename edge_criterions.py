import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import Callable, Dict, Tuple, Optional
from itertools import combinations


def edges_equal_units(
    g: pd.DataFrame,
    item_to_local: Dict[int, int],
    return_weights: bool = False,
    weight_mode: str = "one",   # "one" | "value" | "group_size"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Create edges between items that sold the EXACT SAME number of units today.

    Parameters
    ----------
    g : pd.DataFrame
        Per-day slice with at least ["item_id", "value"].
    item_to_local : dict
        Maps item_id -> local node index for this day.
    return_weights : bool
        If True, returns edge_weight aligned with edge_index.
    weight_mode : {"one","value","group_size"}
        - "one":        every edge gets weight 1.0
        - "value":      weight equals the common units sold (v)
        - "group_size": weight equals the size of the clique (number of items with that v)

    Returns
    -------
    edge_index : LongTensor [2, E]
    edge_weight : FloatTensor [E] or None
    """
    edge_list = []
    weights = [] if return_weights else None

    # group items that have the same 'value'
    for v, sub in g.groupby("value"):
        ids = sub["item_id"].tolist()
        if len(ids) < 2:
            continue

        locs = [item_to_local[i] for i in ids]
        # choose edge weight per group if requested
        if return_weights:
            if weight_mode == "one":
                w_group = 1.0
            elif weight_mode == "value":
                w_group = float(v)
            elif weight_mode == "group_size":
                w_group = float(len(locs))
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

        # fully connect the group (undirected -> add both directions)
        for i, j in combinations(locs, 2):
            edge_list.append((i, j))
            edge_list.append((j, i))
            if return_weights:
                weights.extend([w_group, w_group])

    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), None

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_weight = torch.tensor(weights, dtype=torch.float32) if return_weights else None
    return edge_index, edge_weight


def edges_positive_pct_change(
    g: pd.DataFrame,
    item_to_local: Dict[int, int],
    threshold: float = 0.0,   # e.g., 0.0 for >0%, 0.05 for ≥5%
    eps: float = 1e-6,
    return_weights: bool = False,
    weight_agg: str = "min"   # "min" | "mean" | "max"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Create edges between items that had a positive (or >= threshold) *relative*
    change vs previous value on this day.

    Requires g to have columns: ["item_id", "value", "prev_value"] for the day.

    Parameters
    ----------
    threshold : float
        Minimum relative change required to include an item. 0.0 means just positive.
        0.05 means ≥ +5%, etc.
    eps : float
        Small constant to avoid division by zero when prev_value == 0.
    return_weights : bool
        If True, returns an edge_weight vector based on the two nodes' pct changes.
    weight_agg : {"min","mean","max"}
        How to combine the two endpoints' pct changes into an edge weight.
    """
    # Compute relative change safely
    prev = g["prev_value"].to_numpy(dtype=float)
    curr = g["value"].to_numpy(dtype=float)
    denom = np.maximum(np.abs(prev), eps)  # avoid div-by-zero; uses eps when prev==0
    pct = (curr - prev) / denom

    # Mask: previous exists AND relative change ≥ threshold
    mask = (~np.isnan(prev)) & (pct >= threshold)
    if mask.sum() < 2:
        return torch.empty((2, 0), dtype=torch.long), None

    # Keep qualifying items
    qual = g.loc[mask, ["item_id"]].copy()
    qual["pct"] = pct[mask]

    # Map to local node indices
    locs = [item_to_local[i] for i in qual["item_id"].tolist()]
    pcts = qual["pct"].to_numpy()

    # Build edges (undirected → store both directions)
    edge_list = []
    weights = []  # only if return_weights=True
    for idx_a, idx_b in combinations(range(len(locs)), 2):
        a, b = locs[idx_a], locs[idx_b]
        edge_list.extend([(a, b), (b, a)])

        if return_weights:
            pa, pb = pcts[idx_a], pcts[idx_b]
            if weight_agg == "min":
                w = min(pa, pb)
            elif weight_agg == "max":
                w = max(pa, pb)
            else:  # "mean"
                w = 0.5 * (pa + pb)
            weights.extend([w, w])

    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), None

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_weight = torch.tensor(weights, dtype=torch.float32) if return_weights else None
    return edge_index, edge_weight
