# wrappers.py
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.utils import degree

def add_unit_self_loops(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Add missing self-loops with weight=1.0. Leaves existing edges & weights intact.

    Args:
      edge_index: [2, E] long
      edge_weight: [E] float or None
      num_nodes: int

    Returns:
      edge_index2, edge_weight2
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0

    device = edge_index.device
    # mark nodes that already have a self-loop
    has_self = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if edge_index.numel() > 0:
        mask = edge_index[0] == edge_index[1]
        if mask.any():
            has_self[edge_index[0, mask]] = True

    missing = torch.arange(num_nodes, device=device)[~has_self]
    if missing.numel() == 0:
        return edge_index, edge_weight  # nothing to add

    self_loops = torch.stack([missing, missing], dim=0)  # [2, L]
    edge_index2 = torch.cat([edge_index, self_loops], dim=1) if edge_index.numel() else self_loops

    if edge_weight is None:
        edge_weight2 = torch.ones(self_loops.shape[1], dtype=torch.float32, device=device)
    else:
        ones = torch.ones(self_loops.shape[1], dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight2 = torch.cat([edge_weight, ones], dim=0) if edge_weight.numel() else ones

    return edge_index2, edge_weight2


def gcn_normalize(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
    eps: float = 1e-12,
):
    """
    Symmetric GCN normalization:  \tilde{A} = D^{-1/2} A D^{-1/2}
    Works on old/new PyG. If edge_weight is None, assume 1 for all edges.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_weight  # nothing to normalize

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)

    row, col = edge_index[0], edge_index[1]

    # --- weighted degree: try PyG's degree(weight=...), else manual index_add_ ---
    try:
        from torch_geometric.utils import degree as pyg_degree
        deg = pyg_degree(row, num_nodes=num_nodes, dtype=edge_weight.dtype, weight=edge_weight)  # new PyG
    except TypeError:
        # old PyG: no weight= kwarg
        deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.index_add_(0, row, edge_weight)  # sum weights per source node

    deg_inv_sqrt = torch.pow(deg.clamp(min=eps), -0.5)
    norm_w = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm_w


def add_self_loops_and_norm(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    add_self_loops: bool = True,
    normalize: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Convenience wrapper:
      - optionally add unit self-loops
      - optionally GCN-normalize weights
    """
    if add_self_loops:
        edge_index, edge_weight = add_unit_self_loops(edge_index, edge_weight, num_nodes)
    if normalize:
        edge_index, edge_weight = gcn_normalize(edge_index, edge_weight, num_nodes)
    return edge_index, edge_weight
