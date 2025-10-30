import torch
import pandas as pd
from torch_geometric.data import Data
from typing import Callable, Dict, Tuple, Optional
from itertools import combinations
def edges_equal_units(g: pd.DataFrame, item_to_local: Dict[int, int]):
    edge_list = []
    for v, sub in g.groupby("value"):
        locs = [item_to_local[i] for i in sub["item_id"].tolist()]
        if len(locs) >= 2:
            for i, j in combinations(locs, 2):
                edge_list.append((i, j))
                edge_list.append((j, i))
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long), None
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index, None

def edges_positive_variation(g: pd.DataFrame, item_to_local: Dict[int, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # keep items with prev_value available and positive delta
    mask = g["prev_value"].notna() & (g["value"] > g["prev_value"])
    inc_items = g.loc[mask, "item_id"].tolist()

    if len(inc_items) < 2:
        return torch.empty((2, 0), dtype=torch.long), None

    locs = [item_to_local[i] for i in inc_items]
    edge_list = []
    for a, b in combinations(locs, 2):
        edge_list.append((a, b))
        edge_list.append((b, a))  # undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index, None