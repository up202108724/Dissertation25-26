import torch
import pandas as pd
from torch_geometric.data import Data
from typing import Callable, Dict, Tuple, Optional
from itertools import combinations
from torch_geometric.utils import add_self_loops as pyg_add_self_loops

def build_daily_graphs(
    df: pd.DataFrame,
    edge_fn: Callable[[pd.DataFrame, Dict[int, int]], torch.Tensor],
    node_feat_fn: Optional[Callable[[pd.DataFrame], torch.Tensor]] = None,
    add_self_loops: bool = False
) -> Dict[pd.Timestamp, Data]:
    graphs_by_date = {}
    dates = df['date'].unique()
    
    for date in dates:
        g = df[df['date'] == date]
        item_ids = g['item_id'].unique()
        item_to_local = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        edge_index = edge_fn(g, item_to_local)
        
        if add_self_loops:
            edge_index = pyg_add_self_loops(edge_index, num_nodes=len(item_ids))
        
        if node_feat_fn:
            x = node_feat_fn(g)
        else:
            x = None
        
        data = Data(x=x, edge_index=edge_index)
        graphs_by_date[date] = data
    
    return graphs_by_date