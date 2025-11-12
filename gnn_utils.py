# gnn_utils.py
from typing import Callable, Dict, Optional
import pandas as pd
import torch
from torch_geometric.data import Data
from wrappers import add_self_loops_and_norm
EdgeFn = Callable[[pd.DataFrame, Dict[int, int]], tuple]  # (edge_index, edge_weight|None)

def prepare_daily_with_prev(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    # collapse to daily totals (safe even if already unique)
    daily = dfx.groupby(["date", "item_id"], as_index=False)["value"].sum()
    # compute previous observation per item
    daily = daily.sort_values(["item_id", "date"])
    daily["prev_value"] = daily.groupby("item_id")["value"].shift(1)
    return daily


def build_daily_graphs(df: pd.DataFrame,
                       edge_fn,
                       node_feat_fn=None,
                       normalize: bool = True,
                       add_self_loops: bool = True,
                       edge_fn_kwargs: dict | None = None):
    """
    self_loops_mode:
      - None:        no self-loops
      - 'zero_weight': add self-loops with weight=0 (keeps correlation weights intact)
      - 'unit_norm': add unit self-loops and apply GCN normalization to edge weights
    """
    edge_fn_kwargs = edge_fn_kwargs or {}

    daily = prepare_daily_with_prev(df)

    graphs_by_date = {}
    for day, g in daily.groupby("date", sort=True):
        # g now has columns: date, item_id, value, prev_value
        items = g["item_id"].to_numpy()
        item_to_local = {itm: i for i, itm in enumerate(items)}
        num_nodes = len(items)

        # Node features (default = today's value only)
        if node_feat_fn is None:
            x = torch.tensor(g["value"].to_numpy().reshape(-1, 1), dtype=torch.float32)
        else:
            x = node_feat_fn(g)  # must align with g's row order

        # Edges (+ optional weights) from your pluggable criterion
        edge_index, edge_weight = edge_fn(g, item_to_local, **edge_fn_kwargs)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Self-loops handling
        if add_self_loops:
            edge_index, edge_weight = add_self_loops_and_norm(
                edge_index,
                edge_weight,
                num_nodes,
                add_self_loops=add_self_loops,
                normalize=normalize,
            )
        # Package graph
        data = Data(
            x=x,
            edge_index=edge_index,
            item_id=torch.tensor(items, dtype=torch.long),
            num_nodes=num_nodes,
        )
        if edge_weight is not None:
            data.edge_weight = edge_weight
        data.date_str = str(day.date())
        graphs_by_date[day] = data

    return graphs_by_date
