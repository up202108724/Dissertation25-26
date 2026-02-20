# gnn_utils.py
from typing import Callable, Dict, Optional
from collections import defaultdict
import pandas as pd
import torch
from torch_geometric.data import Data
from wrappers import add_self_loops_and_norm
import numpy as np
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

def get_daily_edge_sets_item_ids(graphs_by_date, df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    value_lookup = df.groupby(['date','item_id'])['value'].sum().unstack(fill_value=0)

    edge_sets = {}
    for day, data in graphs_by_date.items():
        pairs = {}
        if data.edge_index.numel() == 0:
            edge_sets[day] = pairs
            continue

        items = data.item_id.cpu().numpy()
        ei = data.edge_index.cpu().numpy().T

        if day not in value_lookup.index:
            edge_sets[day] = pairs
            continue
        day_values = value_lookup.loc[day]

        for u, v in ei:
            a, b = int(items[u]), int(items[v])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            va = float(day_values.get(a, 0.0))
            vb = float(day_values.get(b, 0.0))
            pairs[(a,b)] = {'value_a': va, 'value_b': vb, 'value_min': min(va,vb)}
        edge_sets[day] = pairs
    return edge_sets

def get_daily_edge_sets_item_ids_fast(graphs_by_date, df: pd.DataFrame):
    """
    Faster version: precomputes numpy arrays and mappings so we avoid
    repeated pandas lookups for each edge.
    """
    # ensure datetime alignment
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Precompute lookup for (date,item_id)->value as a dense numpy matrix
    value_lookup = (
        df.groupby(["date", "item_id"])["value"]
          .sum()
          .unstack(fill_value=0)
    )
    # Row mapping: date -> row index in value_arr
    days = value_lookup.index.to_numpy()
    day_to_row = {d: i for i, d in enumerate(days)}

    # Column info: item ids for each column and a dict item_id -> col index
    col_items = value_lookup.columns.to_numpy()            # shape [M,]
    col_map = {int(it): j for j, it in enumerate(col_items)}  # item_id -> column idx

    # The dense numeric array: shape [D, M]
    value_arr = value_lookup.to_numpy()

    edge_sets = {}
    # Optional cache: map tuple(items) -> local_col array to avoid recomputing for same node ordering
    local_col_cache = {}

    for day, data in graphs_by_date.items():
        pairs = {}

        if data.edge_index.numel() == 0:
            edge_sets[day] = pairs
            continue

        if day not in day_to_row:
            edge_sets[day] = pairs
            continue

        # local mapping: local node index -> global item id
        items = data.item_id.cpu().numpy().astype(int)  # shape [N,]

        # try cache to avoid recomputing local->col many times
        items_key = tuple(items)
        local_col = local_col_cache.get(items_key)
        if local_col is None:
            # -1 means this item_id is not present in value_lookup columns -> treat as zero
            local_col = np.array([col_map.get(it, -1) for it in items], dtype=np.int32)
            local_col_cache[items_key] = local_col

        # row for this day
        row = value_arr[day_to_row[day]]  # shape [M,]

        # edges as numpy array shape [E,2]
        ei = data.edge_index.cpu().numpy().T
        u = ei[:, 0].astype(int)
        v = ei[:, 1].astype(int)

        # filter self-loops quickly
        mask = (u != v)
        if not mask.any():
            edge_sets[day] = pairs
            continue

        u = u[mask]; v = v[mask]

        # item ids for endpoints
        a_items = items[u]
        b_items = items[v]

        # column indices for endpoints
        col_u = local_col[u]
        col_v = local_col[v]

        # canonicalize unordered pair so (a,b) is always a<=b
        swap = a_items > b_items
        a_items2 = np.where(swap, b_items, a_items)
        b_items2 = np.where(swap, a_items, b_items)
        col_a = np.where(swap, col_v, col_u)
        col_b = np.where(swap, col_u, col_v)

        # fetch values (where col == -1, use 0.0)
        va = np.where(col_a >= 0, row[col_a], 0.0)
        vb = np.where(col_b >= 0, row[col_b], 0.0)
        vmin = np.minimum(va, vb)

        # populate dict (single loop over edges, minimal Python work)
        for ai, bi, vai, vbi, vmini in zip(a_items2, b_items2, va, vb, vmin):
            pairs[(int(ai), int(bi))] = {
                "value_a": float(vai),
                "value_b": float(vbi),
                "value_min": float(vmini),
            }

        edge_sets[day] = pairs

    return edge_sets

def compute_edge_persistence(edge_sets, require_consecutive_days: bool = True, min_len: int = 2) -> pd.DataFrame:
    """
    For each undirected item-id edge, compute streaks and include start/end dates.

    Parameters
    ----------
    edge_sets : dict[pd.Timestamp, dict[(int,int) -> dict]]
        Per-day undirected edges in ITEM-ID space. Values (meta dicts) are ignored.
    require_consecutive_days : bool
        If True, split into maximal runs of calendar-consecutive days.
        If False, return a single row per edge with all its days.
    min_len : int
        Minimum streak length to keep (default=2).
    """
    days_sorted = sorted(edge_sets.keys())

    # For each edge, store list of days when it appears
    occurs = defaultdict(list)
    for d in days_sorted:
        day_edges = edge_sets[d]
        for (a, b) in day_edges.keys():  # ignore meta
            if a == b:  # skip self-loops defensively
                continue
            edge = (min(a, b), max(a, b))
            occurs[edge].append(d)

    rows = []
    for edge, dlist in occurs.items():
        if not dlist:
            continue

        # sort by day
        dlist = sorted(dlist)

        if not require_consecutive_days:
            if len(dlist) >= min_len:
                rows.append({
                    "edge": edge,
                    "streak_len": len(dlist),
                    "start_day": dlist[0],
                    "end_day": dlist[-1],
                    "days": dlist,
                })
            continue

        # split into maximal consecutive runs
        run = [dlist[0]]
        for i in range(1, len(dlist)):
            if (dlist[i] - dlist[i - 1]).days == 1:
                run.append(dlist[i])
            else:
                if len(run) >= min_len:
                    rows.append({
                        "edge": edge,
                        "streak_len": len(run),
                        "start_day": run[0],
                        "end_day": run[-1],
                        "days": run.copy(),
                    })
                run = [dlist[i]]

        # flush last run
        if len(run) >= min_len:
            rows.append({
                "edge": edge,
                "streak_len": len(run),
                "start_day": run[0],
                "end_day": run[-1],
                "days": run.copy(),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["edge"].map(lambda e: e[0] != e[1])]  # extra self-loop safety
    return df


def summarize_dense_adj(A: torch.Tensor):
    """
    Compute basic statistics for a dense adjacency matrix A (torch.Tensor).
    Assumes undirected adjacency stored with both (i,j) and (j,i).
    """
    # ensure CPU and numpy-friendly
    A = A.detach().cpu().to(torch.float32)
    n = A.shape[0]
    # remove diagonal (self-loops) from counts for statistics
    A_nodiag = A.clone()
    A_nodiag.fill_diagonal_(0)
    # number of undirected edges
    total_edge_entries = A_nodiag.sum().item()           # counts both directions
    num_edges = int(total_edge_entries / 2)
    # degrees (counting each neighbor once)
    deg = A_nodiag.sum(dim=1).cpu().numpy()
    avg_deg = deg.mean()
    med_deg = np.median(deg)
    max_deg = deg.max()
    min_deg = deg.min()
    std_deg = deg.std()
    num_isolates = int((deg == 0).sum())
    possible_edges = n * (n - 1) / 2
    density = num_edges / possible_edges if possible_edges > 0 else 0.0

    hist_counts, hist_bins = np.histogram(deg, bins='auto')
    top_degrees = np.argsort(deg)[-10:][::-1]  # indices of up to 10 highest-degree nodes

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": float(avg_deg),
        "median_degree": float(med_deg),
        "min_degree": int(min_deg),
        "max_degree": int(max_deg),
        "std_degree": float(std_deg),
        "num_isolates": num_isolates,
        "degree_hist_counts": hist_counts,
        "degree_hist_bins": hist_bins,
        "top_degree_node_indices": top_degrees,
        "top_degree_values": deg[top_degrees].tolist(),
    }


