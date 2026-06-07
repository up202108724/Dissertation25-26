"""
PyG-aware dataset for the GAT + LSTM forecaster.

Mirrors the layout of ``Graph2vec_FixedThreshold/LSTM/graph2vecdataset.py``
but yields a *trainable* graph (a ``torch_geometric.data.Data`` ego-graph)
instead of a precomputed Graph2Vec embedding.[cite: 1]
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


'''
# ── node feature builder ───────────────────────────────────────────────────
def _window_node_features_catch22(
    window_values: np.ndarray, catch24: bool = True
) -> np.ndarray:
    """
    catch22 (CAnonical Time-series CHaracteristics) node features.

    window_values : (n_nodes, window_size) raw values per node over the window
    catch24       : when True append DN_Mean + DN_Spread_Std (24 features),
                    restoring the absolute-scale information that the 22 shape
                    features discard via catch22's internal z-normalisation.
    Returns       : (n_nodes, 24) if catch24 else (n_nodes, 22)

    catch22 strictly z-normalises each series before extracting shape/dynamics
    features (autocorrelation, entropy, linearity, …), so scale is lost unless
    catch24 is used.  Short or flat windows (e.g. a 30-day run of zeros — common
    in erratic retail data) make several features ill-defined and return NaN;
    those are replaced with 0.0 so a single degenerate node cannot poison the
    whole graph during GCN message passing.
    """
    if window_values.ndim != 2:
        raise ValueError("window_values must be 2D (n_nodes, window_size)")


    n_nodes = window_values.shape[0]
    width = _CATCH24_WIDTH if catch24 else _CATCH22_WIDTH
    feats = np.zeros((n_nodes, width), dtype=np.float32)
    for i in range(n_nodes):
        ts = np.asarray(window_values[i], dtype=np.float64).tolist()   # C backend wants a list
        try:
            vals = pycatch22.catch22_all(ts, catch24=catch24)["values"]
            feats[i] = np.asarray(vals, dtype=np.float32)
        except Exception:
            # flat / constant series can crash the C backend → leave zeros
            feats[i] = 0.0
    # NaN/Inf guard (critical on short 30-day windows, see docstring)
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feats
'''


def _window_node_features_raw_values(window_values: np.ndarray) -> np.ndarray:
    """
    window_values : (n_nodes, window_size) raw values per node over the window
    Returns       : (n_nodes, window_size) — the raw sequence itself.
    """
    if window_values.ndim != 2:
        raise ValueError("window_values must be 2D (n_nodes, window_size)")
    return window_values.astype(np.float32)

def _window_node_features(window_values: np.ndarray) -> np.ndarray:
    """
    window_values : (n_nodes, window_size) raw values per node over the window
    Returns       : (n_nodes, 8) feature matrix
                    [mean, std, min, max, first, last, slope, sum]
    """
    if window_values.ndim != 2:
        raise ValueError("window_values must be 2D (n_nodes, window_size)")

    mean  = window_values.mean(axis=1)
    std   = window_values.std(axis=1)
    mn    = window_values.min(axis=1)
    mx    = window_values.max(axis=1)
    first = window_values[:, 0]
    last  = window_values[:, -1]
    slope = (last - first) / max(window_values.shape[1] - 1, 1)
    s     = window_values.sum(axis=1)

    feats = np.stack([mean, std, mn, mx, first, last, slope, s], axis=1)
    return feats.astype(np.float32)

def _window_node_features_hybrid(window_values: np.ndarray) -> np.ndarray:
    # 1. Get the 8 basic stats
    mean  = window_values.mean(axis=1)
    std   = window_values.std(axis=1)
    mn    = window_values.min(axis=1)
    mx    = window_values.max(axis=1)
    first = window_values[:, 0]
    last  = window_values[:, -1]
    slope = (last - first) / max(window_values.shape[1] - 1, 1)
    s     = window_values.sum(axis=1)
    
    stats = np.stack([mean, std, mn, mx, first, last, slope, s], axis=1)
    
    # 2. Get just the last 7 days (short enough to not confuse the GNN Linear layer)
    recent_7_days = window_values[:, -7:]
    
    # 3. Combine: 8 stats + 7 raw days = 15 robust features
    feats = np.concatenate([stats, recent_7_days], axis=1)
    return feats.astype(np.float32)


def build_pyg_graphs_from_nx_windows(
    graphs,
    df_wide,
    product_id,
    window_size: int,
    step_size: int = 1,
    max_neighbours: Optional[int] = None,
):
    """
    Convert the list of per-window NetworkX graphs into the list of
    per-window PyG ego-graphs consumed by ``GATTimeSeriesDataset``.
    """
    if product_id not in df_wide.index:
        raise KeyError(f"product_id {product_id} not in df_wide.index")

    values_full = df_wide.values            # (n_items, T)
    label_to_row = {lbl: i for i, lbl in enumerate(df_wide.index)}

    pyg_list = []
    for i, G in enumerate(graphs):
        if product_id in G:
            nbrs = list(G.neighbors(product_id))
            if max_neighbours is not None and len(nbrs) > max_neighbours:
                nbrs = sorted(
                    nbrs,
                    key=lambda u: float(G[product_id][u].get("weight", 0.0)),
                    reverse=True,
                )[:max_neighbours]
        else:
            nbrs = []

        node_order = [product_id] + [n for n in nbrs if n in label_to_row]

        t0 = i * step_size
        t1 = t0 + window_size
        if t1 > values_full.shape[1]:
            t1 = values_full.shape[1]
            t0 = max(t1 - window_size, 0)
        rows = [label_to_row[lbl] for lbl in node_order]
        window_values = values_full[rows, t0:t1]

        H = G.subgraph(node_order).copy() if product_id in G else G.__class__()
        H.add_nodes_from(node_order)
        pyg_list.append(nx_window_to_pyg(H, node_order, window_values, product_id))

    return pyg_list


def nx_window_to_pyg(
    G,
    node_order: Sequence,
    window_values: np.ndarray,
    target_node,
) -> Data:
    if node_order[0] != target_node:
        raise ValueError("target_node must be node_order[0]")

    n = len(node_order)
    label_to_idx = {lbl: i for i, lbl in enumerate(node_order)}

    src, dst, w = [], [], []
    for u, v, data in G.edges(data=True):
        if u in label_to_idx and v in label_to_idx:
            iu, iv = label_to_idx[u], label_to_idx[v]
            weight = float(data.get("weight", 1.0))
            src += [iu, iv]
            dst += [iv, iu]
            w   += [weight, weight]

    if not src:           
        src, dst, w = [0], [0], [1.0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)
    x          = torch.from_numpy(_window_node_features(window_values))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


# ── dataset ─────────────────────────────────────────────────────────────────
class GATTimeSeriesDataset(Dataset):
    """
    Drop-in analogue of ``TimeSeriesDataset`` (Graph2Vec variant) that
    feeds a trainable GAT instead of frozen embeddings.
    """

    def __init__(
        self,
        target_data: np.ndarray,
        exog_data: Optional[np.ndarray],
        seq_length: int,
        pyg_graphs: List[Data],
        graph_window_size: int = 15,
        target_node_idx: int = 0,
    ):
        self.target_data       = np.asarray(target_data, dtype=np.float32)
        self.exog_data         = (
            None if exog_data is None else np.asarray(exog_data, dtype=np.float32)
        )
        self.seq_length        = int(seq_length)
        self.graph_window_size = int(graph_window_size)
        self.target_node_idx   = int(target_node_idx)
        self.has_exog          = self.exog_data is not None

        T = len(self.target_data)
        if len(pyg_graphs) < T:
            # As in the Graph2Vec dataset we zero-pad the first graph_window_size days[cite: 1]
            sample = pyg_graphs[0]
            n_nodes  = sample.num_nodes
            in_feats = sample.x.shape[1]
            pad = Data(
                x=torch.zeros(n_nodes, in_feats, dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                edge_attr=torch.zeros(1, 1, dtype=torch.float32),
                num_nodes=n_nodes,
            )
            self.pyg_graphs = [pad] * self.graph_window_size + list(pyg_graphs)
        else:
            self.pyg_graphs = list(pyg_graphs)

        if len(self.pyg_graphs) < T:
            raise ValueError(
                f"After padding, pyg_graphs has {len(self.pyg_graphs)} entries "
                f"but target_data has {T}."
            )

    def __len__(self):
        return len(self.target_data) - self.seq_length

    def __getitem__(self, idx):
        target_seq = self.target_data[idx : idx + self.seq_length]
        y          = self.target_data[idx + self.seq_length]

        if self.has_exog:
            exog_seq = self.exog_data[idx + 1 : idx + self.seq_length + 1]
            ts = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            ts = target_seq.reshape(-1, 1)

        L   = self.seq_length
        end = len(self.pyg_graphs) - 1
        graphs = [
            self.pyg_graphs[min(max(idx + 1 + i, 0), end)]
            for i in range(L)
        ]

        ts_tensor = torch.from_numpy(ts.astype(np.float32))
        y_tensor  = torch.tensor([y], dtype=torch.float32)
        return graphs, ts_tensor, y_tensor


# ── collate ────────────────────────────────────────────────────────────────
def collate_pyg_ts(batch):
    graph_lists, ts_seqs, ys = zip(*batch)
    L = len(graph_lists[0])
    flat_graphs = [g for gl in graph_lists for g in gl]   # B*L Data objects
    pyg_batch   = Batch.from_data_list(flat_graphs)
    ts_batch    = torch.stack(ts_seqs, dim=0)
    y_batch     = torch.stack(ys, dim=0)
    target_idx  = pyg_batch.ptr[:-1]                      # (B*L,)
    return pyg_batch, ts_batch, y_batch, target_idx, L