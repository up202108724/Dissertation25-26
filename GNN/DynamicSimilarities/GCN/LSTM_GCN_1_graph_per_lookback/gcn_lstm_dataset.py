"""
PyG-aware dataset for the GCN + LSTM forecaster.

Mirrors the layout of ``Graph2vec_FixedThreshold/LSTM/graph2vecdataset.py``
but yields a *trainable* graph (a ``torch_geometric.data.Data`` ego-graph)
instead of a precomputed Graph2Vec embedding.

Per sample (idx) we return three tensors:
    pyg_graph : torch_geometric.data.Data
        Ego-graph aligned to the most recent window seen by the LSTM,
        i.e. the graph computed on days ``idx + seq_length - graph_window_size
        .. idx + seq_length - 1`` (one graph per sliding window).
    ts_seq    : FloatTensor  (seq_length, 1 + n_exog)
        [target ‖ exog] sequence consumed by the LSTM.
    y         : FloatTensor  (1,)
        Target value at step ``idx + seq_length``.

The list ``pyg_graphs`` is expected to have one entry per sliding-window
position.  As in the Graph2Vec dataset we zero-pad the first
``graph_window_size`` days so that ``pyg_graphs[t]`` represents the
graph built from the ``graph_window_size`` days preceding ``t``.

A helper ``nx_window_to_pyg`` is provided to build a ``Data`` ego-graph
from a NetworkX graph + the raw window slice (used by callers that
already have the dynamic-graph pickles from the Graph2Vec pipeline).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# ── node feature builder ───────────────────────────────────────────────────
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


def build_pyg_graphs_from_nx_windows(
    graphs,
    df_wide,
    product_id,
    window_size: int,
    step_size: int = 1,
    max_neighbours: Optional[int] = None,
):
    """
    Convert the list of per-window NetworkX graphs produced by the Graph2Vec
    pipeline (``generate_graph2vecwithadaptativethreshold``) into the list of
    per-window PyG ego-graphs consumed by ``GCNTimeSeriesDataset``.

    Each entry ``graphs[i]`` is the graph built on days
    ``i*step_size .. i*step_size + window_size - 1`` of ``df_wide``.

    Parameters
    ----------
    graphs         : list of networkx.Graph (one per sliding-window position)
    df_wide        : pandas.DataFrame  (item_id × date)  — the full pivot of
                     the target column, must contain ``product_id`` and every
                     neighbour label that appears in the graphs.
    product_id     : label of the target item (must be a node in the graphs)
    window_size    : width of each window in days
    step_size      : stride between consecutive windows (default 1)
    max_neighbours : optional cap on the number of neighbours kept per window
                     (top-weight kept first); ``None`` = keep all neighbours.

    Returns
    -------
    list[Data] of length ``len(graphs)``, target node is row 0 of each Data.
    """
    if product_id not in df_wide.index:
        raise KeyError(f"product_id {product_id} not in df_wide.index")

    values_full = df_wide.values            # (n_items, T)
    label_to_row = {lbl: i for i, lbl in enumerate(df_wide.index)}

    pyg_list = []
    for i, G in enumerate(graphs):
        # neighbours of the target node in this window's graph
        if product_id in G:
            nbrs = list(G.neighbors(product_id))
            if max_neighbours is not None and len(nbrs) > max_neighbours:
                # keep top-weight neighbours
                nbrs = sorted(
                    nbrs,
                    key=lambda u: float(G[product_id][u].get("weight", 0.0)),
                    reverse=True,
                )[:max_neighbours]
        else:
            nbrs = []

        node_order = [product_id] + [n for n in nbrs if n in label_to_row]

        # window slice for these nodes (aligned to graphs[i] convention)
        t0 = i * step_size
        t1 = t0 + window_size
        if t1 > values_full.shape[1]:
            t1 = values_full.shape[1]
            t0 = max(t1 - window_size, 0)
        rows = [label_to_row[lbl] for lbl in node_order]
        window_values = values_full[rows, t0:t1]

        # restrict G to node_order so nx_window_to_pyg only sees relevant edges
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
    """
    Build a PyG Data object from a NetworkX graph + raw window slice.

    G              : networkx.Graph with edge attr 'weight' (similarity score)
    node_order     : ordered iterable of node labels (defines row 0..N-1)
                     **the first entry MUST be the target node**
    window_values  : (n_nodes, window_size) aligned to node_order
    target_node    : the label of the target (should equal node_order[0])
    """
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

    if not src:           # isolated target: keep a self-loop so GCN has work
        src, dst, w = [0], [0], [1.0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)
    x          = torch.from_numpy(_window_node_features(window_values))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


# ── dataset ─────────────────────────────────────────────────────────────────
class GCNTimeSeriesDataset(Dataset):
    """
    Drop-in analogue of ``TimeSeriesDataset`` (Graph2Vec variant) that
    feeds a trainable GCN instead of frozen embeddings.

    Parameters
    ----------
    target_data       : (T,) scaled target series
    exog_data         : (T, n_exog) scaled exogenous matrix (or None)
    seq_length        : LSTM lookback (L)
    pyg_graphs        : list of length T (after padding) of ``Data`` ego-graphs
                        — entry ``t`` is built on days ``t - graph_window_size
                        .. t - 1``.  Pass the unpadded list and we'll zero-pad.
    graph_window_size : window width used to build the graphs
    target_node_idx   : index of the target node inside every graph (default 0,
                        i.e. ego at row 0 by ``nx_window_to_pyg`` convention)
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

        # ─── zero-pad the graph list so pyg_graphs[t] is aligned to day t ───
        T = len(self.target_data)
        if len(pyg_graphs) < T:
            # build a tiny empty graph with the right node-feature width
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

        # ONE GRAPH PER LOOKBACK STEP.
        # Step i in [0, L)  ->  pyg_graphs[idx + 1 + i], the graph built on the
        # `graph_window_size` days preceding day idx + 1 + i.  This matches the
        # Graph2Vec per-timestep concatenation convention.
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
    """
    PER-STEP collation.  Batches a list of (list[Data] of length L, ts_seq, y)
    tuples into:

        pyg_batch  : torch_geometric.data.Batch of B*L ego-graphs
                     (flattened row-major: sample0_step0, sample0_step1, ...,
                      sample0_step{L-1}, sample1_step0, ...)
        ts_batch   : (B, L, F)
        y_batch    : (B, 1)
        target_idx : (B*L,) absolute indices of each subgraph's target node
                     inside the batched node-feature matrix (= ptr[:-1]).
                     The model reshapes ``z`` to (B, L, d_g) using L.
        L          : int — lookback length (needed for the reshape)
    """
    graph_lists, ts_seqs, ys = zip(*batch)
    L = len(graph_lists[0])
    flat_graphs = [g for gl in graph_lists for g in gl]   # B*L Data objects
    pyg_batch   = Batch.from_data_list(flat_graphs)
    ts_batch    = torch.stack(ts_seqs, dim=0)
    y_batch     = torch.stack(ys, dim=0)
    target_idx  = pyg_batch.ptr[:-1]                      # (B*L,)
    return pyg_batch, ts_batch, y_batch, target_idx, L
