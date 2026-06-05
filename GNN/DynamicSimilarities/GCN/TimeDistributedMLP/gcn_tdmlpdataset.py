"""
PyG-aware dataset for the GCN + TimeDistributed MLP forecaster.

Companion to ``Graph2vec_FixedThreshold/TimeDistributedMLP/graph2vecdataset.py``
but yields a *trainable* graph (a ``torch_geometric.data.Data`` ego-graph)
at every lookback step instead of a precomputed Graph2Vec embedding row.

Per sample (idx) we return:

    graphs : list[Data]  of length L = lookback_window
        ``graphs[i]`` is the ego-graph aligned to lookback step ``i``,
        i.e. the graph built on the ``graph_window_size`` days strictly
        preceding day ``idx + 1 + i`` of ``target_data``.  The shared
        per-step MLP head consumes one ``z_i`` per step alongside
        ``ts_seq[:, i, :]``.
    ts_seq : FloatTensor  (lookback_window, 1 + n_exog)
        [target ‖ exog] sequence consumed by the TimeDistributed MLP.
        Exog rows are shifted by +1 so each row carries the exogenous
        features of the day being predicted at that step (matches
        ``make_windows`` from the Graph2Vec / TDMLP pipeline).
    y      : FloatTensor  (1,)
        Target value at step ``idx + lookback_window``.

The list ``pyg_graphs`` is expected to be aligned to the target timeline
(one Data per day).  When shorter than ``len(target_data)`` we left-pad
with empty single-node graphs so ``self.pyg_graphs[t]`` always exists.

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


# ── node feature builders ─────────────────────────────────────────────────
def _window_node_features(window_values: np.ndarray) -> np.ndarray:
    """
    window_values : (n_nodes, window_size) raw values per node over the window
    Returns       : (n_nodes, window_size) — the raw sequence itself.
    """
    if window_values.ndim != 2:
        raise ValueError("window_values must be 2D (n_nodes, window_size)")
    return window_values.astype(np.float32)


def _window_node_stats_features(window_values: np.ndarray) -> np.ndarray:
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


# ── NX-window -> PyG ego-graph ────────────────────────────────────────────
def nx_window_to_pyg(
    G,
    node_order: Sequence,
    window_values: np.ndarray,
    target_node,
    node_feature_mode: str = 'raw',
    node_labels: Optional[Sequence] = None,
) -> Data:
    """
    Build a PyG Data object from a NetworkX graph + raw window slice.

    G                 : networkx.Graph with edge attr 'weight'
    node_order        : ordered iterable of node labels (row 0..N-1).
                        **the first entry MUST be the target node**
    window_values     : (n_nodes, window_size) aligned to node_order
    target_node       : the label of the target (= node_order[0])
    node_feature_mode : 'raw' (full sequence) or 'stats' (8-dim summary)
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
    if node_feature_mode == 'stats':
        x = torch.from_numpy(_window_node_stats_features(window_values))
    elif node_feature_mode == 'raw':
        x = torch.from_numpy(_window_node_features(window_values))
    else:
        raise ValueError(
            f"node_feature_mode must be 'raw' or 'stats', got {node_feature_mode!r}"
        )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
    data.node_labels = list(node_labels) if node_labels is not None else list(node_order)
    return data


# ── batch builder over the per-window NX graph list ───────────────────────
def build_pyg_graphs_from_nx_windows(
    graphs,
    df_wide,
    product_id,
    window_size: int,
    step_size: int = 1,
    max_neighbours: Optional[int] = None,
    node_scalers: Optional[dict] = None,
    node_feature_mode: str = 'raw',
):
    """
    Convert the list of per-window NetworkX graphs produced by the Graph2Vec
    pipeline (``generate_graph2vecwithadaptativethreshold``) into the list of
    per-window PyG ego-graphs consumed by ``GCNTimeSeriesDataset``.

    Each entry ``graphs[i]`` is the graph built on days
    ``i*step_size .. i*step_size + window_size - 1`` of ``df_wide``.
    Per-node values are z-scored (within the window or with a fitted
    scaler) before being stacked into the Data.x matrix.

    Returns
    -------
    list[Data] of length ``len(graphs)``; target node is always row 0.
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
        # Values are already globally z-scored (df_wide_scaled); no local
        # re-normalisation is applied so node features are on a consistent
        # scale across all windows.
        window_values = values_full[rows, t0:t1].astype(np.float32)

        H = G.subgraph(node_order).copy() if product_id in G else G.__class__()
        H.add_nodes_from(node_order)
        pyg_list.append(
            nx_window_to_pyg(
                H, node_order, window_values, product_id,
                node_feature_mode=node_feature_mode,
                node_labels=node_order,
            )
        )

    return pyg_list


# ── dataset ───────────────────────────────────────────────────────────────
class GCNTimeSeriesDataset(Dataset):
    """
    Dataset for the GCN + TimeDistributed MLP forecaster.

    For every index ``idx`` we materialise the lookback window
    ``target_data[idx : idx + lookback_window]`` plus the L per-step
    ego-graphs the shared MLP will consume.

    Parameters
    ----------
    target_data       : (T,) scaled target series
    exog_data         : (T, n_exog) scaled exogenous matrix (or None).
                        Exog rows are read at ``idx + 1 .. idx + L`` so the
                        model sees the calendar features of the day being
                        predicted at each lookback step (same shift as
                        ``make_windows`` in the Graph2Vec / TDMLP pipeline).
    lookback_window   : MLP lookback length (L) — same name as the constant
                        used at the runner level (``main_gcn_mlp.py``).
    pyg_graphs        : per-day list of ``Data`` ego-graphs.  Entry ``t``
                        must be the graph built on the ``graph_window_size``
                        days strictly preceding day ``t`` of ``target_data``.
                        If shorter than T, we left-pad with empty graphs so
                        alignment is preserved.
    graph_window_size : window width used to build the graphs (kept as an
                        attribute for downstream introspection only — the
                        per-day alignment is enforced by ``pyg_graphs``).
    target_node_idx   : index of the target node inside every graph
                        (default 0, i.e. ego at row 0 per
                        ``nx_window_to_pyg`` convention).
    """

    def __init__(
        self,
        target_data: np.ndarray,
        exog_data: Optional[np.ndarray],
        lookback_window: int,
        pyg_graphs: List[Data],
        graph_window_size: int = 15,
        target_node_idx: int = 0,
    ):
        self.target_data       = np.asarray(target_data, dtype=np.float32)
        self.exog_data         = (
            None if exog_data is None else np.asarray(exog_data, dtype=np.float32)
        )
        self.lookback_window   = int(lookback_window)
        self.graph_window_size = int(graph_window_size)
        self.target_node_idx   = int(target_node_idx)
        self.has_exog          = self.exog_data is not None

        # ─── left-pad the graph list so pyg_graphs[t] is aligned to day t ───
        # Callers may pass either:
        #   (a) pre-aligned graphs of length T  (full pipeline)
        #   (b) graphs of length T - lookback_window  (ablation dummy graphs,
        #       or any case where the val-context prepend adds lookback rows
        #       to target_data but the graph list is not extended)
        # In every case we pad with exactly (T - len(pyg_graphs)) dummies.
        T = len(self.target_data)
        n_graphs = len(pyg_graphs)
        if n_graphs > T:
            raise ValueError(
                f"pyg_graphs has {n_graphs} entries but target_data has only {T}."
            )
        if n_graphs < T:
            sample   = pyg_graphs[0]
            n_nodes  = sample.num_nodes
            in_feats = sample.x.shape[1]
            pad = Data(
                x=torch.zeros(n_nodes, in_feats, dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                edge_attr=torch.zeros(1, 1, dtype=torch.float32),
                num_nodes=n_nodes,
            )
            self.pyg_graphs = [pad] * (T - n_graphs) + list(pyg_graphs)
        else:
            self.pyg_graphs = list(pyg_graphs)

    def __len__(self):
        return len(self.target_data) - self.lookback_window

    def __getitem__(self, idx):
        L = self.lookback_window
        target_seq = self.target_data[idx : idx + L]
        y          = self.target_data[idx + L]

        if self.has_exog:
            # Shift exog forward by 1 so each lookback row carries the
            # calendar features of the day being predicted at that step.
            exog_seq = self.exog_data[idx + 1 : idx + L + 1]
            ts = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            ts = target_seq.reshape(-1, 1)

        # ONE GRAPH PER LOOKBACK STEP.
        # Step i in [0, L)  ->  pyg_graphs[idx + 1 + i] — the graph built on
        # the graph_window_size days strictly preceding day idx + 1 + i.
        # The shared per-step MLP receives [ts_seq[:, i, :] || z_i] at each i.
        end = len(self.pyg_graphs) - 1
        graphs = [
            self.pyg_graphs[min(max(idx + 1 + i, 0), end)]
            for i in range(L)
        ]

        ts_tensor = torch.from_numpy(ts.astype(np.float32))
        y_tensor  = torch.tensor([y], dtype=torch.float32)
        return graphs, ts_tensor, y_tensor


# ── collate ───────────────────────────────────────────────────────────────
def collate_pyg_ts(batch):
    """
    PER-STEP collation for the TimeDistributed MLP head.

    Batches a list of ``(list[Data] of length L, ts_seq, y)`` tuples into:

        pyg_batch  : torch_geometric.data.Batch of B*L ego-graphs
                     (flattened row-major: sample0_step0, sample0_step1, ...,
                      sample0_step{L-1}, sample1_step0, ...)
        ts_batch   : FloatTensor (B, L, F)
        y_batch    : FloatTensor (B, 1)
        target_idx : LongTensor  (B*L,) — absolute indices of each subgraph's
                     target node inside the batched node-feature matrix
                     (= ``pyg_batch.ptr[:-1]``).  ``SimpleGCNMLPForecaster``
                     gathers the target rows of the GCN output, reshapes to
                     ``(B, L, d_g)`` and concatenates with ``ts_batch`` before
                     the shared per-step MLP.
        L          : int — lookback length (needed for the (B, L, d_g) reshape).
    """
    graph_lists, ts_seqs, ys = zip(*batch)
    L = len(graph_lists[0])
    flat_graphs = [g for gl in graph_lists for g in gl]    # B*L Data objects
    pyg_batch   = Batch.from_data_list(flat_graphs)
    ts_batch    = torch.stack(ts_seqs, dim=0)
    y_batch     = torch.stack(ys, dim=0)
    target_idx  = pyg_batch.ptr[:-1]                       # (B*L,)
    return pyg_batch, ts_batch, y_batch, target_idx, L
