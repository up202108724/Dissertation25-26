import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Pure MLP baseline helpers (no graph)
# ---------------------------------------------------------------------------

def make_xy_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
) -> tuple:
    """
    Sliding-window dataset for the pure-MLP (no-graph) baseline.

    Returns
    -------
    X : (N, lookback, C)  float32
    y : (N, horizon, 1)   float32
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    exog_indices = [i for i in range(C) if i != target_channel]

    X = np.zeros((N, lookback, C), dtype=np.float32)
    y = np.zeros((N, horizon, 1), dtype=np.float32)

    for i in range(N):
        window = series[i : i + lookback].copy()
        # Shift exogenous features forward by 1 (target day's exog visible)
        if exog_indices:
            window[:, exog_indices] = series[i + 1 : i + lookback + 1, exog_indices]
        X[i] = window
        y[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

    return X, y


# ---------------------------------------------------------------------------
# Pure GraphSAGE helpers (one graph per sample)
# ---------------------------------------------------------------------------

class SingleGraphDataset(Dataset):
    """
    Dataset for the GNN+LSTM forecaster.

    Each sample contains:
        y      – (horizon, 1)  target values to predict
        graph  – a single PyG Data object whose node features encode the
                 full temporal context (see make_single_windows)
        ts_seq – (lookback, 1+cal_dim)  explicit LSTM input sequence
    """

    def __init__(self, y: np.ndarray, graphs, ts_seqs: np.ndarray = None):
        self.y = torch.from_numpy(y)   # (N, H, 1)
        self.graphs = graphs            # List[Data], length N
        self.ts_seqs = (torch.from_numpy(ts_seqs) if ts_seqs is not None else None)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.ts_seqs is not None:
            return self.y[idx], self.graphs[idx], self.ts_seqs[idx]
        return self.y[idx], self.graphs[idx]


def single_graph_collate(batch):
    """Custom collate: stack y tensors + merge PyG graphs into a Batch.
    Handles both 2-item (y, graph) and 3-item (y, graph, ts_seq) batches."""
    y_batch     = torch.stack([item[0] for item in batch])
    graphs_batch = Batch.from_data_list([item[1] for item in batch])
    if len(batch[0]) == 3:
        ts_batch = torch.stack([item[2] for item in batch])
        return y_batch, graphs_batch, ts_batch
    return y_batch, graphs_batch


def make_single_windows(
    series: np.ndarray,
    cal: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    graphs=None,
    graph_window_size: int = 15,
    include_cal_lookback: bool = False,
    node_features: list = None,
    cal_columns: list = None,
) -> tuple:
    """
    Creates one (y, graph, ts_seq) tuple per sliding window for the GNN+LSTM forecaster.

    Mirrors graphsagedataset.make_single_windows exactly, using GCNConv's
    generate_node_features from gnn_pyg instead of SAGEConv's from graphsage_pyg.

    For each sample *i* the graph used is the one whose similarity window ends at
    the last observation of the lookback period (timestep i + lookback - 1).

    ts_seq[t] = (value_t, cal_{t+1})
    The last element ts_seq[lookback-1] = (value_{lookback-1}, cal_next) is
    consistent with the cal_next embedded in the target node’s features.

    Parameters
    ----------
    series               : (T, 1)        – scaled target-channel values
    cal                  : (T, cal_dim)  – scaled calendar features
    lookback             : int
    horizon              : int
    target_channel       : int
    graphs               : List[Data]    – one graph per timestep from neighbourhood_graph
    graph_window_size    : int
    include_cal_lookback : bool          – if True, the (lookback × cal_dim) calendar matrix
                                           is flattened into the target-node features
    node_features        : list[str]     – named features for generate_node_features
    cal_columns          : list[str]     – calendar column names for per-column feature keys

    Returns
    -------
    y             : (N, horizon, 1)   np.float32
    window_graphs : List[Data]        length N
    ts_seqs       : (N, lookback, 1+cal_dim)  np.float32
    """
    from gnn_pyg import generate_node_features

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    # Compute feature_dim dynamically so any node_features combination works.
    _dummy_ts  = np.zeros(lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    feature_dim = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))

    # Pad graphs so that padded_graphs[t] = graph whose window ends at day t.
    # neighbourhood_graph builds sliding windows of size graph_window_size so
    # we need exactly (graph_window_size - 1) leading dummy graphs.
    if graphs is not None:
        num_missing = graph_window_size - 1
        dummy_graph = Data(
            x=torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * num_missing + list(graphs)
    else:
        dummy_graph = Data(
            x=torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * T

    y             = np.zeros((N, horizon, 1),          dtype=np.float32)
    window_graphs = []
    # ts_seqs[i, t] = (value_t, cal_{t+1})
    ts_seqs       = np.zeros((N, lookback, 1 + cal_dim), dtype=np.float32)

    for i in range(N):
        y[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        # ---- ts_seq: (value_t, cal_{t+1}) for t = 0..lookback-1 ----
        ts_vals = series[i : i + lookback, target_channel]          # (lookback,)
        _cal_end = i + lookback + 1
        if _cal_end <= T:
            _cal_shifted = cal[i + 1 : _cal_end]                    # (lookback, cal_dim)
        else:
            _avail = cal[i + 1 : T]
            _pad   = np.repeat(cal[T - 1 : T], lookback - len(_avail), axis=0)
            _cal_shifted = np.vstack([_avail, _pad]) if len(_avail) > 0 else _pad
        if cal_dim > 0:
            ts_seqs[i] = np.concatenate([ts_vals[:, None], _cal_shifted], axis=1)
        else:
            ts_seqs[i, :, 0] = ts_vals

        # ---- Graph whose sliding window ends at i + lookback - 1 ----
        base_graph = padded_graphs[i + lookback - 1]
        n_nodes    = base_graph.x.shape[0]
        x_new      = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

        # Target node (index 0): full lookback + [cal_lookback] + next-step cal + stats
        target_ts = series[i : i + lookback, target_channel]           # (lookback,)
        cal_next  = cal[i + lookback] if (i + lookback) < T else cal[-1]  # (cal_dim,)
        cal_lb    = cal[i : i + lookback] if include_cal_lookback else None

        x_new[0] = torch.tensor(
            generate_node_features(
                target_ts, cal_next=cal_next, cal_lookback=cal_lb,
                selected_features=node_features, cal_columns=cal_columns,
            ),
            dtype=torch.float32,
        )

        # Neighbor nodes: zeros for calendar/extra ts slots, right-align ts
        for node_idx in range(1, n_nodes):
            orig_feat   = base_graph.x[node_idx].numpy()
            neighbor_ts = orig_feat[:graph_window_size]
            x_new[node_idx] = torch.tensor(
                generate_node_features(
                    neighbor_ts, selected_features=node_features,
                    is_neighbor=True, pad_ts_to=lookback, cal_columns=cal_columns,
                ),
                dtype=torch.float32,
            )

        new_graph = Data(
            x=x_new,
            edge_index=base_graph.edge_index.clone(),
            edge_attr=(base_graph.edge_attr.clone()
                       if base_graph.edge_attr is not None else None),
            central_node_idx=0,
        )
        window_graphs.append(new_graph)

    return y, window_graphs, ts_seqs


# ---------------------------------------------------------------------------
# GNN+LSTM multi-graph helpers removed – see above (single-graph approach now
# mirrors GraphSAGE_LSTM exactly with GCN initialising LSTM h₀/c₀).
# ---------------------------------------------------------------------------
    """
    Creates one (y, [L graphs]) tuple per sliding window for the GNN+LSTM forecaster.

    For sample *i* the LSTM sequence consists of L=lookback consecutive graphs:
        graph at timestep i+0, i+1, ..., i+L-1

    Each graph's target-node (index 0) features are rebuilt using
    ``compute_target_node_features_seq`` so that step t carries the
    *graph-window-sized* ts context ending at timestep i+t together
    with the calendar features at that date.

    feature_dim = graph_window_size + cal_dim + 8
    (NOT lookback + cal_dim + 8 — that is for the pure-GNN approach)

    Parameters
    ----------
    series           : (T, 1)        – scaled target-channel values
    cal              : (T, cal_dim)  – scaled calendar features
    lookback         : int           – LSTM sequence length L
    horizon          : int
    target_channel   : int
    graphs           : List[Data]    – one graph per timestep from neighbourhood_graph
    graph_window_size: int

    Returns
    -------
    y             : (N, horizon, 1)   np.float32
    multi_graphs  : List[List[Data]]  length N, each inner list has L Data objects
    """
    from gnn_pyg import compute_target_node_features_seq, compute_neighbor_node_features_pure

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    feature_dim = graph_window_size + cal_dim + 8

    # Pad graphs so that padded_graphs[t] = graph whose window ends at timestep t.
    # neighbourhood_graph builds (T - window_size + 1) graphs; padding from the
    # front aligns the index so padded_graphs[t] ends at t.
    if graphs is not None:
        num_missing = T - len(graphs)
        dummy_graph = Data(
            x=torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * num_missing + list(graphs)
    else:
        dummy_graph = Data(
            x=torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * T

    y            = np.zeros((N, horizon, 1), dtype=np.float32)
    multi_graphs = []

    for i in range(N):
        y[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        window_graphs = []
        for t in range(lookback):
            # Graph whose window ends at timestep i + t
            base_graph = padded_graphs[i + t]
            n_nodes = base_graph.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # Target node: ts = graph_window_size values ending at i + t
            ts_end   = i + t + 1
            ts_start = max(0, ts_end - graph_window_size)
            ts_t     = series[ts_start : ts_end, target_channel]          # (≤gw,)
            cal_at_t = cal[i + t] if (i + t) < T else cal[-1]            # (cal_dim,)

            x_new[0] = torch.tensor(
                compute_target_node_features_seq(ts_t, cal_at_t, feature_dim),
                dtype=torch.float32,
            )

            # Neighbor nodes: right-fill ts with zeros for missing positions
            for node_idx in range(1, n_nodes):
                orig_feat   = base_graph.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                x_new[node_idx] = torch.tensor(
                    compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                    dtype=torch.float32,
                )

            new_graph = Data(
                x=x_new,
                edge_index=base_graph.edge_index.clone(),
                edge_attr=(base_graph.edge_attr.clone()
                           if base_graph.edge_attr is not None else None),
                central_node_idx=0,
            )
            window_graphs.append(new_graph)

        multi_graphs.append(window_graphs)

    return y, multi_graphs


# ---------------------------------------------------------------------------
# GNN + LSTM helpers  (L graphs per sample, one per lookback step)
# ---------------------------------------------------------------------------

class SequenceGraphDataset(Dataset):
    """
    Dataset for SimpleGNNLSTMForecaster.

    Each sample:
        y              – (horizon, 1)          target values
        graph_sequence – List[Data] of length L (one graph per lookback step)
    """

    def __init__(self, y: np.ndarray, graph_sequences):
        self.y = torch.from_numpy(y)           # (N, H, 1)
        self.graph_sequences = graph_sequences  # List[List[Data]], len N

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.graph_sequences[idx]


def sequence_graph_collate(batch):
    """
    Collate for SequenceGraphDataset.

    Returns
    -------
    y_batch   : (B, H, 1)
    pyg_batch : PyG Batch of B×L graphs in sample-major order
    B         : int  (batch size)
    L         : int  (lookback length)
    """
    y_list   = [item[0] for item in batch]
    seq_list = [item[1] for item in batch]   # List[List[Data]]

    B = len(seq_list)
    L = len(seq_list[0])

    y_batch = torch.stack(y_list)   # (B, H, 1)

    # Sample-major: [s0_t0, s0_t1, ..., s1_t0, ...]
    flat_graphs = [seq_list[b][l] for b in range(B) for l in range(L)]
    pyg_batch   = Batch.from_data_list(flat_graphs)

    return y_batch, pyg_batch, B, L


def make_sequence_windows(
    series: np.ndarray,
    cal: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    graphs=None,
    graph_window_size: int = 15,
) -> tuple:
    """
    Build L-graph sequences for the SimpleGNNLSTMForecaster.

    For each sample i, for each lookback step l (0 … L-1):
      * global timestep  t = i + l
      * use the graph whose similarity window ends at t
      * target-node features: [ts_window | cal[t+1] (next-step) | stats_8]
        computed by compute_target_node_features_seq
      * neighbour features:   compute_neighbor_node_features_pure(ts, feature_dim)
        where feature_dim = graph_window_size + cal_dim + 8

    Parameters
    ----------
    series          : (T, 1)        scaled target
    cal             : (T, cal_dim)  scaled calendar features
    lookback        : int           L  (LSTM sequence length)
    horizon         : int
    graphs          : List[Data]    one graph per timestep from neighbourhood_graph
    graph_window_size : int

    Returns
    -------
    y_arr          : (N, H, 1)        float32
    graph_sequences: List[List[Data]] len N, inner len L
    """
    from gnn_pyg import (compute_target_node_features_seq,
                          compute_neighbor_node_features_pure)

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    feature_dim = graph_window_size + cal_dim + 8

    # Pad graphs so padded_graphs[t] = graph whose window ends at t
    dummy_x = torch.zeros((1, feature_dim), dtype=torch.float32)
    dummy_graph = Data(
        x=dummy_x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float),
        central_node_idx=0,
    )
    if graphs is not None:
        num_missing = T - len(graphs)
        padded_graphs = [dummy_graph] * num_missing + list(graphs)
    else:
        padded_graphs = [dummy_graph] * T

    y_arr          = np.zeros((N, horizon, 1), dtype=np.float32)
    graph_sequences = []

    for i in range(N):
        y_arr[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        seq_graphs = []
        for l in range(lookback):
            t          = i + l                   # global time position
            base_graph = padded_graphs[t]
            n_nodes    = base_graph.x.shape[0]
            x_new      = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # ts window: last graph_window_size values ending at t (right-aligned)
            ts_start = max(0, t - graph_window_size + 1)
            ts_win   = series[ts_start : t + 1, target_channel]

            # Next-step calendar (same exog-shift convention as make_xy_windows)
            cal_next = cal[t + 1] if (t + 1) < T else cal[-1]

            x_new[0] = torch.tensor(
                compute_target_node_features_seq(ts_win, cal_next, feature_dim),
                dtype=torch.float32,
            )

            for node_idx in range(1, n_nodes):
                orig_feat   = base_graph.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                x_new[node_idx] = torch.tensor(
                    compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                    dtype=torch.float32,
                )

            new_graph = Data(
                x=x_new,
                edge_index=base_graph.edge_index.clone(),
                edge_attr=(base_graph.edge_attr.clone()
                           if base_graph.edge_attr is not None else None),
                central_node_idx=0,
            )
            seq_graphs.append(new_graph)

        graph_sequences.append(seq_graphs)

    return y_arr, graph_sequences


# ---------------------------------------------------------------------------
# SAGE + LSTM helpers  (L graphs per sample, one per lookback step)
# ---------------------------------------------------------------------------

class SequenceGraphDataset(Dataset):
    """
    Dataset for the GraphSAGE+LSTM forecaster.

    Each sample:
        y              – (horizon, 1)          target values
        graph_sequence – List[Data] of length L (one graph per lookback step)
    """

    def __init__(self, y: np.ndarray, graph_sequences):
        self.y = torch.from_numpy(y)          # (N, H, 1)
        self.graph_sequences = graph_sequences # List[List[Data]], len N

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.graph_sequences[idx]


def sequence_graph_collate(batch):
    """
    Collate for SequenceGraphDataset.

    Returns
    -------
    y_batch   : (B, H, 1)
    pyg_batch : PyG Batch of B×L graphs in sample-major order
    B         : int  (batch size)
    L         : int  (lookback length)
    """
    y_list   = [item[0] for item in batch]
    seq_list = [item[1] for item in batch]   # List[List[Data]]

    B = len(seq_list)
    L = len(seq_list[0])

    y_batch = torch.stack(y_list)   # (B, H, 1)

    # Sample-major: [s0_t0, s0_t1, ..., s0_tL-1, s1_t0, ...]
    flat_graphs = [seq_list[b][l] for b in range(B) for l in range(L)]
    pyg_batch   = Batch.from_data_list(flat_graphs)

    return y_batch, pyg_batch, B, L


def make_sequence_windows(
    series: np.ndarray,
    cal: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    graphs=None,
    graph_window_size: int = 15,
) -> tuple:
    """
    Build L-graph sequences for the GraphSAGE+LSTM forecaster.

    For each sample i, for each lookback step l (0 … L-1):
      * global timestep  t = i + l
      * use the graph whose similarity window ends at t
      * target-node features: [ts_window | cal[t+1] (next-step) | stats_8]
        computed by compute_target_node_features_seq
      * neighbour features:   compute_neighbor_node_features_pure(ts, feature_dim)
        where feature_dim = graph_window_size + cal_dim + 8

    Parameters
    ----------
    series          : (T, 1)        scaled target
    cal             : (T, cal_dim)  scaled calendar features
    lookback        : int           L  (LSTM sequence length)
    horizon         : int           H
    graphs          : List[Data]    T - graph_window_size + 1 graphs from neighbourhood_graph
    graph_window_size : int

    Returns
    -------
    y_arr          : (N, H, 1)   float32
    graph_sequences: List[List[Data]]   len N, inner len L
    """
    from torch_geometric.data import Data
    from gnn_pyg import (compute_target_node_features_seq,
                          compute_neighbor_node_features_pure)

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for the given lookback/horizon.")

    feature_dim = graph_window_size + cal_dim + 8   # per-step node feature size

    # padded_graphs[t] = graph whose window ends at timestep t
    if graphs is not None:
        num_missing = T - len(graphs)
        dummy = Data(
            x          = torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index = torch.empty((2, 0), dtype=torch.long),
            edge_attr  = torch.empty((0, 1), dtype=torch.float),
        )
        padded_graphs = [dummy] * num_missing + list(graphs)
    else:
        dummy = Data(
            x          = torch.zeros((1, feature_dim), dtype=torch.float32),
            edge_index = torch.empty((2, 0), dtype=torch.long),
            edge_attr  = torch.empty((0, 1), dtype=torch.float),
        )
        padded_graphs = [dummy] * T

    y_arr           = np.zeros((N, horizon, 1), dtype=np.float32)
    graph_sequences = []

    for i in range(N):
        y_arr[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        seq_i = []
        for l in range(lookback):
            t       = i + l
            G_base  = padded_graphs[t]
            n_nodes = G_base.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # ── Target node (index 0) ──
            start  = max(0, t - graph_window_size + 1)
            ts_win = series[start : t + 1, target_channel]          # ≤ gw values
            # Next-step shifted calendar (matches the exog shift in make_xy_windows)
            cal_next = cal[t + 1] if (t + 1) < T else cal[t]

            x_new[0] = torch.tensor(
                compute_target_node_features_seq(ts_win, cal_next, feature_dim),
                dtype=torch.float32,
            )

            # ── Neighbor nodes ──
            for node_idx in range(1, n_nodes):
                orig_feat   = G_base.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                x_new[node_idx] = torch.tensor(
                    compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                    dtype=torch.float32,
                )

            G_new = Data(
                x          = x_new,
                edge_index = G_base.edge_index.clone(),
                edge_attr  = (G_base.edge_attr.clone()
                               if G_base.edge_attr is not None else None),
            )
            G_new.central_node_idx = 0
            seq_i.append(G_new)

        graph_sequences.append(seq_i)

    return y_arr, graph_sequences
