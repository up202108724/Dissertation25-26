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
    Dataset for the pure GraphSAGE forecaster.

    Each sample contains:
        y      – (horizon, 1)  target values to predict
        graph  – a single PyG Data object whose node features already encode
                 the full temporal context (see make_single_windows)
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
    y_batch = torch.stack([item[0] for item in batch])
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
    Creates one (y, graph) pair per sliding window for the pure SAGE forecaster.

    For each sample *i* the graph used is the one whose similarity window ends at
    the last observation of the lookback period (timestep i + lookback - 1).

    The target node (always index 0 in each graph) has its feature vector rebuilt
    to include the **full lookback window + [cal_lookback] + next-step calendar +
    8 stats**, giving all temporal context that the (now-removed) MLP used to receive.

    Neighbor nodes are re-padded to the same feature dimension (their calendar
    and extra time-slots are set to zero).

    Parameters
    ----------
    series               : (T, 1)        – scaled target-channel values
    cal                  : (T, cal_dim)  – scaled calendar features
    lookback             : int
    horizon              : int
    target_channel       : int           – column index of the target in `series`
    graphs               : List[Data]    – one graph per timestep from neighbourhood_graph
                                          (may be None for the no-graph dummy path)
    graph_window_size    : int           – window_size used when graphs were built
    include_cal_lookback : bool          – if True, the full (lookback × cal_dim) calendar
                                          matrix is flattened and added to the target node
                                          features between ts_lookback and cal_next_step.
                                          feature_dim becomes lookback*(1+cal_dim)+cal_dim+8
                                          instead of lookback+cal_dim+8.

    Returns
    -------
    y             : (N, horizon, 1)  np.float32
    window_graphs : List[Data]       length N
    """
    from graphsage_pyg import generate_node_features

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    # Compute feature_dim dynamically from the actual feature list so it works
    # with any node_features combination (ts+cal, stats-only, mixed, etc.).
    _dummy_ts  = np.zeros(lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    feature_dim = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))

    # --- Pad graphs so that padded_graphs[t] = graph whose window ends at t ---
    # neighbourhood_graph builds (T - window_size + 1) graphs; the i-th graph
    # covers [i, i + window_size).  We need padded_graphs[t] to be the graph
    # whose window *ends* at t, i.e. graph[t - window_size + 1] covering
    # [t - window_size + 1, t].  This requires exactly (window_size - 1)
    # leading dummy graphs, regardless of how many graphs were passed.
    if graphs is not None:
        num_missing = graph_window_size - 1
        dummy_x = torch.zeros((1, feature_dim), dtype=torch.float32)
        dummy_graph = Data(
            x=dummy_x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * num_missing + list(graphs)
    else:
        dummy_x = torch.zeros((1, feature_dim), dtype=torch.float32)
        dummy_graph = Data(
            x=dummy_x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            central_node_idx=0,
        )
        padded_graphs = [dummy_graph] * T

    y             = np.zeros((N, horizon, 1), dtype=np.float32)
    window_graphs = []
    # ts_seqs[i, t] = (value_t, cal_{t+1}) — the value at step t and the
    # calendar of the NEXT day, so the last element ts_seqs[i, lookback-1]
    # aligns exactly with cal_next used in the target node's features.
    ts_seqs       = np.zeros((N, lookback, 1 + cal_dim), dtype=np.float32)

    for i in range(N):
        y[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        # -- ts_seq: (value_t, cal_{t+1}) for t = 0..lookback-1 --
        ts_vals = series[i : i + lookback, target_channel]   # (lookback,)
        _cal_end = i + lookback + 1
        if _cal_end <= T:
            _cal_shifted = cal[i + 1 : _cal_end]             # (lookback, cal_dim)
        else:
            _avail = cal[i + 1 : T]
            _pad   = np.repeat(cal[T - 1 : T], lookback - len(_avail), axis=0)
            _cal_shifted = np.vstack([_avail, _pad]) if len(_avail) > 0 else _pad
        if cal_dim > 0:
            ts_seqs[i] = np.concatenate([ts_vals[:, None], _cal_shifted], axis=1)
        else:
            ts_seqs[i, :, 0] = ts_vals

        # Graph whose sliding window ends at i + lookback - 1
        base_graph = padded_graphs[i + lookback - 1]

        # --- Rebuild node feature matrix ---
        n_nodes = base_graph.x.shape[0]
        x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

        # Target node (index 0): full lookback + [cal_lookback] + next-step calendar + stats
        target_ts = series[i : i + lookback, target_channel]           # (lookback,)
        cal_next  = cal[i + lookback] if (i + lookback) < T else cal[-1]  # (cal_dim,)
        cal_lb    = cal[i : i + lookback] if include_cal_lookback else None  # (lookback, cal_dim) or None
        
        selected_target = node_features 
        x_new[0]  = torch.tensor(
            generate_node_features(target_ts, cal_next=cal_next, cal_lookback=cal_lb,
                                   selected_features=selected_target, cal_columns=cal_columns),
            dtype=torch.float32,
        )

        # Neighbor nodes: re-pad to feature_dim (zeros for extra ts slots + calendar)
        for node_idx in range(1, n_nodes):
            orig_feat   = base_graph.x[node_idx].numpy()            # (graph_window_size + 8,)
            neighbor_ts = orig_feat[:graph_window_size]              # raw values only
            
            selected_neighbor = node_features
            x_new[node_idx] = torch.tensor(
                generate_node_features(neighbor_ts, selected_features=selected_neighbor,
                                       is_neighbor=True, pad_ts_to=lookback, cal_columns=cal_columns),
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
