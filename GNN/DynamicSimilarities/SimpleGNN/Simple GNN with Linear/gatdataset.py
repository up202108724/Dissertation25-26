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

    def __init__(self, y: np.ndarray, graphs):
        self.y = torch.from_numpy(y)   # (N, H, 1)
        self.graphs = graphs            # List[Data], length N

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.graphs[idx]


def single_graph_collate(batch):
    """Custom collate: stack y tensors + merge PyG graphs into a Batch."""
    y_batch = torch.stack([item[0] for item in batch])
    graphs_batch = Batch.from_data_list([item[1] for item in batch])
    return y_batch, graphs_batch


def make_single_windows(
    series: np.ndarray,
    cal: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    graphs=None,
    graph_window_size: int = 15,
) -> tuple:
    """
    Creates one (y, graph) pair per sliding window for the pure SAGE forecaster.

    For each sample *i* the graph used is the one whose similarity window ends at
    the last observation of the lookback period (timestep i + lookback - 1).

    The target node (always index 0 in each graph) has its feature vector rebuilt
    to include the **full lookback window + next-step calendar + 8 stats**, giving
    all temporal context that the (now-removed) MLP used to receive.

    Neighbor nodes are re-padded to the same feature dimension (their calendar
    and extra time-slots are set to zero).

    Parameters
    ----------
    series          : (T, 1)        – scaled target-channel values
    cal             : (T, cal_dim)  – scaled calendar features
    lookback        : int
    horizon         : int
    target_channel  : int           – column index of the target in `series`
    graphs          : List[Data]    – one graph per timestep from neighbourhood_graph
                                     (may be None for the no-graph dummy path)
    graph_window_size : int         – window_size used when graphs were built

    Returns
    -------
    y             : (N, horizon, 1)  np.float32
    window_graphs : List[Data]       length N
    """
    from gat_pyg import compute_target_node_features_pure, compute_neighbor_node_features_pure

    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1]
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    feature_dim = lookback + cal_dim + 8   # target node feature size

    # --- Pad graphs so that padded_graphs[t] = graph whose window ends at t ---
    # neighbourhood_graph builds (T - window_size + 1) graphs; the i-th graph
    # covers [i, i + window_size).  Padding from the front aligns index t with
    # the graph whose window ends at t.
    if graphs is not None:
        num_missing = T - len(graphs)
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

    for i in range(N):
        y[i, :, 0] = series[i + lookback : i + lookback + horizon, target_channel]

        # Graph whose sliding window ends at i + lookback - 1
        base_graph = padded_graphs[i + lookback - 1]

        # --- Rebuild node feature matrix ---
        n_nodes = base_graph.x.shape[0]
        x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

        # Target node (index 0): full lookback + next-step calendar + stats
        target_ts = series[i : i + lookback, target_channel]           # (lookback,)
        cal_next  = cal[i + lookback] if (i + lookback) < T else cal[-1]  # (cal_dim,)
        x_new[0]  = torch.tensor(
            compute_target_node_features_pure(target_ts, cal_next, feature_dim),
            dtype=torch.float32,
        )

        # Neighbor nodes: re-pad to feature_dim (zeros for extra ts slots + calendar)
        for node_idx in range(1, n_nodes):
            orig_feat   = base_graph.x[node_idx].numpy()            # (graph_window_size + 8,)
            neighbor_ts = orig_feat[:graph_window_size]              # raw values only
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

    return y, window_graphs
