import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def compute_node_features(ts_sequence):
    """
    Computes node features from a time-series sequence.
    ts_sequence: 1D numpy array of sales/demand for the given window (e.g., 15 days).
    """
    ts_sequence = np.array(ts_sequence, dtype=np.float32)
    seq_len = len(ts_sequence)
    
    # 1. Raw Sequence
    raw_seq = ts_sequence.tolist()
    
    # 2. Statistical Features
    last_demand = ts_sequence[-1] if seq_len > 0 else 0.0
    mean7 = np.mean(ts_sequence[-7:]) if seq_len >= 7 else np.mean(ts_sequence)
    mean15 = np.mean(ts_sequence) if seq_len > 0 else 0.0
    std15 = np.std(ts_sequence) if seq_len > 0 else 0.0
    zero_ratio15 = np.mean(ts_sequence == 0) if seq_len > 0 else 0.0
    
    # Slope (linear trend)
    if seq_len > 1:
        x = np.arange(seq_len)
        # polyfit returns [slope, intercept]
        slope15 = np.polyfit(x, ts_sequence, 1)[0]
    else:
        slope15 = 0.0
        
    min_15 = np.min(ts_sequence) if seq_len > 0 else 0.0
    max_15 = np.max(ts_sequence) if seq_len > 0 else 0.0
    
    stats = [last_demand, mean7, mean15, std15, zero_ratio15, slope15, min_15, max_15]
    
    # Total features: len(raw_seq) + 8
    return np.array(raw_seq + stats, dtype=np.float32)

class SimpleGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(SimpleGNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        ew = edge_attr.squeeze(1) if (edge_attr is not None and edge_attr.shape[0] > 0) else None
        h = self.conv1(x, edge_index, edge_weight=ew)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight=ew)
        return h


# ---------------------------------------------------------------------------
# Pure GraphSAGE forecaster (no external MLP)
# ---------------------------------------------------------------------------

def compute_target_node_features_pure(ts_lookback, cal_next_step, feature_dim):
    """
    Builds the target (central) node feature vector for the pure SAGE forecaster.

    Layout: [ts_lookback | cal_next_step | stats_8]
        ts_lookback   – (lookback,) scaled target values covering the full lookback window
        cal_next_step – (cal_dim,)  calendar features for the prediction day
        stats_8       – 8 hand-crafted statistics computed on ts_lookback

    feature_dim = lookback + cal_dim + 8  (must match compute_neighbor_node_features_pure)
    """
    ts = np.array(ts_lookback, dtype=np.float32)
    lookback = len(ts)

    last_demand = float(ts[-1])
    mean7       = float(np.mean(ts[-7:])) if lookback >= 7 else float(np.mean(ts))
    mean_all    = float(np.mean(ts))
    std_all     = float(np.std(ts))
    zero_ratio  = float(np.mean(ts == 0))
    slope       = float(np.polyfit(np.arange(lookback), ts, 1)[0]) if lookback > 1 else 0.0
    min_v       = float(np.min(ts))
    max_v       = float(np.max(ts))
    stats = np.array([last_demand, mean7, mean_all, std_all, zero_ratio, slope, min_v, max_v],
                     dtype=np.float32)

    cal = np.array(cal_next_step, dtype=np.float32)
    return np.concatenate([ts, cal, stats])


def compute_neighbor_node_features_pure(ts_window, feature_dim):
    """
    Builds a neighbor node feature vector padded to `feature_dim`.

    Layout: [zeros_(feature_dim-8) right-filled with ts_window | stats_8]
    Calendar positions are left as zero (neighbor calendar is unknown / irrelevant).

    feature_dim = lookback + cal_dim + 8
    """
    ts = np.array(ts_window, dtype=np.float32)
    seq_len = len(ts)

    last_demand = float(ts[-1]) if seq_len > 0 else 0.0
    mean7       = float(np.mean(ts[-7:])) if seq_len >= 7 else (float(np.mean(ts)) if seq_len > 0 else 0.0)
    mean_all    = float(np.mean(ts)) if seq_len > 0 else 0.0
    std_all     = float(np.std(ts))  if seq_len > 0 else 0.0
    zero_ratio  = float(np.mean(ts == 0)) if seq_len > 0 else 0.0
    slope       = float(np.polyfit(np.arange(seq_len), ts, 1)[0]) if seq_len > 1 else 0.0
    min_v       = float(np.min(ts)) if seq_len > 0 else 0.0
    max_v       = float(np.max(ts)) if seq_len > 0 else 0.0
    stats = np.array([last_demand, mean7, mean_all, std_all, zero_ratio, slope, min_v, max_v],
                     dtype=np.float32)

    # Right-align ts values inside the (feature_dim - 8) non-stats slots
    ts_part = np.zeros(feature_dim - 8, dtype=np.float32)
    fill_len = min(seq_len, feature_dim - 8)
    ts_part[-fill_len:] = ts[-fill_len:]

    return np.concatenate([ts_part, stats])



# ---------------------------------------------------------------------------
# Dynamic named-feature builder (mirrors graphsage_pyg.generate_node_features)
# ---------------------------------------------------------------------------

def generate_node_features(ts, cal_next=None, cal_lookback=None, selected_features=None,
                           is_neighbor=False, pad_ts_to=None, cal_columns=None):
    """
    Build a node feature vector from a named feature list.

    Mirrors graphsage_pyg.generate_node_features exactly so that the same
    NODE_FEATURES list can be shared between the GraphSAGE and SimpleGNN models.

    Parameters
    ----------
    ts               : 1-D array-like  – raw time-series window for this node
    cal_next         : 1-D array-like  – calendar features for the next step
    cal_lookback     : 2-D array-like  – (lookback × cal_dim) calendar matrix
    selected_features: list[str]       – ordered feature names to include
    is_neighbor      : bool            – neighbor nodes receive zeros for calendar/lookback
    pad_ts_to        : int or None     – if set, neighbor ts is right-padded with zeros
    cal_columns      : list[str] or None – column names in cal_next, enabling per-column keys
    """
    ts_arr  = np.array(ts, dtype=np.float32)
    seq_len = len(ts_arr)
    feats   = []

    def get_ts():
        if is_neighbor and pad_ts_to is not None:
            out = np.zeros(pad_ts_to, dtype=np.float32)
            fill = min(seq_len, pad_ts_to)
            if fill > 0:
                out[-fill:] = ts_arr[-fill:]
            return out
        return ts_arr

    builders = {
        'ts': lambda: get_ts(),
        'cal_lookback': (
            lambda: np.zeros_like(np.asarray(cal_lookback, dtype=np.float32).reshape(-1))
            if is_neighbor else
            (np.asarray(cal_lookback, dtype=np.float32).reshape(-1)
             if cal_lookback is not None else np.array([]))
        ),
        'cal_next': (
            lambda: np.zeros_like(np.array(cal_next, dtype=np.float32))
            if is_neighbor else
            (np.array(cal_next, dtype=np.float32) if cal_next is not None else np.array([]))
        ),
        'last_demand': lambda: np.array([ts_arr[-1] if seq_len > 0 else 0.0], dtype=np.float32),
        'mean7': lambda: np.array([
            np.mean(ts_arr[-7:]) if seq_len >= 7 else (np.mean(ts_arr) if seq_len > 0 else 0.0)
        ], dtype=np.float32),
        'mean_all': lambda: np.array([np.mean(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'mean15':   lambda: np.array([np.mean(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'std_all':  lambda: np.array([np.std(ts_arr)  if seq_len > 0 else 0.0], dtype=np.float32),
        'std15':    lambda: np.array([np.std(ts_arr)  if seq_len > 0 else 0.0], dtype=np.float32),
        'zero_ratio':  lambda: np.array([np.mean(ts_arr == 0) if seq_len > 0 else 0.0], dtype=np.float32),
        'zero_ratio15':lambda: np.array([np.mean(ts_arr == 0) if seq_len > 0 else 0.0], dtype=np.float32),
        'slope': lambda: np.array([
            np.polyfit(np.arange(seq_len), ts_arr, 1)[0] if seq_len > 1 else 0.0
        ], dtype=np.float32),
        'slope15': lambda: np.array([
            np.polyfit(np.arange(seq_len), ts_arr, 1)[0] if seq_len > 1 else 0.0
        ], dtype=np.float32),
        'min_v':   lambda: np.array([np.min(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'max_v':   lambda: np.array([np.max(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
    }

    if cal_columns is not None:
        cal_next_arr = (np.array(cal_next, dtype=np.float32) if cal_next is not None else None)
        for _idx, _col in enumerate(cal_columns):
            if is_neighbor or cal_next_arr is None:
                builders[_col] = lambda: np.array([0.0], dtype=np.float32)
            else:
                builders[_col] = (lambda _i=_idx: np.array([cal_next_arr[_i]], dtype=np.float32))

    for feat in (selected_features or []):
        if feat in builders:
            v = builders[feat]()
            if v.size > 0:
                feats.append(v)

    if not feats:
        return np.array([], dtype=np.float32)
    return np.concatenate(feats)


def compute_target_node_features_seq(ts_window, cal_at_t, feature_dim):
    """
    Target-node features for the sequential (SAGE+LSTM) approach.

    Layout: [ts_window (right-aligned, gw slots) | cal_at_t (cal_dim) | stats_8]
    where  gw = feature_dim - cal_dim - 8  (= graph_window_size).

    feature_dim = graph_window_size + cal_dim + 8
    """
    ts  = np.array(ts_window, dtype=np.float32)
    cal = np.array(cal_at_t,  dtype=np.float32)
    seq_len = len(ts)
    cal_dim = len(cal)
    gw      = feature_dim - cal_dim - 8

    ts_part = np.zeros(gw, dtype=np.float32)
    fill    = min(seq_len, gw)
    ts_part[-fill:] = ts[-fill:]

    last_demand = float(ts[-1]) if seq_len > 0 else 0.0
    mean7       = float(np.mean(ts[-7:])) if seq_len >= 7 else (float(np.mean(ts)) if seq_len > 0 else 0.0)
    mean_all    = float(np.mean(ts))  if seq_len > 0 else 0.0
    std_all     = float(np.std(ts))   if seq_len > 0 else 0.0
    zero_ratio  = float(np.mean(ts == 0)) if seq_len > 0 else 0.0
    slope       = float(np.polyfit(np.arange(seq_len), ts, 1)[0]) if seq_len > 1 else 0.0
    min_v       = float(np.min(ts)) if seq_len > 0 else 0.0
    max_v       = float(np.max(ts)) if seq_len > 0 else 0.0
    stats = np.array([last_demand, mean7, mean_all, std_all, zero_ratio, slope, min_v, max_v],
                     dtype=np.float32)

    return np.concatenate([ts_part, cal, stats])


class SimpleGNNLSTMForecaster(nn.Module):
    """
    GCN encoder + LSTM forecaster.

    Mirrors GraphSAGELSTMForecaster (graphsage_pyg.py) with GCNConv instead of SAGEConv.

    The GCN processes ONE ego-graph per sample (the graph whose window ends at
    the last observation of the lookback period).  The target-node embedding z
    is projected to initialise the LSTM hidden and cell states (h₀, c₀), giving
    the sequential model cross-item graph context from the first step.

    The LSTM then processes the explicit lookback sequence:
        ts_seq[:, t, :] = [value_t, cal_{t+1}]
    i.e. at each step the LSTM sees the current scaled target value and the
    calendar features of the *next* day, consistent with training labels.

    Forward inputs
    --------------
    pyg_batch           : PyG Batch of B graphs (ONE per sample)
    target_node_indices : LongTensor (B,) – use pyg_batch.ptr[:-1]
    ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)

    Returns : (B, horizon, 1)
    """

    def __init__(
        self,
        in_channels: int,       # node feature dim (computed from generate_node_features)
        hidden_channels: int,
        out_channels: int,      # GCN output channels = LSTM init projection input
        lstm_input_size: int,   # 1 + cal_dim
        lstm_hidden: int,
        lstm_layers: int,
        horizon: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.activation = nn.ReLU()
        self.gnn_drop   = nn.Dropout(p=dropout)
        # Project graph embedding → LSTM initial hidden + cell states
        self.h_proj = nn.Linear(out_channels, lstm_hidden * lstm_layers)
        self.c_proj = nn.Linear(out_channels, lstm_hidden * lstm_layers)
        # LSTM over the lookback sequence
        self.lstm = nn.LSTM(
            lstm_input_size, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_drop  = nn.Dropout(p=dropout)
        self.head       = nn.Linear(lstm_hidden, horizon)
        self.horizon    = horizon
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        pyg_batch           : PyG Batch of B graphs
        target_node_indices : LongTensor (B,)
        ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)

        Returns : (B, horizon, 1)
        """
        ew = (pyg_batch.edge_attr.squeeze(1)
              if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.shape[0] > 0)
              else None)
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_weight=ew)
        h = self.activation(h)
        h = self.gnn_drop(h)
        h = self.conv2(h, pyg_batch.edge_index, edge_weight=ew)

        z = h[target_node_indices]   # (B, out_channels)
        B = z.size(0)

        h0 = (self.h_proj(z)
                  .view(B, self.lstm_layers, self.lstm_hidden)
                  .permute(1, 0, 2).contiguous())   # (layers, B, lstm_hidden)
        c0 = (self.c_proj(z)
                  .view(B, self.lstm_layers, self.lstm_hidden)
                  .permute(1, 0, 2).contiguous())   # (layers, B, lstm_hidden)

        lstm_out, _ = self.lstm(ts_seq, (h0, c0))   # (B, lookback, lstm_hidden)
        out = self.lstm_drop(lstm_out[:, -1, :])     # (B, lstm_hidden)
        pred = self.head(out)                        # (B, horizon)
        return pred.unsqueeze(-1)                    # (B, horizon, 1)
