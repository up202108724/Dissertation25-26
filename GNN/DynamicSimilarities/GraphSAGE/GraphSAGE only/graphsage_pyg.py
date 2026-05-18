import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

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

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEEncoder, self).__init__()
        # SAGEConv aggregates neighborhood features.
        # in_channels: size of your node features (e.g., 15 raw + 8 stats = 23)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        # 1st SAGE Layer
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        
        # 2nd SAGE Layer
        h = self.conv2(h, edge_index)
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


class PureGraphSAGEForecaster(nn.Module):
    """
    Pure GraphSAGE forecaster.

    All temporal and spatial context is encoded directly in the node features:
      - Target node : full lookback window  +  next-step calendar  +  8 stats
      - Neighbor nodes: graph_window_size values (right-aligned, zero-padded) +  8 stats

    After two SAGEConv layers, the central node's embedding is projected to `horizon`
    predictions via a single linear layer – no external MLP needed.
    """

    def __init__(
        self,
        in_channels: int,        # lookback + cal_dim + 8
        hidden_channels: int,
        out_channels: int,
        horizon: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(out_channels, horizon)
        self.horizon = horizon

    def forward(self, pyg_batch, target_node_indices):
        """
        pyg_batch           : PyG Batch of B graphs (one graph per sample)
        target_node_indices : LongTensor (B,) – global index of the central node
                              in each graph.  Use pyg_batch.ptr[:-1] since the
                              central node is always node 0 in every graph.

        Returns : (B, horizon, 1)
        """
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h, pyg_batch.edge_index)

        z   = h[target_node_indices]   # (B, out_channels)
        out = self.head(z)             # (B, horizon)
        return out.unsqueeze(-1)       # (B, horizon, 1)
