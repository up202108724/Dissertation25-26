import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

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

class GATEncoder(nn.Module):
    """Two-layer GAT encoder.  Edge attributes (similarity weights) are fed
    into the attention mechanism when present."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads: int = 4, att_dropout: float = 0.0, dropout: float = 0.2):
        super(GATEncoder, self).__init__()
        # Conv1: multi-head attention; output dim = hidden_channels * heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             concat=True, dropout=att_dropout, edge_dim=1,
                             add_self_loops=True)
        # Conv2: single-head, output dim = out_channels
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=att_dropout, edge_dim=1,
                             add_self_loops=True)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        ea = edge_attr if (edge_attr is not None and edge_attr.shape[0] > 0) else None
        h = self.conv1(x, edge_index, edge_attr=ea)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr=ea)
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


class PureGATForecaster(nn.Module):
    """
    Pure GAT forecaster.

    Replaces the GraphSAGE uniform mean-aggregation with attention-weighted
    message passing.  For the small ego-graphs produced by the similarity
    threshold (typically 1-3 neighbours), this means the model *learns* how
    much to trust each correlated product rather than blending them equally.

    Node feature layout (identical to the old SAGE version):
      - Target node : full lookback window  +  next-step calendar  +  8 stats
      - Neighbor nodes: graph_window_size values (right-aligned, zero-padded)
                        +  8 stats

    The edge weights (similarity/distance scores stored in edge_attr) are fed
    into the attention mechanism via edge_dim=1, giving the attention heads
    additional information about *how* correlated each neighbor is today.

    After two GATConv layers, the central node's embedding is projected to
    `horizon` predictions via a single linear layer.
    """

    def __init__(
        self,
        in_channels: int,        # lookback + cal_dim + 8
        hidden_channels: int,
        out_channels: int,
        horizon: int,
        heads: int = 4,
        att_dropout: float = 0.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Layer 1: multi-head attention → hidden_channels * heads
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=heads, concat=True,
            dropout=att_dropout, edge_dim=1, add_self_loops=True,
        )
        # Layer 2: single head → out_channels (final per-node embedding)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels,
            heads=1, concat=False,
            dropout=att_dropout, edge_dim=1, add_self_loops=True,
        )
        self.activation = nn.ELU()
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
        x          = pyg_batch.x
        edge_index = pyg_batch.edge_index
        edge_attr  = pyg_batch.edge_attr

        # Pass edge_attr only when edges actually exist; GATConv handles self-loops
        ea = edge_attr if (edge_attr is not None and edge_attr.shape[0] > 0) else None

        h = self.conv1(x, edge_index, edge_attr=ea)   # (N, hidden * heads)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr=ea)   # (N, out_channels)

        z   = h[target_node_indices]   # (B, out_channels)
        out = self.head(z)             # (B, horizon)
        return out.unsqueeze(-1)       # (B, horizon, 1)
