import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

def generate_node_features(ts, cal_next=None, cal_lookback=None, selected_features=None, is_neighbor=False, pad_ts_to=None):
    """
    Builder function to generate node features dynamically based on selected_features.
    
    selected_features can include:
    'ts'           : the raw time-series sequence (padded for neighbors if pad_ts_to is set).
    'cal_lookback' : calendar features for the lookback window (flattened). Ignored/zeroed for neighbors.
    'cal_next'     : calendar features for the next step. Ignored/zeroed for neighbors.
    'last_demand'  : last sequence value.
    'mean7'        : mean of last 7 days.
    'mean_all'     : mean of the entire sequence.
    'std_all'      : standard deviation of the sequence.
    'zero_ratio'   : ratio of zeros.
    'slope'        : linear slope.
    'min_v'        : minimum value.
    'max_v'        : maximum value.
    """

    ts_arr = np.array(ts, dtype=np.float32)
    seq_len = len(ts_arr)

    # Initialize feats list
    feats = []

    # Helper function for padded neighbor ts
    def get_ts():
        if is_neighbor and pad_ts_to is not None:
            ts_part = np.zeros(pad_ts_to, dtype=np.float32)
            fill_len = min(seq_len, pad_ts_to)
            if fill_len > 0:
                ts_part[-fill_len:] = ts_arr[-fill_len:]
            return ts_part
        return ts_arr

    # Define builder dictionary for compute logic
    builders = {
        'ts': lambda: get_ts(),
        'cal_lookback': lambda: np.zeros_like(np.asarray(cal_lookback, dtype=np.float32).reshape(-1)) if is_neighbor else np.asarray(cal_lookback, dtype=np.float32).reshape(-1) if cal_lookback is not None else np.array([]),
        'cal_next': lambda: np.zeros_like(np.array(cal_next, dtype=np.float32)) if is_neighbor else np.array(cal_next, dtype=np.float32) if cal_next is not None else np.array([]),
        'last_demand': lambda: np.array([ts_arr[-1] if seq_len > 0 else 0.0], dtype=np.float32),
        'mean7': lambda: np.array([np.mean(ts_arr[-7:]) if seq_len >= 7 else (np.mean(ts_arr) if seq_len > 0 else 0.0)], dtype=np.float32),
        'mean_all': lambda: np.array([np.mean(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'mean15': lambda: np.array([np.mean(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'std_all': lambda: np.array([np.std(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'std15': lambda: np.array([np.std(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'zero_ratio': lambda: np.array([np.mean(ts_arr == 0) if seq_len > 0 else 0.0], dtype=np.float32),
        'zero_ratio15': lambda: np.array([np.mean(ts_arr == 0) if seq_len > 0 else 0.0], dtype=np.float32),
        'slope': lambda: np.array([np.polyfit(np.arange(seq_len), ts_arr, 1)[0] if seq_len > 1 else 0.0], dtype=np.float32),
        'slope15': lambda: np.array([np.polyfit(np.arange(seq_len), ts_arr, 1)[0] if seq_len > 1 else 0.0], dtype=np.float32),
        'min_v': lambda: np.array([np.min(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'min_15': lambda: np.array([np.min(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'max_v': lambda: np.array([np.max(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
        'max_15': lambda: np.array([np.max(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
    }

    for feat in selected_features:
        if feat in builders:
            feat_val = builders[feat]()
            if feat_val.size > 0:
                feats.append(feat_val)
            
    if not feats:
        return np.array([], dtype=np.float32)
    return np.concatenate(feats)


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
