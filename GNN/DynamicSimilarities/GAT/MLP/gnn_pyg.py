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
    Features are computed on a z-score normalized copy so the GCN input is on
    the same scale as the scaled ts_seq / cal_seq fed to the MLP.
    """
    ts_sequence = np.array(ts_sequence, dtype=np.float32)
    seq_len = len(ts_sequence)

    # Normalize raw ts to zero-mean unit-variance so GCN output stays ~O(1)
    ts_mean = np.mean(ts_sequence)
    ts_std  = np.std(ts_sequence)
    if ts_std > 1e-8:
        ts_norm = (ts_sequence - ts_mean) / ts_std
    else:
        ts_norm = ts_sequence - ts_mean  # all-zero or constant series

    # 1. Normalized Raw Sequence
    raw_seq = ts_norm.tolist()

    # 2. Statistical Features (computed on original scale for interpretability,
    #    then also normalized by ts_std so they stay bounded)
    denom = ts_std if ts_std > 1e-8 else 1.0
    last_demand  = (ts_sequence[-1] - ts_mean) / denom if seq_len > 0 else 0.0
    mean7        = (np.mean(ts_sequence[-7:]) - ts_mean) / denom if seq_len >= 7 else 0.0
    mean15       = 0.0  # by definition after z-score
    std15        = 1.0 if ts_std > 1e-8 else 0.0  # by definition
    zero_ratio15 = float(np.mean(ts_sequence == 0)) if seq_len > 0 else 0.0

    if seq_len > 1:
        x = np.arange(seq_len, dtype=np.float32)
        slope15 = float(np.polyfit(x, ts_norm, 1)[0])  # slope of normalized series
    else:
        slope15 = 0.0

    min_15 = float(ts_norm.min()) if seq_len > 0 else 0.0
    max_15 = float(ts_norm.max()) if seq_len > 0 else 0.0

    stats = [last_demand, mean7, mean15, std15, zero_ratio15, slope15, min_15, max_15]

    # Total features: len(raw_seq) + 8  (same shape as before)
    return np.array(raw_seq + stats, dtype=np.float32)

class SimpleGNNEncoder(nn.Module):
    """Two-layer GCN encoder. Edge weights (similarity/distance scores stored
    in edge_attr) are passed as scalar multipliers to GCNConv."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # GCNConv expects a 1-D edge_weight; squeeze from (E, 1) if present
        ew = edge_attr.squeeze(1) if (edge_attr is not None and edge_attr.shape[0] > 0) else None
        h = self.conv1(x, edge_index, edge_weight=ew)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight=ew)
        return h
