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
