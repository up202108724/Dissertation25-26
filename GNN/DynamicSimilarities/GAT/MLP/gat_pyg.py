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
    """Two-layer GAT encoder. Edge attributes (similarity weights) are fed
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
