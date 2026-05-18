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
