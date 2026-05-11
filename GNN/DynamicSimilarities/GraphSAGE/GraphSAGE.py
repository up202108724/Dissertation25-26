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

def convert_nx_to_pyg(nx_graph, df_wide, window_start_date, window_end_date):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    Requires df_wide to fetch the actual historical time-series sequences.
    """
    # Create mapping from node ID to index (0 to N-1) for PyG
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    reverse_mapping = {i: node for node, i in node_mapping.items()}
    
    # 1. Extract Node Features
    # Determine the date columns for the current window to fetch the raw sequences
    date_cols = df_wide.columns.astype(str).tolist()
    
    try:
        start_idx = date_cols.index(str(window_start_date))
        end_idx = date_cols.index(str(window_end_date)) + 1
        window_cols = date_cols[start_idx:end_idx]
    except ValueError:
        raise ValueError(f"Dates {window_start_date} or {window_end_date} not found in df_wide columns.")

    x_list = []
    for i in range(len(node_mapping)):
        item_id = reverse_mapping[i]
        # Fetch the 15-day sequence for this item
        ts_sequence = df_wide.loc[item_id, window_cols].values
        
        # Compute the rich feature vector
        features = compute_node_features(ts_sequence)
        x_list.append(features)
        
    x = torch.tensor(np.vstack(x_list), dtype=torch.float)
    
    # 2. Extract Edges (PyG expects edge_index to be shape [2, num_edges])
    edge_list = []
    edge_weights = []
    for u, v, data in nx_graph.edges(data=True):
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        
        # Add both directions for an undirected graph
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
        
        weight = data.get('weight', 1.0)
        edge_weights.extend([weight, weight])
        
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    else:
        # Handling disconnected graph case
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.float)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_mapping=node_mapping)

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