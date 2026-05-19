from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Tuple, List
from torch_geometric.data import Data, Batch

class WindowGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, graphs: List[List[Data]]):
        self.X = torch.from_numpy(X)  # (N, L, C)
        self.y = torch.from_numpy(y)  # (N, H, C)
        self.graphs = graphs  # List of N sequences, each containing L Data objects

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.graphs[idx]

def py_geometric_collate(batch):
    """
    Custom collate function to handle standard PyTorch tensors and PyG Data objects together.
    Flattens the sequences of graphs into a single PyG Batch.
    """
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    # Flatten the graphs: batch of size B, each sequence has L graphs -> list of B*L graphs
    flat_graphs = []
    for item in batch:
        flat_graphs.extend(item[2])
        
    graphs_batch = Batch.from_data_list(flat_graphs)
    
    return X_batch, y_batch, graphs_batch

def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    graphs: List[Data] = None,
    graph_window_size: int = 15
) -> Tuple[np.ndarray, np.ndarray, List[List[Data]]]:

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]  # (T, 1)

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    X = np.zeros((N, lookback, C), dtype=np.float32)
    y = np.zeros((N, horizon, C), dtype=np.float32)
    
    window_graphs = []

    # Handle graph padding (just like Graph2Vec zeros)
    # The first few steps don't have enough history for a graph, so we add dummy isolated graphs.
    padded_graphs = []
    feature_dim = graph_window_size  # matches sage_in_channels passed by caller
    if graphs is not None and len(graphs) > 0:
        # Infer real feature dim from first non-dummy graph if available
        feature_dim = graphs[0].x.shape[1]
    if graphs is not None:
        num_missing = T - len(graphs)
        if num_missing > 0:
            dummy_x = torch.zeros((1, feature_dim), dtype=torch.float32)
            dummy_graph = Data(
                x=dummy_x, 
                edge_index=torch.empty((2,0), dtype=torch.long),
                edge_attr=torch.empty((0,1), dtype=torch.float),
                central_node_idx=0
            )
            padded_graphs = [dummy_graph for _ in range(num_missing)] + graphs
        else:
            padded_graphs = graphs
    else:
        dummy_x = torch.zeros((1, feature_dim), dtype=torch.float32)
        dummy_graph = Data(
            x=dummy_x, 
            edge_index=torch.empty((2,0), dtype=torch.long),
            edge_attr=torch.empty((0,1), dtype=torch.float),
            central_node_idx=0
        )
        padded_graphs = [dummy_graph for _ in range(T)]

    exog_indices = [idx for idx in range(C) if idx != target_channel]

    for i in range(N):
        window_base = series[i : i + lookback].copy()
        
        # Shift exogenous variables forward by 1 so the model sees the target day's exog features
        # X[i] target corresponds to steps i ... i+lookback-1
        # X[i] exog corresponds to steps i+1 ... i+lookback
        if len(exog_indices) > 0:
            window_base[:, exog_indices] = series[i + 1 : i + lookback + 1, exog_indices]
            
        X[i] = window_base
        y[i] = series[i + lookback : i + lookback + horizon]

        # Extract the sequence of graphs for this specific lookback window
        seq_graphs = padded_graphs[i : i + lookback]
        window_graphs.append(seq_graphs)

    return X, y, window_graphs