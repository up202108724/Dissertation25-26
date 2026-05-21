import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

def generate_node_features(ts, cal_next=None, cal_lookback=None, selected_features=None, is_neighbor=False, pad_ts_to=None, cal_columns=None):
    """
    Builder function to generate node features dynamically based on selected_features.
    
    selected_features can include:
    'ts'           : the raw time-series sequence (padded for neighbors if pad_ts_to is set).
    'cal_lookback' : calendar features for the lookback window (flattened). Ignored/zeroed for neighbors.
    'cal_next'     : all calendar features for the next step as a single vector. Ignored/zeroed for neighbors.
    'last_demand'  : last sequence value.
    'mean7'        : mean of last 7 days.
    'mean_all'     : mean of the entire sequence.
    'std_all'      : standard deviation of the sequence.
    'zero_ratio'   : ratio of zeros.
    'slope'        : linear slope.
    'min_v'        : minimum value.
    'max_v'        : maximum value.
    '<col_name>'   : any individual calendar column name present in cal_columns (e.g. 'dow_sin',
                     'is_weekend'). Ignored/zeroed for neighbors.
    
    Parameters
    ----------
    cal_columns : list[str], optional
        Ordered list of column names in cal_next / cal_lookback rows.
        When provided, each column name is available as an individual feature key.
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
        'max_v': lambda: np.array([np.max(ts_arr) if seq_len > 0 else 0.0], dtype=np.float32),
    }

    # Dynamically add per-column calendar feature builders when cal_columns is provided.
    # Each column maps to a single scalar extracted from cal_next by its index.
    # Neighbors always receive 0.0 for calendar features (cal_next may be None for them).
    if cal_columns is not None:
        cal_next_arr = (np.array(cal_next, dtype=np.float32)
                        if cal_next is not None else None)
        for _idx, _col in enumerate(cal_columns):
            if is_neighbor or cal_next_arr is None:
                builders[_col] = lambda: np.array([0.0], dtype=np.float32)
            else:
                builders[_col] = (lambda _i=_idx: np.array([cal_next_arr[_i]], dtype=np.float32))

    for feat in selected_features:
        if feat in builders:
            feat_val = builders[feat]()
            if feat_val.size > 0:
                feats.append(feat_val)
            
    if not feats:
        return np.array([], dtype=np.float32)
    return np.concatenate(feats)


class GraphSAGELSTMForecaster(nn.Module):
    """
    GraphSAGE + LSTM forecaster.

    The GraphSAGE encoder processes the ego-graph and produces a target-node
    embedding z (shape B × sage_out_ch).  z is projected to initialize both
    the hidden state h_0 and cell state c_0 of an LSTM, giving the sequential
    model cross-item graph context from the start.

    The LSTM then processes the raw lookback sequence as its input:
        ts_seq[:, t, :] = [value_t, cal_{t+1}]
    i.e. at each step the LSTM sees the current scaled value and the calendar
    features of the *next* day.  The final LSTM hidden state is projected to
    the horizon prediction via a linear head.

    Forward inputs
    --------------
    pyg_batch           : PyG Batch (B graphs merged)
    target_node_indices : LongTensor (B,) – global index of the target node
                          in each graph (always node 0, so use pyg_batch.ptr[:-1])
    ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        lstm_input_size: int,   # 1 + cal_dim
        lstm_hidden: int,
        lstm_layers: int,
        horizon: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        self.sage_drop = nn.Dropout(p=dropout)
        # Project graph embedding → LSTM initial hidden + cell states
        self.h_proj = nn.Linear(out_channels, lstm_hidden * lstm_layers)
        self.c_proj = nn.Linear(out_channels, lstm_hidden * lstm_layers)
        # LSTM over the lookback sequence
        self.lstm = nn.LSTM(
            lstm_input_size, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(lstm_hidden, horizon)
        self.horizon = horizon
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        pyg_batch           : PyG Batch of B graphs
        target_node_indices : LongTensor (B,)
        ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)

        Returns : (B, horizon, 1)
        """
        # 1. GraphSAGE → target-node embedding z
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index)
        h = self.activation(h)
        h = self.sage_drop(h)
        h = self.conv2(h, pyg_batch.edge_index)
        z = h[target_node_indices]   # (B, out_channels)

        # 2. Project z to LSTM initial states
        B = z.size(0)
        h0 = (self.h_proj(z)
                  .view(B, self.lstm_layers, self.lstm_hidden)
                  .permute(1, 0, 2).contiguous())   # (layers, B, lstm_hidden)
        c0 = (self.c_proj(z)
                  .view(B, self.lstm_layers, self.lstm_hidden)
                  .permute(1, 0, 2).contiguous())   # (layers, B, lstm_hidden)

        # 3. LSTM over the lookback sequence, conditioned on graph context
        lstm_out, _ = self.lstm(ts_seq, (h0, c0))   # (B, lookback, lstm_hidden)
        out = self.lstm_drop(lstm_out[:, -1, :])      # (B, lstm_hidden) – last step

        # 4. Linear head → forecast
        pred = self.head(out)          # (B, horizon)
        return pred.unsqueeze(-1)      # (B, horizon, 1)
