import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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
    # Neighbors always receive 0.0 for calendar features.
    if cal_columns is not None and cal_next is not None:
        cal_next_arr = np.array(cal_next, dtype=np.float32)
        for _idx, _col in enumerate(cal_columns):
            if is_neighbor:
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



class GCNMLPForecaster(nn.Module):
    """
    GCN + MLP forecaster — FiLM conditional fusion.

    Instead of concatenating [z || ts_enc] and feeding the result to an MLP
    (which can learn to ignore the graph branch entirely), the graph embedding
    ``z`` is used to produce per-feature affine parameters (gamma, beta) that
    modulate the ts representation:

        ts_mod = gamma(z) * LayerNorm(ts_enc) + beta(z)

    The graph branch therefore enters the prediction multiplicatively and
    cannot be trivially zeroed out by the head. ``gamma`` is initialised to 1
    and ``beta`` to 0, so at the start of training the model behaves like the
    pure ts-MLP baseline and the graph branch only contributes once it learns
    useful modulation.

    Forward inputs
    --------------
    pyg_batch           : PyG Batch (B graphs merged)
    target_node_indices : LongTensor (B,) – global index of the target node
                          in each graph (always node 0, so use pyg_batch.ptr[:-1])
    ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)
    """

    def __init__(
        self,
        in_channels: int,           # node feature dim
        hidden_channels: int,       # GCNConv hidden dim
        out_channels: int,          # GCNConv output dim (recommend 64)
        ts_input_size: int,         # lookback * (1 + cal_dim) – flattened ts_seq dim
        mlp_hidden_sizes: list,     # e.g. [64, 32]
        horizon: int,
        dropout: float = 0.2,
        ts_proj_dim: int = 64,      # projects flat_ts to this dim
        film_hidden: int = 64,      # hidden dim of the FiLM generator
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.activation = nn.ReLU()
        self.gcn_drop = nn.Dropout(p=dropout)

        # Project flat ts to ts_proj_dim
        self.ts_proj = nn.Linear(ts_input_size, ts_proj_dim)

        # LayerNorm both branches before fusion
        self.z_norm  = nn.LayerNorm(out_channels)
        self.ts_norm = nn.LayerNorm(ts_proj_dim)

        # FiLM generator: z -> (gamma, beta) over ts_proj_dim features.
        # gamma is parameterised as 1 + delta so it starts at the identity
        # (delta init ~ 0), making the fused representation initially equal to
        # the ts branch alone.
        self.film = nn.Sequential(
            nn.Linear(out_channels, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * ts_proj_dim),
        )
        # Zero-init the final FiLM layer so gamma=1, beta=0 at step 0.
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # MLP head over the modulated ts representation (same width as the
        # ts branch — no concat).
        layers = []
        prev = ts_proj_dim
        for h in mlp_hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout)]
            prev = h
        layers.append(nn.Linear(prev, horizon))
        self.mlp = nn.Sequential(*layers)
        self.horizon = horizon
        self.ts_proj_dim = ts_proj_dim

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        pyg_batch           : PyG Batch of B graphs
        target_node_indices : LongTensor (B,)
        ts_seq              : FloatTensor (B, lookback, 1 + cal_dim)

        Returns : (B, horizon, 1)
        """
        # 1. GCN → target-node embedding z
        ea = pyg_batch.edge_attr
        ew = ea.squeeze(1) if (ea is not None and ea.shape[0] > 0) else None
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_weight=ew)
        h = self.activation(h)
        h = self.gcn_drop(h)
        h = self.conv2(h, pyg_batch.edge_index, edge_weight=ew)
        z = h[target_node_indices]              # (B, out_channels)
        z = self.z_norm(z)

        # 2. Project ts branch and normalise it
        B = z.size(0)
        ts_flat = ts_seq.view(B, -1)
        ts_enc  = self.activation(self.ts_proj(ts_flat))   # (B, ts_proj_dim)
        ts_enc  = self.ts_norm(ts_enc)

        # 3. FiLM modulation: gamma, beta come from z
        gb = self.film(z)                                  # (B, 2 * ts_proj_dim)
        delta_gamma, beta = gb.chunk(2, dim=-1)            # each (B, ts_proj_dim)
        gamma = 1.0 + delta_gamma                          # init at identity
        ts_mod = gamma * ts_enc + beta                     # (B, ts_proj_dim)

        # 4. MLP head → forecast
        pred = self.mlp(ts_mod)                            # (B, horizon)
        return pred.unsqueeze(-1)                          # (B, horizon, 1)


# Backward-compatibility alias
GraphSAGEMLPForecaster = GCNMLPForecaster
