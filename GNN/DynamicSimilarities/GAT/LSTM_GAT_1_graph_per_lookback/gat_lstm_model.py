"""
Simple GAT + LSTM forecaster — trainable replacement for the
offline Graph2Vec embedding pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SimpleGATLSTMForecaster(nn.Module):
    """
    GAT encoder (per-window ego-graph) → broadcast → concat with ts_seq → LSTM → linear.

    Parameters
    ----------
    in_channels      : int   – node-feature dimension fed to the GAT
    gat_hidden       : int   – hidden width of GATConv₁ (per attention head)
    d_g              : int   – GAT output dim (target-node embedding size)
    lstm_input_size  : int   – per-step temporal feature width (e.g. 1 + n_exog)
    lstm_hidden      : int   – LSTM hidden size
    lstm_layers      : int   – number of stacked LSTM layers
    horizon          : int   – model horizon (1 → wrap recursively at inference)[cite: 2]
    dropout          : float – dropout for GAT + LSTM + head
    attention_heads  : int   – number of parallel attention heads in the GAT
    """

    def __init__(
        self,
        in_channels: int,
        gat_hidden: int = 32,
        d_g: int = 16,
        lstm_input_size: int = 1,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        horizon: int = 1,
        dropout: float = 0.2,
        attention_heads: int = 4, # Added parameter for GAT
    ):
        super().__init__()

        # ── GAT encoder ───────────────────────────────────────────────────
        # Hidden layer concatenates heads, so output dim is gat_hidden * attention_heads
        self.conv1 = GATConv(
            in_channels, 
            gat_hidden, 
            heads=attention_heads, 
            concat=True, 
            dropout=dropout,
            edge_dim=1 # Expects 1D edge_attr from the dataset
        )
        
        # Output layer averages heads (concat=False), ensuring output is exactly d_g
        self.conv2 = GATConv(
            gat_hidden * attention_heads, 
            d_g, 
            heads=1, 
            concat=False, 
            dropout=dropout,
            edge_dim=1
        )
        
        self.activation = nn.ReLU()
        self.gnn_drop   = nn.Dropout(p=dropout)
        self.z_norm     = nn.LayerNorm(d_g)

        # ── LSTM over augmented sequence ──────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size + d_g,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(p=dropout)
        self.head      = nn.Linear(lstm_hidden, horizon)

        self.d_g     = d_g
        self.horizon = horizon

        # diagnostic switch — when True the GAT embedding is zeroed out[cite: 2]
        self.ablate_z = False

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        PER-STEP variant: one ego-graph per lookback day.
        """
        B, L, _ = ts_seq.shape

        if self.ablate_z:
            z_seq = torch.zeros(B, L, self.d_g, device=ts_seq.device, dtype=ts_seq.dtype)
        else:
            # Pass edge_attr instead of edge_weight for GAT
            edge_attr = pyg_batch.edge_attr if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.numel() > 0) else None

            h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_attr=edge_attr)
            h = self.activation(h)
            h = self.gnn_drop(h)
            h = self.conv2(h, pyg_batch.edge_index, edge_attr=edge_attr)
            z_flat = self.z_norm(h[target_node_indices])           # (B*L, d_g)

            if z_flat.shape[0] != B * L:
                raise RuntimeError(
                    f"Expected {B*L} target indices for per-step model, got {z_flat.shape[0]}. "
                    "Make sure you are using the per-step collate that returns "
                    "(pyg_batch, ts_batch, y_batch, target_idx, L)."
                )

            z_seq = z_flat.view(B, L, self.d_g)

        # per-step LSTM input = [ y_t  ||  exog_t  ||  z ]   shape (B, L, 1 + n_exog + d_g)[cite: 2]
        x = torch.cat([ts_seq, z_seq], dim=-1)                 

        # LSTM + head
        out, _ = self.lstm(x)
        last   = self.lstm_drop(out[:, -1, :])                 # (B, hidden)
        pred   = self.head(last)                               # (B, horizon)
        return pred.unsqueeze(-1)                              # (B, horizon, 1)