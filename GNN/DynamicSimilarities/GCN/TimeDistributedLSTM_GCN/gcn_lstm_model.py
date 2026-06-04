"""
Simple GCN + LSTM forecaster — trainable replacement for the
offline Graph2Vec embedding pipeline.

For each sliding window we build a small ego-graph (target item +
similar neighbours).  A 2-layer GCN encodes that graph into a single
vector ``z`` for the target node, which is then broadcast across the
LSTM lookback and concatenated with the temporal/exogenous features:

    per-step LSTM input = [ y_t  ||  exog_t  ||  z ]   shape (B, L, 1 + n_exog + d_g)

The whole stack (GCN + LSTM + head) is optimised jointly on the
forecasting loss — no offline embeddings, no Doc2Vec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleGCNLSTMForecaster(nn.Module):
    """
    GCN encoder (per-window ego-graph) → broadcast → concat with ts_seq → LSTM → linear.

    Parameters
    ----------
    in_channels      : int   – node-feature dimension fed to the GCN
    gcn_hidden       : int   – hidden width of GCNConv₁
    d_g              : int   – GCN output dim (target-node embedding size)
    lstm_input_size  : int   – per-step temporal feature width (e.g. 1 + n_exog)
    lstm_hidden      : int   – LSTM hidden size
    lstm_layers      : int   – number of stacked LSTM layers
    horizon          : int   – model horizon (1 → wrap recursively at inference)
    dropout          : float – dropout for GCN + LSTM + head
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 32,
        d_g: int = 16,
        lstm_input_size: int = 1,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        horizon: int = 1,
        dropout: float = 0.2,
        time_distributed: bool = False,
    ):
        super().__init__()

        # ── GCN encoder ───────────────────────────────────────────────────
        self.conv1 = GCNConv(in_channels, gcn_hidden, add_self_loops=True)
        self.conv2 = GCNConv(gcn_hidden,  d_g,        add_self_loops=True)
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

        # bookkeeping
        self.d_g             = d_g
        self.horizon         = horizon
        self.time_distributed = time_distributed

        # diagnostic switch — when True the GCN embedding is zeroed out
        # before being concatenated with the temporal features (ablation).
        self.ablate_z = False

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        PER-STEP variant: one ego-graph per lookback day.

        pyg_batch           : torch_geometric.data.Batch of B*L ego-graphs,
                              row-major (sample0_step0, sample0_step1, ...,
                              sample0_step{L-1}, sample1_step0, ...).
        target_node_indices : LongTensor (B*L,)  — usually ``pyg_batch.ptr[:-1]``.
        ts_seq              : FloatTensor (B, L, lstm_input_size)
                              per-step [target ‖ exog_t]

        Returns
        -------
        FloatTensor (B, horizon, 1)
        """
        B, L, _ = ts_seq.shape

        if self.ablate_z:
            # Skip GCN entirely — output is zeroed, no point paying the cost.
            z_seq = torch.zeros(B, L, self.d_g, device=ts_seq.device, dtype=ts_seq.dtype)
        else:
            # GCN branch — encodes B*L graphs in a single sparse forward.
            ew = (
                pyg_batch.edge_attr.squeeze(-1)
                if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.numel() > 0)
                else None
            )
            h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_weight=ew)
            h = self.activation(h)
            h = self.gnn_drop(h)
            h = self.conv2(h, pyg_batch.edge_index, edge_weight=ew)
            z_flat = self.z_norm(h[target_node_indices])           # (B*L, d_g)

            if z_flat.shape[0] != B * L:
                raise RuntimeError(
                    f"Expected {B*L} target indices for per-step model, got {z_flat.shape[0]}. "
                    "Make sure you are using the per-step collate that returns "
                    "(pyg_batch, ts_batch, y_batch, target_idx, L)."
                )

            # (B*L, d_g) -> (B, L, d_g)  in the same row-major order produced by collate
            z_seq = z_flat.view(B, L, self.d_g)

        # Per-step concat with the temporal/exogenous features
        x = torch.cat([ts_seq, z_seq], dim=-1)                 # (B, L, in+d_g)

        # LSTM + head
        out, _ = self.lstm(x)
        if self.time_distributed:
            # apply head to every step → dense per-step supervision
            pred = self.head(self.lstm_drop(out))              # (B, L, horizon)
            # with horizon=1 this is already (B, L, 1); keep consistent shape
            return pred if pred.shape[-1] > 1 else pred        # (B, L, 1)
        else:
            last = self.lstm_drop(out[:, -1, :])               # (B, hidden)
            pred = self.head(last)                             # (B, horizon)
            return pred.unsqueeze(-1)                          # (B, horizon, 1)
