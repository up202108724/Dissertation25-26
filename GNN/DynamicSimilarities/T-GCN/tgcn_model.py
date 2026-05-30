"""
T-GCN (Temporal Graph Convolutional Network) forecaster — joint per-step
GCN encoder feeding a GRU temporal head.

Design choice (vs. the canonical Zhao-2019 T-GCN cell)
------------------------------------------------------
The original T-GCN replaces every dense input transform inside a GRU cell
with a GCN, producing a per-node hidden state that evolves over time on a
*static* graph (e.g. a road network).  In our setting the ego-graph is
genuinely dynamic — the neighbour set and topology change at every
sliding window.  Carrying a per-node hidden state across time is therefore
ill-defined: at step t+1 the node set is no longer the same as at step t.

We adopt the standard "GCN-encoder + GRU" formulation used in most
practical dynamic-graph T-GCN implementations:

    z_t = GCN(X_t, A_t)[target]                    # per-step spatial context
    u_t = [ y_t  ||  exog_t  ||  z_t ]
    h_t = GRUCell(u_t, h_{t-1})
    ŷ   = head(h_L)

The GCN sees a *different* edge_index/edge_attr at every step (true
dynamic topology), so the spatial path is fully T-GCN-style; the temporal
recurrence happens on the target-node embedding rather than across the
whole graph.

Forward signature is identical to ``SimpleGCNMLPForecaster`` so the
existing GCNTimeSeriesDataset / collate_pyg_ts / _recursive_forecast_gcn_perstep
pipeline works unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleTGCNForecaster(nn.Module):
    """
    Per-step GCN encoder → GRU over the L lookback steps → linear head.

    Parameters
    ----------
    in_channels    : int   – node-feature dimension fed to the GCN
    gcn_hidden     : int   – hidden width of GCNConv1
    d_g            : int   – GCN output dim (target-node embedding size)
    ts_input_size  : int   – per-step temporal feature width (1 + n_exog)
    gru_hidden     : int   – hidden size of the GRU cell
    num_gru_layers : int   – number of stacked GRU layers
    horizon        : int   – model horizon (1 → wrap recursively at inference)
    dropout        : float – dropout for GCN + GRU + head
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 32,
        d_g: int = 16,
        ts_input_size: int = 1,
        gru_hidden: int = 64,
        num_gru_layers: int = 1,
        horizon: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── GCN spatial encoder ──────────────────────────────────────────
        self.conv1 = GCNConv(in_channels, gcn_hidden, add_self_loops=True)
        self.conv2 = GCNConv(gcn_hidden,  d_g,        add_self_loops=True)
        self.activation = nn.ReLU()
        self.gnn_drop   = nn.Dropout(p=dropout)
        self.z_norm     = nn.LayerNorm(d_g)

        # ── GRU temporal head ────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=ts_input_size + d_g,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.head_drop = nn.Dropout(p=dropout)
        self.head      = nn.Linear(gru_hidden, horizon)

        # bookkeeping
        self.d_g           = d_g
        self.ts_input_size = ts_input_size
        self.gru_hidden    = gru_hidden
        self.horizon       = horizon

        # diagnostic switch — when True the GCN embedding is zeroed out
        # before being concatenated with the temporal features (ablation).
        self.ablate_z = False

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        pyg_batch           : Batch of B*L ego-graphs (row-major).
        target_node_indices : LongTensor (B*L,) = pyg_batch.ptr[:-1].
        ts_seq              : (B, L, ts_input_size)

        Returns
        -------
        (B, horizon, 1)
        """
        B, L, _ = ts_seq.shape

        if self.ablate_z:
            z_seq = torch.zeros(B, L, self.d_g, device=ts_seq.device, dtype=ts_seq.dtype)
        else:
            ew = (
                pyg_batch.edge_attr.squeeze(-1)
                if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.numel() > 0)
                else None
            )
            h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_weight=ew)
            h = self.activation(h)
            h = self.gnn_drop(h)
            h = self.conv2(h, pyg_batch.edge_index, edge_weight=ew)
            z_flat = self.z_norm(h[target_node_indices])     # (B*L, d_g)

            if z_flat.shape[0] != B * L:
                raise RuntimeError(
                    f"Expected {B*L} target indices for per-step model, "
                    f"got {z_flat.shape[0]}."
                )
            z_seq = z_flat.view(B, L, self.d_g)

        x = torch.cat([ts_seq, z_seq], dim=-1)               # (B, L, ts_in + d_g)
        out, _ = self.gru(x)                                  # (B, L, gru_hidden)
        last = self.head_drop(out[:, -1, :])                  # (B, gru_hidden)
        pred = self.head(last)                                # (B, horizon)
        return pred.unsqueeze(-1)                             # (B, horizon, 1)
