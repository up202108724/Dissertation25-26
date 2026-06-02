"""
Joint GCN + MLP forecaster — trainable replacement for the offline
Graph2Vec → MLP pipeline.

For each lookback day we build a small ego-graph (target item + similar
neighbours).  A 2-layer GCN encodes that graph into a vector ``z`` for the
target node.  The L per-step embeddings are concatenated with the temporal
features along the feature axis, the whole (L, 1 + n_exog + d_g) window is
flattened, and a flat MLP regresses the next value.

    per-step features = [ y_t  ||  exog_t  ||  z_t ]   shape (B, L, 1 + n_exog + d_g)
    time-distributed  = shared MLP applied to each of the L steps
    pooling           = cat[last_step, mean_over_L]     shape (B, 2H)
    linear head       → (B, horizon)

The whole stack (GCN + MLP) is optimised jointly on the forecasting loss —
no offline embeddings, no Doc2Vec.

The forward signature ``(pyg_batch, target_node_indices, ts_seq) -> (B, H, 1)``
matches ``SimpleGCNLSTMForecaster`` so callers can reuse the GCNTimeSeriesDataset
+ collate_pyg_ts + recursive_forecast pipeline from the LSTM sibling unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleGCNMLPForecaster(nn.Module):
    """
    GCN encoder (per-window ego-graph) → broadcast → concat with ts_seq
    → flatten → MLP → linear head.

    Parameters
    ----------
    in_channels      : int   – node-feature dimension fed to the GCN
    gcn_hidden       : int   – hidden width of GCNConv₁
    d_g              : int   – GCN output dim (target-node embedding size)
    ts_input_size    : int   – per-step temporal feature width (e.g. 1 + n_exog)
    lookback_window  : int   – lookback length L (kept for bookkeeping; the
                              shared per-step MLP is size-invariant in L)
    hidden_sizes     : tuple – widths of the MLP hidden layers
    horizon          : int   – model horizon (1 → wrap recursively at inference)
    dropout          : float – dropout for GCN + MLP
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 32,
        d_g: int = 16,
        ts_input_size: int = 1,
        lookback_window: int = 30,
        hidden_sizes: tuple = (64, 32),
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── GCN encoder ───────────────────────────────────────────────────
        self.conv1 = GCNConv(in_channels, gcn_hidden, add_self_loops=True)
        self.conv2 = GCNConv(gcn_hidden,  d_g,        add_self_loops=True)
        self.activation = nn.ReLU()
        self.gnn_drop   = nn.Dropout(p=dropout)
        self.z_norm     = nn.LayerNorm(d_g)

        # ── Time-distributed MLP (shared weights across timesteps) ──────────
        # Applied independently to each of the L steps: (B*L, ts+d_g) → (B*L, H)
        # then pooled via cat[last, mean] → (B, 2H) → linear head.
        per_step_in = ts_input_size + d_g
        layers = []
        prev = per_step_in
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), nn.ReLU(), nn.Dropout(p=dropout)]
            prev = hs
        self.timestep_net = nn.Sequential(*layers)
        self.head = nn.Linear(prev * 2, horizon)  # cat[last, mean] → output

        # bookkeeping
        self.d_g             = d_g
        self.lookback_window = lookback_window
        self.ts_input_size   = ts_input_size
        self.horizon       = horizon

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
        ts_seq              : FloatTensor (B, L, ts_input_size)
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

            z_seq = z_flat.view(B, L, self.d_g)

        # Per-step concat with the temporal/exogenous features.
        x = torch.cat([ts_seq, z_seq], dim=-1)                     # (B, L, ts_in+d_g)

        # Time-distributed forward: apply shared MLP to every timestep.
        B2, L2, C = x.shape
        h = self.timestep_net(x.reshape(B2 * L2, C))              # (B*L, H)
        h = h.reshape(B2, L2, -1)                                  # (B, L, H)
        last = h[:, -1, :]                                         # (B, H) — most recent
        mean = h.mean(dim=1)                                       # (B, H) — global context
        combined = torch.cat([last, mean], dim=-1)                 # (B, 2H)

        pred = self.head(combined)                                 # (B, horizon)
        return pred.unsqueeze(-1)                                  # (B, horizon, 1)
