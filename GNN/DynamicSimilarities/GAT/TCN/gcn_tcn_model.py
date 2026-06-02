"""
Joint GCN + TCN forecaster — trainable replacement for the offline
Graph2Vec → LSTM/MLP pipeline, with a Temporal Convolutional Network
(causal dilated 1-D convs with residual connections) as the temporal head.

For each lookback day we build a small ego-graph (target item + similar
neighbours).  A 2-layer GCN encodes that graph into a vector ``z`` for the
target node.  The L per-step embeddings are concatenated with the temporal
features along the feature axis to form the (B, L, 1 + n_exog + d_g)
sequence consumed by the TCN.  The TCN's last-step activation is mapped
to the forecast horizon by a linear head.

    per-step features = [ y_t  ||  exog_t  ||  z_t ]   shape (B, L, 1 + n_exog + d_g)
    TCN over time     -> (B, C_last, L)
    last-step pool    -> (B, C_last)
    head              -> (B, H)

The whole stack (GCN + TCN + head) is optimised jointly on the forecasting
loss — no offline embeddings, no Doc2Vec.

The forward signature ``(pyg_batch, target_node_indices, ts_seq) -> (B, H, 1)``
matches ``SimpleGCNLSTMForecaster`` so callers can reuse the
GCNTimeSeriesDataset + collate_pyg_ts + recursive_forecast pipeline.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv


# ──────────────────────────────────────────────────────────────────────────
# Building blocks (classic Bai-2018 TCN)
# ──────────────────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    """Trim the trailing ``chomp_size`` time steps to keep convolutions causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual block with two dilated causal 1-D convolutions.
    Each conv is wrapped in weight_norm; the residual path uses a 1x1 conv
    when the input/output channel counts differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # left-pad amount for causality

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalConvNet(nn.Module):
    """
    Stacked TCN: ``len(channels)`` residual blocks with dilations
    1, 2, 4, ..., 2**(N-1).  Receptive field grows exponentially with depth.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for i, c in enumerate(channels):
            blocks.append(TemporalBlock(
                in_channels=prev, out_channels=c,
                kernel_size=kernel_size, dilation=2 ** i,
                dropout=dropout,
            ))
            prev = c
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (B, C_in, L) -> (B, C_last, L)
        return self.network(x)


# ──────────────────────────────────────────────────────────────────────────
# Joint GCN + TCN forecaster
# ──────────────────────────────────────────────────────────────────────────
class SimpleGCNTCNForecaster(nn.Module):
    """
    GCN encoder (per-window ego-graph) → broadcast → concat with ts_seq
    → TCN over time → linear head.

    Parameters
    ----------
    in_channels      : int        – node-feature dimension fed to the GCN
    gcn_hidden       : int        – hidden width of GCNConv₁
    d_g              : int        – GCN output dim (target-node embedding size)
    ts_input_size    : int        – per-step temporal feature width (e.g. 1 + n_exog)
    tcn_channels     : tuple[int] – output channels of each TCN block
    tcn_kernel_size  : int        – kernel size of each TCN conv
    horizon          : int        – model horizon (1 → wrap recursively at inference)
    dropout          : float      – dropout for GCN + TCN
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 32,
        d_g: int = 16,
        ts_input_size: int = 1,
        tcn_channels: Sequence[int] = (32, 32, 32),
        tcn_kernel_size: int = 3,
        horizon: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── GCN encoder ───────────────────────────────────────────────────
        self.conv1 = GCNConv(in_channels, gcn_hidden, add_self_loops=True)
        self.conv2 = GCNConv(gcn_hidden,  d_g,        add_self_loops=True)
        self.activation = nn.ReLU()
        self.gnn_drop   = nn.Dropout(p=dropout)
        self.z_norm     = nn.LayerNorm(d_g)

        # ── TCN head over augmented sequence ──────────────────────────────
        self.tcn = TemporalConvNet(
            in_channels=ts_input_size + d_g,
            channels=tuple(tcn_channels),
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        self.head_drop = nn.Dropout(p=dropout)
        self.head      = nn.Linear(tcn_channels[-1], horizon)

        # bookkeeping
        self.d_g           = d_g
        self.ts_input_size = ts_input_size
        self.horizon       = horizon

        # diagnostic switch — when True the GCN embedding is zeroed out
        # before being concatenated with the temporal features (ablation).
        self.ablate_z = False

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        """
        PER-STEP variant: one ego-graph per lookback day.

        pyg_batch           : torch_geometric.data.Batch of B*L ego-graphs,
                              row-major (sample0_step0, ..., sample0_step{L-1},
                              sample1_step0, ...).
        target_node_indices : LongTensor (B*L,)  — usually ``pyg_batch.ptr[:-1]``.
        ts_seq              : FloatTensor (B, L, ts_input_size)
                              per-step [target ‖ exog_t]

        Returns
        -------
        FloatTensor (B, horizon, 1)
        """
        B, L, _ = ts_seq.shape

        if self.ablate_z:
            # Skip GCN entirely — z is zeroed, no point paying the cost.
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
                    f"Expected {B*L} target indices for per-step model, "
                    f"got {z_flat.shape[0]}.  Make sure you are using the "
                    "per-step collate that returns "
                    "(pyg_batch, ts_batch, y_batch, target_idx, L)."
                )

            # (B*L, d_g) -> (B, L, d_g)  in the same row-major order produced by collate
            z_seq = z_flat.view(B, L, self.d_g)

        # Per-step concat with temporal/exogenous features
        x = torch.cat([ts_seq, z_seq], dim=-1)            # (B, L, ts_in + d_g)

        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2).contiguous()                 # (B, ts_in+d_g, L)
        y = self.tcn(x)                                    # (B, C_last, L)
        last = self.head_drop(y[:, :, -1])                 # (B, C_last)
        pred = self.head(last)                             # (B, horizon)
        return pred.unsqueeze(-1)                          # (B, horizon, 1)
