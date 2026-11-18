"""
Graph-encoder + DLinear / PatchTST forecasters.

Same per-step contract as the LSTM / MLP hybrids in ``gcn_models.py``:

    forward(pyg_batch, target_node_indices, ts_seq) -> (B, horizon, 1)

so they use the existing dataset / collate / training / recursive inference
unchanged.  The graph encoders are layer-for-layer copies of the ones in
``gcn_models.py``; the per-day embedding z is passed to the temporal model as
covariate channels:

    temporal input = [ y_t ‖ z_t ]      (exog columns of ts_seq are ignored,
                                         as in the univariate baselines)

DLinear / PatchTST zero-initialise their covariate path, so each hybrid starts
out identical to its univariate baseline (DLinearBaseline / PatchTSTBaseline)
and only departs from it if z lowers the training loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from DLinearBaseline.dlinear import DLinear
from PatchTSTBaseline.patchtst import PatchTST
from gcn_models import EdgeWeightedGATConv

ENCODERS = ('gcn', 'gat', 'graph2vec')


class _GraphEncoderHybrid(nn.Module):
    """Builds the encoder named ``encoder`` and turns a batch of B*L per-day
    graphs into z of shape (B, L, d_g).  Attribute names (conv1/conv2/proj,
    z_norm) match ``gcn_models`` so ``train_methods`` diagnostics apply."""

    def _build_encoder(self, encoder, in_channels, hidden, d_g, dropout, heads):
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; expected one of {ENCODERS}")
        self.encoder_kind = encoder
        if encoder == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden, add_self_loops=True)
            self.conv2 = GCNConv(hidden, d_g, add_self_loops=True)
            self.activation = nn.ReLU()
        elif encoder == 'gat':
            self.conv1 = EdgeWeightedGATConv(
                in_channels, hidden, heads=heads, concat=True, dropout=dropout,
                edge_dim=1, add_self_loops=True, fill_value=1.0,
            )
            self.conv2 = EdgeWeightedGATConv(
                hidden * heads, d_g, heads=1, concat=False, dropout=dropout,
                edge_dim=1, add_self_loops=True, fill_value=1.0,
            )
            self.activation = nn.ELU()
        else:
            self.proj = nn.Linear(in_channels, d_g)
            self.activation = nn.ReLU()
        self.gnn_drop = nn.Dropout(p=dropout)
        self.z_norm = nn.LayerNorm(d_g)
        self.d_g = d_g
        self.ablate_z = False

    def _encode(self, pyg_batch, target_node_indices, B, L, like):
        if self.ablate_z:
            return torch.zeros(B, L, self.d_g, device=like.device, dtype=like.dtype)
        if self.encoder_kind == 'graph2vec':
            h = self.proj(pyg_batch.x[target_node_indices])
            z = self.z_norm(self.gnn_drop(self.activation(h)))
        else:
            ea = (pyg_batch.edge_attr
                  if pyg_batch.edge_attr is not None and pyg_batch.edge_attr.numel() > 0
                  else None)
            if self.encoder_kind == 'gcn':
                kw = {'edge_weight': None if ea is None else ea.squeeze(-1)}
            else:
                kw = {'edge_attr': ea}
            h = self.conv1(pyg_batch.x, pyg_batch.edge_index, **kw)
            h = self.gnn_drop(self.activation(h))
            h = self.conv2(h, pyg_batch.edge_index, **kw)
            z = self.z_norm(h[target_node_indices])
        if z.shape[0] != B * L:
            raise RuntimeError(
                f"Expected {B*L} target indices for per-step model, got {z.shape[0]}. "
                "Make sure you are using the per-step collate.")
        return z.view(B, L, self.d_g)

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        B, L, _ = ts_seq.shape
        z = self._encode(pyg_batch, target_node_indices, B, L, ts_seq)
        return self.temporal(torch.cat([ts_seq[..., :1], z], dim=-1))   # (B, H, 1)


class GraphDLinearForecaster(_GraphEncoderHybrid):
    """GCN / GAT / Graph2Vec encoder → z → DLinear with z as covariates."""

    def __init__(self, encoder, in_channels, lookback, hidden=32, d_g=16,
                 dropout=0.2, attention_heads=4, kernel_size=7, horizon=1):
        super().__init__()
        self._build_encoder(encoder, in_channels, hidden, d_g, dropout, attention_heads)
        self.temporal = DLinear(lookback=lookback, horizon=horizon,
                                kernel_size=kernel_size, n_covariates=d_g)
        self.horizon = horizon


class GraphPatchTSTForecaster(_GraphEncoderHybrid):
    """GCN / GAT / Graph2Vec encoder → z → PatchTST with z as covariates."""

    def __init__(self, encoder, in_channels, lookback, hidden=32, d_g=16,
                 dropout=0.2, attention_heads=4, horizon=1):
        super().__init__()
        self._build_encoder(encoder, in_channels, hidden, d_g, dropout, attention_heads)
        self.temporal = PatchTST(lookback=lookback, horizon=horizon,
                                 dropout=dropout, n_covariates=d_g)
        self.horizon = horizon
