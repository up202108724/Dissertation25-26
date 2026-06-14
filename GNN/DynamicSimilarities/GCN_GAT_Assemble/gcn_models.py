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
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter as _pyg_scatter


class EdgeWeightedGATConv(GATConv):
    """GATConv whose attention is multiplicatively gated by the edge weight.

    Stock ``GATConv`` consumes ``edge_attr`` only as an *additive bias* inside the
    attention logits, so the hand-crafted similarity never *weights* aggregation
    the way GCN's ``edge_weight`` does.  Here we keep the learned attention but
    multiply each softmax-normalised coefficient by the edge weight and
    renormalise per target node — a low-similarity neighbour is down-weighted no
    matter what attention says, recovering the GCN's strong similarity prior
    while retaining learned attention.

    The weight is assumed "bigger == more similar" for every metric, which the
    graph builder guarantees via ``utils.edge_weight_from_metric`` (distance
    metrics are mapped to ``1/(1+d)``).  ``clamp(min=0)`` is a defensive floor
    only — weights are already non-negative.

    Requires ``add_self_loops=True`` + ``fill_value=1.0`` (as configured by the
    callers): every node then has a self-edge with weight 1.0, so the per-target
    normaliser is never zero.
    """

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size):
        # Learned, softmax-normalised attention (still uses edge_attr as a bias).
        alpha = super().edge_update(alpha_j, alpha_i, edge_attr, index, ptr, dim_size)
        if edge_attr is not None:
            w = edge_attr.view(-1).clamp(min=0.0)           # (E,) similarity prior
            alpha = alpha * w.unsqueeze(-1)                  # gate every head
            denom = _pyg_scatter(alpha, index, dim=0, dim_size=dim_size, reduce='sum')
            alpha = alpha / (denom[index] + 1e-16)           # renormalise per target
        return alpha

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
        self.d_g     = d_g
        self.horizon = horizon

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
        last   = self.lstm_drop(out[:, -1, :])                 # (B, hidden)
        pred   = self.head(last)                               # (B, horizon)
        return pred.unsqueeze(-1)                              # (B, horizon, 1)


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
        # Hidden layer concatenates heads, so output dim is gat_hidden * attention_heads.
        # add_self_loops/fill_value made explicit for parity with the GCN path:
        # the target node's self-loop carries edge_attr=1.0 (max similarity) so the
        # attention bias treats "trust my own signal" as strongly as the GCN does.
        self.conv1 = EdgeWeightedGATConv(
            in_channels,
            gat_hidden,
            heads=attention_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,          # Expects 1D edge_attr from the dataset
            add_self_loops=True,
            fill_value=1.0,
        )

        # Output layer averages heads (concat=False), ensuring output is exactly d_g
        self.conv2 = EdgeWeightedGATConv(
            gat_hidden * attention_heads,
            d_g,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
            fill_value=1.0,
        )

        # ELU is the canonical GAT activation (inner attention already uses
        # LeakyReLU); a hard ReLU here would zero every negative coming out of
        # the multi-head concat and waste head capacity.
        self.activation = nn.ELU()
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

class SimpleGATMLPForecaster(nn.Module):
    """
    GAT encoder (per-window ego-graph) → broadcast → concat with ts_seq
    → flatten → MLP → linear head.

    Parameters
    ----------
    in_channels      : int   – node-feature dimension fed to the GAT
    gat_hidden       : int   – hidden width per head in GATConv₁
    gat_heads        : int   – number of attention heads in GATConv₁
                               (output dim of layer 1 = gat_hidden * gat_heads)
    d_g              : int   – GAT output dim (target-node embedding size);
                               GATConv₂ uses 1 head with concat=False
    ts_input_size    : int   – per-step temporal feature width (e.g. 1 + n_exog)
    lookback_window  : int   – lookback length L (kept for bookkeeping; the
                              shared per-step MLP is size-invariant in L)
    hidden_sizes     : tuple – widths of the MLP hidden layers
    horizon          : int   – model horizon (1 → wrap recursively at inference)
    dropout          : float – dropout for GAT attention + MLP
    """

    def __init__(
        self,
        in_channels: int,
        gat_hidden: int = 32,
        gat_heads: int = 4,
        d_g: int = 16,
        ts_input_size: int = 1,
        lookback_window: int = 30,
        hidden_sizes: tuple = (64, 32),
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── GAT encoder ───────────────────────────────────────────────────
        # Layer 1: multi-head attention, concat → output (gat_hidden * gat_heads)
        # fill_value=1.0 → self-loop edge_attr = max similarity (parity with GCN's
        # self-loop weight of 1.0), so the node's own signal is trusted maximally.
        self.conv1 = EdgeWeightedGATConv(
            in_channels, gat_hidden,
            heads=gat_heads, concat=True,
            dropout=dropout, edge_dim=1, add_self_loops=True, fill_value=1.0,
        )
        # Layer 2: single head, averaged → output d_g
        self.conv2 = EdgeWeightedGATConv(
            gat_hidden * gat_heads, d_g,
            heads=1, concat=False,
            dropout=dropout, edge_dim=1, add_self_loops=True, fill_value=1.0,
        )
        self.activation = nn.ELU()
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
        self.horizon         = horizon

        # diagnostic switch — when True the GAT embedding is zeroed out
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
            # Skip GAT entirely — output is zeroed, no point paying the cost.
            z_seq = torch.zeros(B, L, self.d_g, device=ts_seq.device, dtype=ts_seq.dtype)
        else:
            # GAT branch — encodes B*L graphs in a single sparse forward.
            # Edge attributes (similarity weights) are passed as edge_dim=1 features
            # so the attention mechanism can incorporate them.
            ea = (
                pyg_batch.edge_attr
                if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.numel() > 0)
                else None
            )
            h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_attr=ea)
            h = self.activation(h)
            h = self.gnn_drop(h)
            h = self.conv2(h, pyg_batch.edge_index, edge_attr=ea)
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