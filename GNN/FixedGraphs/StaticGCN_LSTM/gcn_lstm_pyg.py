"""
gcn_lstm_pyg.py — Static-graph GCN + LSTM forecaster (end-to-end).

Architecture
------------
1. A SINGLE global graph G is computed once from training-period Spearman
   (or Pearson) similarities between all items.  Every item is a node;
   edges exist where sim ≥ threshold with weight = sim.

2. A GCNEncoder runs on G to produce node embeddings Z ∈ R^(N × gcn_out).
   This single forward pass covers ALL items simultaneously.

3. For a batch of B samples targeting items [i₁, …, i_B]:
       z_b = Z[i_b]   (the target node's embedding)

4. z_b is projected to initialise the LSTM hidden/cell states (h₀, c₀).

5. The LSTM then processes the explicit lookback sequence
       ts_seq[:, t, :] = [value_t, cal_{t+1}]
   and produces the horizon forecast.

Key difference from the per-sample ego-graph approach
------------------------------------------------------
* Old: one small ego-graph built per sample (sliding-window node features).
* New: one global graph processed ONCE per forward call; all N items share
  the same GCN pass and each sample looks up its target embedding.

GCN parameters + LSTM parameters are optimised jointly → end-to-end.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# GCN encoder
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    """
    N-layer GCNConv that embeds every node in the global graph.

    Risk-2 mitigation: `num_layers` is configurable.  With a 1-hop ego-graph
    a single layer is sufficient; depth > 1 is only useful when 2-hop nodes
    are included (include_2hop=True).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"
        if num_layers == 1:
            dims = [(in_channels, out_channels)]
        else:
            dims = ([(in_channels, hidden_channels)]
                    + [(hidden_channels, hidden_channels)] * (num_layers - 2)
                    + [(hidden_channels, out_channels)])
        self.convs = nn.ModuleList(
            [GCNConv(d_in, d_out, add_self_loops=True) for d_in, d_out in dims]
        )
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:   # activation + dropout between layers only
                x = self.act(x)
                x = self.drop(x)
        return x   # (N_items, out_channels)


# ---------------------------------------------------------------------------
# Combined GCN + LSTM forecaster
# ---------------------------------------------------------------------------

class StaticGCNLSTMForecaster(nn.Module):
    """
    Static-graph GCN + LSTM end-to-end forecaster.

    Parameters
    ----------
    in_channels        : node feature dimension (8 static stats per item)
    gcn_hidden         : GCN intermediate-layer width
    gcn_out            : GCN output width
    lstm_input_size    : 1 + cal_dim  (value + calendar features at each step)
    lstm_hidden        : LSTM hidden size
    lstm_layers        : LSTM depth
    horizon            : number of forecast steps
    dropout            : dropout rate
    gcn_layers         : number of GCN layers (Risk-2 mitigation: use 1 for
                         pure 1-hop ego-graphs, 2+ when include_2hop=True)
    graph_conditioning : how the GCN embedding conditions the LSTM
                         'init'   – project z into h₀/c₀  (default)
                         'concat' – concatenate z to every LSTM input step
                                    (Risk-5 mitigation: prevents dilution)

    Forward signature
    -----------------
    forward(global_x, edge_index, edge_weight, target_node_indices, ts_seq)

        global_x            : (N, in_channels)      – all-item node features
        edge_index          : (2, E)                – fixed graph topology
        edge_weight         : (E,) or None          – edge similarities
        target_node_indices : (B,) LongTensor       – node index per sample
        ts_seq              : (B, lookback, 1+cal_dim)

    Returns : (B, horizon, 1)
    """

    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int,
        gcn_out: int,
        lstm_input_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        horizon: int,
        dropout: float = 0.2,
        gcn_layers: int = 2,
        graph_conditioning: str = 'init',
    ):
        super().__init__()
        if graph_conditioning not in ('init', 'concat'):
            raise ValueError(f"graph_conditioning must be 'init' or 'concat', got {graph_conditioning!r}")
        self.graph_conditioning = graph_conditioning
        self.gcn = GCNEncoder(in_channels, gcn_hidden, gcn_out,
                              num_layers=gcn_layers, dropout=dropout)

        if graph_conditioning == 'init':
            # Project gcn_out → lstm_hidden for h₀ and c₀ independently
            self.h0_proj = nn.Linear(gcn_out, lstm_hidden * lstm_layers)
            self.c0_proj = nn.Linear(gcn_out, lstm_hidden * lstm_layers)
            effective_lstm_input = lstm_input_size
        else:  # 'concat'
            # z is appended to every time step; no h₀/c₀ projection
            effective_lstm_input = lstm_input_size + gcn_out

        self.lstm = nn.LSTM(
            effective_lstm_input,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.drop    = nn.Dropout(p=dropout)
        self.fc_out  = nn.Linear(lstm_hidden, horizon)

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.horizon     = horizon
        self.gcn_out     = gcn_out

    def forward(
        self,
        global_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        target_node_indices: torch.Tensor,
        ts_seq: torch.Tensor,
    ) -> torch.Tensor:
        # 1. GCN on the ego-graph → embeddings for all nodes
        node_emb = self.gcn(global_x, edge_index, edge_weight)   # (N, gcn_out)

        # 2. Extract each sample's target node embedding
        z = node_emb[target_node_indices]                        # (B, gcn_out)
        z = self.drop(z)

        B = z.size(0)

        if self.graph_conditioning == 'init':
            # 3a. Project z into LSTM initial states
            h0 = self.h0_proj(z)   # (B, layers * hidden)
            c0 = self.c0_proj(z)
            h0 = h0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
            c0 = c0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
            # 4a. LSTM over the lookback sequence
            out, _ = self.lstm(ts_seq, (h0, c0))                 # (B, lookback, hidden)
        else:
            # 3b. Concatenate z to every input step (prevents h₀ dilution)
            z_exp = z.unsqueeze(1).expand(-1, ts_seq.size(1), -1)  # (B, lookback, gcn_out)
            ts_aug = torch.cat([ts_seq, z_exp], dim=-1)            # (B, lookback, input+gcn_out)
            # 4b. LSTM with zero initial state
            out, _ = self.lstm(ts_aug)                             # (B, lookback, hidden)

        last = out[:, -1, :]                 # (B, hidden)

        # 5. Output projection
        pred = self.fc_out(last)              # (B, horizon)
        return pred.unsqueeze(-1)             # (B, horizon, 1)
