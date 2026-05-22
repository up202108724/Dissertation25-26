"""
gat_lstm_pyg.py — Static-graph GAT + LSTM forecaster (end-to-end).

Architecture
------------
1. A SINGLE global graph G is computed once from training-period Spearman
   (or Pearson) similarities between all items.  Every item is a node;
   edges exist where sim ≥ threshold with weight = sim.

2. A GATEncoder runs on G to produce node embeddings Z ∈ R^(N × gat_out).
   Multi-head attention is used; edge similarity weights are passed as
   1-D edge features so each attention head can attend over them.
   This single forward pass covers ALL items simultaneously.

3. For a batch of B samples targeting items [i₁, …, i_B]:
       z_b = Z[i_b]   (the target node's embedding)

4. z_b is projected to initialise the LSTM hidden/cell states (h₀, c₀).

5. The LSTM then processes the explicit lookback sequence
       ts_seq[:, t, :] = [value_t, cal_{t+1}]
   and produces the horizon forecast.

Key difference from GCN variant
--------------------------------
* GCNConv uses symmetric normalisation (fixed aggregation weights).
* GATConv learns per-edge attention coefficients, conditioned on both
  neighbour node features and the edge similarity weight (edge_dim=1).

GAT parameters + LSTM parameters are optimised jointly → end-to-end.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# GAT encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Two-layer GATConv that embeds every node in the global graph.

    Layer 1 uses `heads` attention heads with concatenation → output width
    is hidden_channels * heads.
    Layer 2 uses a single head with averaging → output width is out_channels.
    Edge similarity weights are passed as 1-D edge features (edge_dim=1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=heads, concat=True,
            edge_dim=1, dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels,
            heads=1, concat=False,
            edge_dim=1, dropout=dropout,
            add_self_loops=True,
        )
        self.act  = nn.ELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # GATConv expects edge features as (E, edge_dim)
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return h   # (N_items, out_channels)


# ---------------------------------------------------------------------------
# Combined GAT + LSTM forecaster
# ---------------------------------------------------------------------------

class StaticGATLSTMForecaster(nn.Module):
    """
    Static-graph GAT + LSTM end-to-end forecaster.

    Parameters
    ----------
    in_channels     : node feature dimension (8 static stats per item)
    gat_hidden      : GAT first-layer width per head
    gat_out         : GAT output width  (→ h₀/c₀ projection input)
    lstm_input_size : 1 + cal_dim  (value + calendar features at each step)
    lstm_hidden     : LSTM hidden size
    lstm_layers     : LSTM depth
    horizon         : number of forecast steps
    dropout         : dropout rate (feature dropout + GAT attention dropout)
    gat_heads       : number of attention heads in the first GAT layer

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
        gat_hidden: int,
        gat_out: int,
        lstm_input_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        horizon: int,
        dropout: float = 0.2,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.gat = GATEncoder(in_channels, gat_hidden, gat_out,
                              heads=gat_heads, dropout=dropout)

        # Project gat_out → lstm_hidden for h₀ and c₀ independently
        self.h0_proj = nn.Linear(gat_out, lstm_hidden * lstm_layers)
        self.c0_proj = nn.Linear(gat_out, lstm_hidden * lstm_layers)

        self.lstm = nn.LSTM(
            lstm_input_size,
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

    def forward(
        self,
        global_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        target_node_indices: torch.Tensor,
        ts_seq: torch.Tensor,
    ) -> torch.Tensor:
        # 1. GAT on the full global graph → embeddings for ALL nodes
        node_emb = self.gat(global_x, edge_index, edge_weight)   # (N, gat_out)

        # 2. Extract each sample's target node embedding
        z = node_emb[target_node_indices]                        # (B, gat_out)
        z = self.drop(z)

        # 3. Project to LSTM initial states
        B = z.size(0)
        h0 = self.h0_proj(z)   # (B, layers * hidden)
        c0 = self.c0_proj(z)

        # Reshape to (layers, B, hidden) as required by nn.LSTM
        h0 = h0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()

        # 4. LSTM over the lookback sequence
        out, _ = self.lstm(ts_seq, (h0, c0))   # (B, lookback, hidden)
        last   = out[:, -1, :]                 # (B, hidden)

        # 5. Output projection
        pred = self.fc_out(last)              # (B, horizon)
        return pred.unsqueeze(-1)             # (B, horizon, 1)
