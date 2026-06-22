"""
Faithful PyTorch implementation of AGCRN — Adaptive Graph Convolutional
Recurrent Network, from:

    Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang (2020),
    "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting",
    NeurIPS 2020.  arXiv:2007.02842

This is a GLOBAL multivariate forecaster: one model over all N products jointly,
producing a per-product forecast for each horizon step.  It learns the graph
end-to-end from data — no Spearman/Kendall/CID prior is supplied — which makes
it the canonical "learned topology" competitor to the dissertation's Path-A
(hand-crafted similarity) graphs.

Two mechanisms (paper §3):

  * DAGG  — Data Adaptive Graph Generation.  A single shared adjacency is built
    from a learnable node-embedding dictionary E (N, d):
        A = softmax(relu(E Eᵀ))
    No predefined adjacency is needed.

  * NAPL  — Node Adaptive Parameter Learning.  Instead of one shared GCN weight
    matrix, each node's weights are factorised from its embedding:
        W_n = E_n · W_pool   (W_pool shared, E_n node-specific)
    so every product gets personalised spatial-temporal parameters while the
    parameter count stays O(d) rather than O(N).

The GCN (``AVWGCN``) replaces the dense layers inside a GRU cell (``AGCRNCell``),
stacked into an encoder (``AVWDCRNN``); a 1×1 conv head maps the final hidden
state to the multi-horizon forecast.

Input  X : (B, T_in, N, C)   — C = 1 (demand) [+ exogenous channels]
Output   : (B, horizon, N, out_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Adaptive Vertex-Wise Graph Convolution (DAGG + NAPL)
# ════════════════════════════════════════════════════════════════════════════
class AVWGCN(nn.Module):
    """Chebyshev-style graph conv over the data-adaptive adjacency, with
    node-adaptive weights.

    cheb_k : order of the support expansion (paper uses 2 → support set [I, A]).
    """

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = cheb_k
        # NAPL pools: node weights/bias are generated from the embedding E
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.constant_(self.bias_pool, 0.0)

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        x               : (B, N, dim_in)
        node_embeddings : (N, embed_dim)
        Returns         : (B, N, dim_out)
        """
        N = node_embeddings.shape[0]

        # DAGG: A = softmax(relu(E Eᵀ))  — row-normalised, (N, N)
        supports = F.softmax(F.relu(node_embeddings @ node_embeddings.t()), dim=1)

        # Chebyshev support set: [I, A, A², ...] up to cheb_k terms
        support_set = [torch.eye(N, device=x.device), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(2 * supports @ support_set[-1] - support_set[-2])
        supports = torch.stack(support_set, dim=0)                  # (cheb_k, N, N)

        # NAPL: per-node weights & bias from the embedding
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool                     # (N, dim_out)

        # graph convolution
        x_g = torch.einsum('knm,bmc->bknc', supports, x)            # (B, cheb_k, N, in)
        x_g = x_g.permute(0, 2, 1, 3)                               # (B, N, cheb_k, in)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv                                              # (B, N, dim_out)


# ════════════════════════════════════════════════════════════════════════════
# AGCRN recurrent cell — a GRU whose dense layers are AVWGCN graph convs
# ════════════════════════════════════════════════════════════════════════════
class AGCRNCell(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int,
                 cheb_k: int, embed_dim: int):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        # gate computes update (z) and reset (r) together → 2*dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, N, dim_in)   input at this timestep
        state : (B, N, dim_out)  previous hidden state
        Returns new hidden state (B, N, dim_out).
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return z * state + (1 - z) * hc

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


# ════════════════════════════════════════════════════════════════════════════
# Stacked encoder
# ════════════════════════════════════════════════════════════════════════════
class AVWDCRNN(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int,
                 cheb_k: int, embed_dim: int, num_layers: int):
        super().__init__()
        assert num_layers >= 1
        self.node_num = node_num
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim)]
            + [AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
               for _ in range(num_layers - 1)]
        )

    def forward(self, x: torch.Tensor, init_state: torch.Tensor,
                node_embeddings: torch.Tensor):
        """
        x          : (B, T, N, dim_in)
        init_state : (num_layers, B, N, dim_out)
        Returns (output_seq (B, T, N, dim_out), final_states (num_layers, B, N, dim_out)).
        """
        B, T, N, _ = x.shape
        current = x
        out_states = []
        for l, cell in enumerate(self.cells):
            state = init_state[l]
            inner = []
            for t in range(T):
                state = cell(current[:, t, :, :], state, node_embeddings)
                inner.append(state)
            out_states.append(state)
            current = torch.stack(inner, dim=1)                     # (B, T, N, dim_out)
        return current, torch.stack(out_states, dim=0)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.stack(
            [c.init_hidden_state(batch_size) for c in self.cells], dim=0
        )


# ════════════════════════════════════════════════════════════════════════════
# Full AGCRN forecaster
# ════════════════════════════════════════════════════════════════════════════
class AGCRN(nn.Module):
    """
    Parameters
    ----------
    num_nodes  : N — number of products (61)
    in_dim     : C — input channels (1 demand [+ exog])
    out_dim    : output channels per node (1 = demand)
    horizon    : H — forecast horizon
    rnn_units  : hidden width of the AGCRN cells (paper: 64)
    embed_dim  : node-embedding dim d for DAGG/NAPL (paper: 10)
    num_layers : stacked AGCRN cells (paper: 2)
    cheb_k     : support order (paper: 2)
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        out_dim: int = 1,
        horizon: int = 1,
        rnn_units: int = 64,
        embed_dim: int = 10,
        num_layers: int = 2,
        cheb_k: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.rnn_units = rnn_units

        # learnable node-embedding dictionary E — drives both DAGG and NAPL
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))

        self.encoder = AVWDCRNN(
            num_nodes, in_dim, rnn_units, cheb_k, embed_dim, num_layers
        )
        # 1×1 conv head over the final hidden state → all horizons at once
        self.end_conv = nn.Conv2d(
            1, horizon * out_dim, kernel_size=(1, rnn_units), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N, C)
        Returns ŷ : (B, horizon, N, out_dim)
        """
        B = x.shape[0]
        init_state = self.encoder.init_hidden(B).to(x.device)
        _, final_states = self.encoder(x, init_state, self.node_embeddings)
        h = final_states[-1:]                                       # (1, B, N, units)
        h = h.permute(1, 0, 2, 3)                                   # (B, 1, N, units)

        out = self.end_conv(h)                                      # (B, H*out, N, 1)
        out = out.squeeze(-1).reshape(B, self.horizon, self.out_dim, self.num_nodes)
        out = out.permute(0, 1, 3, 2)                               # (B, H, N, out)
        return out

    @torch.no_grad()
    def learned_adjacency(self) -> torch.Tensor:
        """Return the current data-adaptive adjacency A = softmax(relu(E Eᵀ)).

        Exposed for the dissertation's graph-quality analysis (§3.6): lets you
        inspect / compare the learned topology against the Spearman/CID graphs.
        """
        E = self.node_embeddings
        return F.softmax(F.relu(E @ E.t()), dim=1)
