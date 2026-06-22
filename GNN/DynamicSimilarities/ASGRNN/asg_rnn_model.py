"""
Faithful PyTorch implementation of the Attribute-Space Graph Recurrent
Neural Network (ASG-RNN) from:

    Shaohui Ma (2024), "Retail store-SKU level replenishment planning with
    attribute-space graph recurrent neural networks", Expert Systems with
    Applications 249:123727.  https://doi.org/10.1016/j.eswa.2024.123727

This is a GLOBAL forecaster: one model over all N products jointly, producing
a per-product quantile forecast for each horizon step.  Equation numbers below
refer to the paper.

Architecture (paper Fig. 1):

    A ──▶ AttributeEmbedder ──▶ Ã ──▶ ProductNetworkBuilder ──▶ (edge_index, e)
                                                                      │
    P_fut ───────────────────────────────────▶ GatedGraphConv ◀──────┘
                                                       │  g_{t+h}
    [X,y]_hist ──▶ Seq2Seq encoder ──▶ s_T,c_T ──▶ Seq2Seq decoder ──▶ ŷ^(q)
                                                       ▲
    X_fut ─────────────────────────────────────────────┘

Design notes / deviations, all explicit:
  * Attribute space is F-dimensional (one learned 1-D coordinate per categorical
    attribute via Tanh(linear(one-hot))), matching the text and Figs. 7-8.
  * The kNN graph is rebuilt every forward pass from the *learnable* attribute
    coordinates.  Neighbour SELECTION is non-differentiable (top-K), but the
    edge WEIGHTS (normalised inverse distances, Eq. 6 wording) are differentiable,
    so gradient flows into the attribute projection.
  * Gated Graph Convolution (Eqs. 5-7) follows Li et al. (2016): a per-layer
    message matrix Theta^(l) and a single shared GRUCell across the L
    propagation steps.
  * Quantile (tick) loss lives in ``tick_loss`` below; with Q={0.5} it reduces
    to 0.5*MAE, giving point-forecast comparability with the existing pipeline.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# 3.2.1  Product network builder
# ════════════════════════════════════════════════════════════════════════════
class AttributeEmbedder(nn.Module):
    """One-hot categorical attributes → F-dimensional continuous attribute space.

    Each of the F attributes is projected to a single scalar coordinate via a
    Tanh-activated linear layer (paper: ``b_f = Tanh(w_f' ã_f)``).  Stacking the
    F coordinates yields ``Ã_j = [b^1_j, ..., b^F_j] ∈ R^F``.

    Parameters
    ----------
    level_counts : sequence of int
        ``L_f`` — number of one-hot levels for each of the F attributes.
    """

    def __init__(self, level_counts: Sequence[int]):
        super().__init__()
        self.level_counts = list(level_counts)
        self.F = len(self.level_counts)
        # one Linear(L_f -> 1) per attribute; no bias (pure projection of one-hot)
        self.projs = nn.ModuleList(
            [nn.Linear(L_f, 1, bias=False) for L_f in self.level_counts]
        )

    def forward(self, onehots: Sequence[torch.Tensor]) -> torch.Tensor:
        """onehots : list of F tensors, each (N, L_f).  Returns Ã (N, F)."""
        coords = [torch.tanh(self.projs[f](onehots[f])) for f in range(self.F)]
        return torch.cat(coords, dim=-1)            # (N, F)


class ProductNetworkBuilder(nn.Module):
    """Directed K-nearest-neighbour graph in attribute space (Eq. 4).

    For each target node j, edges arrive from its K nearest neighbours k
    (Euclidean distance in Ã).  Edge weight ``e_kj`` is the normalised inverse
    distance so that ``sum_k e_kj = 1`` over j's incoming neighbours
    (paper §3.2.2).

    Returns a dense weighted adjacency ``W`` of shape (N, N) where ``W[j, k]``
    is the message weight from neighbour k into target j (row-stochastic).
    Using a dense N×N matrix is deliberate: N≈61 makes this trivially cheap and
    lets the gated graph conv be a couple of matmuls instead of scatter ops.
    """

    def __init__(self, k: int, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, attr: torch.Tensor) -> torch.Tensor:
        """attr : (N, F) learnable attribute coordinates.  Returns W (N, N)."""
        N = attr.shape[0]
        k = min(self.k, N - 1)

        # pairwise squared Euclidean distance, (N, N); dist[j, k]
        dist = torch.cdist(attr, attr, p=2)                     # (N, N)
        # never pick self as a neighbour
        eye = torch.eye(N, device=attr.device, dtype=torch.bool)
        dist = dist.masked_fill(eye, float('inf'))

        # K nearest neighbours per target row j
        knn_d, knn_idx = torch.topk(dist, k, dim=1, largest=False)   # (N, k)

        # inverse-distance weights, normalised per target (row) → sum_k e_kj = 1
        inv = 1.0 / (knn_d + self.eps)                          # (N, k)
        inv = inv / inv.sum(dim=1, keepdim=True)

        W = torch.zeros(N, N, device=attr.device, dtype=attr.dtype)
        W.scatter_(1, knn_idx, inv)                             # W[j, neighbour]=e
        return W


# ════════════════════════════════════════════════════════════════════════════
# 3.2.2  Gated Graph Convolution operator (Eqs. 5-7)
# ════════════════════════════════════════════════════════════════════════════
class GatedGraphConv(nn.Module):
    """L-step gated graph convolution over a fixed weighted adjacency.

    g^(0) = [P, 0]                                            (Eq. 5)
    m^(l+1)_j = sum_k e_kj · Θ^(l) g^(l)_k                    (Eq. 6)
    g^(l+1)_j = GRU(m^(l+1)_j, g^(l)_j)                       (Eq. 7)

    Operates on a batch of independent graphs that all share the same topology
    W (one product network), so the message pass is ``W @ (g Θ^lᵀ)``.

    Shapes use a flattened (B*N) node axis so a batch of episodes runs in one go.
    """

    def __init__(self, p_in: int, hg: int, layers: int):
        super().__init__()
        self.hg = hg
        self.layers = layers
        self.p_in = p_in
        # per-layer message transform Θ^(l) ∈ R^{hg×hg}
        self.theta = nn.ModuleList(
            [nn.Linear(hg, hg, bias=False) for _ in range(layers)]
        )
        self.gru = nn.GRUCell(hg, hg)               # shared across propagation steps

    def forward(self, P: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        P : (B, N, p_in) promotional features for this time step.
        W : (N, N)       row-stochastic neighbour weights, W[j,k]=e_kj.
        Returns g : (B, N, hg) aggregated cross-item promotional state.
        """
        B, N, _ = P.shape
        # g^(0) = [P, 0]  → pad promo vector up to hg
        g = F.pad(P, (0, self.hg - self.p_in))                  # (B, N, hg)
        for l in range(self.layers):
            # message: aggregate transformed neighbour states. W[j,k] weights k→j
            msg = torch.einsum('jk,bkh->bjh', W, self.theta[l](g))   # (B, N, hg)
            g = self.gru(
                msg.reshape(B * N, self.hg),
                g.reshape(B * N, self.hg),
            ).reshape(B, N, self.hg)
        return g


# ════════════════════════════════════════════════════════════════════════════
# 3.2.3  Seq2Seq recurrent encoder / decoder (Eqs. 8-12)
# ════════════════════════════════════════════════════════════════════════════
class ASGRNN(nn.Module):
    """Attribute-Space Graph Recurrent Network — full global forecaster.

    Parameters
    ----------
    level_counts : sequence of int   – one-hot width L_f of each static attribute
    n_x          : int               – number of demand-driver features in X
    p_in         : int               – number of promotional features in P
    k            : int               – neighbours for the attribute-space kNN graph
    h_lstm       : int               – LSTM hidden size (paper best: 32)
    hg           : int               – gated-graph-conv GRU width (paper: 16/32)
    ggc_layers   : int               – L, gated-graph-conv steps (paper: 3)
    horizon      : int               – max forecast horizon H
    quantiles    : sequence of float – Q; default (0.5,) ⇒ MAE-comparable point
    """

    def __init__(
        self,
        level_counts: Sequence[int],
        n_x: int,
        p_in: int,
        k: int = 10,
        h_lstm: int = 32,
        hg: int = 32,
        ggc_layers: int = 3,
        horizon: int = 1,
        quantiles: Sequence[float] = (0.5,),
    ):
        super().__init__()
        self.quantiles = list(quantiles)
        self.nQ = len(self.quantiles)
        self.horizon = horizon
        self.n_x = n_x

        # ── attribute-space graph ─────────────────────────────────────────
        self.attr_embed = AttributeEmbedder(level_counts)
        self.graph_build = ProductNetworkBuilder(k)
        self.ggc = GatedGraphConv(p_in, hg, ggc_layers)

        # ── encoder (Eqs. 8-9) ────────────────────────────────────────────
        # input per step is [X, y] → n_x + 1, passed through Tanh linear first
        self.enc_in = nn.Linear(n_x + 1, n_x + 1)
        self.encoder = nn.LSTMCell(n_x + 1, h_lstm)

        # ── decoder (Eqs. 10-12) ──────────────────────────────────────────
        # z̃ = [ Tanh(W_ZD [X_{t+h}, s] + b)  ‖  g_{t+h} ]
        self.dec_in = nn.Linear(n_x + h_lstm, n_x + h_lstm)
        self.decoder = nn.LSTMCell(n_x + h_lstm + hg, h_lstm)
        self.head = nn.Linear(h_lstm, self.nQ)      # ReLU applied in forward (Eq.12)

        self.h_lstm = h_lstm

    # ----------------------------------------------------------------------
    def forward(self, attr_onehots, y_hist, X_hist, X_fut, P_fut):
        """
        attr_onehots : list of F tensors (N, L_f) — static, shared across batch
        y_hist       : (B, N, T_in)            locally-scaled demand history
        X_hist       : (B, N, T_in, n_x)       driver history
        X_fut        : (B, N, H,   n_x)        known future drivers
        P_fut        : (B, N, H,   p_in)       known future promo features

        Returns ŷ : (B, N, H, nQ)   per-product, per-horizon quantile forecasts
                                    (in locally-scaled space; rescale outside).
        """
        B, N, T_in = y_hist.shape
        BN = B * N

        # 1) build the (static) product network once for this batch
        attr = self.attr_embed(attr_onehots)                    # (N, F)
        W = self.graph_build(attr)                              # (N, N)

        # 2) encoder over [X, y] history (Eqs. 8-9), flattened over (B*N)
        z = torch.cat([X_hist, y_hist.unsqueeze(-1)], dim=-1)   # (B,N,T_in,n_x+1)
        z = torch.tanh(self.enc_in(z)).reshape(BN, T_in, self.n_x + 1)
        h = z.new_zeros(BN, self.h_lstm)
        c = z.new_zeros(BN, self.h_lstm)
        for t in range(T_in):
            h, c = self.encoder(z[:, t, :], (h, c))
        s_enc = h                                               # s_T, (BN, h_lstm)

        # 3) decoder, one step per horizon (Eqs. 10-12)
        out = []
        for hstep in range(self.horizon):
            # cross-item promotional state from the graph conv at this step
            g = self.ggc(P_fut[:, :, hstep, :], W)              # (B, N, hg)
            g = g.reshape(BN, -1)

            x_fut = X_fut[:, :, hstep, :].reshape(BN, self.n_x)
            zt = torch.tanh(self.dec_in(torch.cat([x_fut, s_enc], dim=-1)))
            zt = torch.cat([zt, g], dim=-1)                     # (BN, n_x+h+hg)
            h, c = self.decoder(zt, (h, c))
            out.append(F.relu(self.head(h)))                    # (BN, nQ)

        y = torch.stack(out, dim=1).reshape(B, N, self.horizon, self.nQ)
        return y


# ════════════════════════════════════════════════════════════════════════════
# 3.3.3  Tick (pinball) loss  (Eq. 13)
# ════════════════════════════════════════════════════════════════════════════
def tick_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: Sequence[float],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum of pinball losses over the quantile set.

    y_hat  : (B, N, H, nQ)
    y_true : (B, N, H)
    mask   : optional (B, N, H) 0/1 — excludes zero-scale / off-shelf series.

    With quantiles=(0.5,) this equals 0.5 * mean(|y_hat - y_true|) → MAE-proportional.
    """
    y_true = y_true.unsqueeze(-1)                               # (B,N,H,1)
    q = y_hat.new_tensor(quantiles).view(1, 1, 1, -1)           # (1,1,1,nQ)
    diff = y_true - y_hat
    loss = torch.maximum(q * diff, (q - 1.0) * diff)            # pinball, (B,N,H,nQ)
    loss = loss.sum(dim=-1)                                     # sum over quantiles
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()
