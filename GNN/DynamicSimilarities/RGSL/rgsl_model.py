"""
Faithful PyTorch implementation of RGSL — Regularized Graph Structure Learning
with Semantic Knowledge for Multi-variate Time-Series Forecasting:

    Hongyuan Yu, Ting Li, Weichen Yu, Jianguo Li, Yan Huang, Liang Wang,
    Alex Liu (2022), IJCAI 2022.  (dissertation ref [78])
    https://github.com/alipay/RGSL

Three modules, following the paper's §2 and Eqs (2)-(8):

  RGG  — Regularized Graph Generation (§2.2, Eqs 2-3).  An IMPLICIT graph is
         learned from a node-embedding dictionary E:
             θ = E Eᵀ                                              (Eq 2)
             A^(l) = σ( (logit(θ) + (g1 − g2)) / s ),  g ~ Gumbel  (Eq 3)
         i.e. a Gumbel-Sigmoid (binary-concrete) relaxation that SPARSIFIES the
         dense similarity matrix and stays differentiable.

  LM³  — Laplacian Matrix Mix-up (§2.3, Eqs 4-6).  The EXPLICIT prior graph
         A^(0) (here the dissertation's Spearman/CID graph) and the implicit
         A^(l) are each convolved with their OWN Chebyshev/GCN weights under
         SYMMETRIC normalisation, then fused in feature space by a gate
         (the paper's self-attention "MixUp Gate"):
             Ã^(0)(X) = (I + D^{-½} A^(0) D^{-½}) X W^(0) + b^(0)   (Eq 4)
             Ã^(l)(X) = (I + D^{-½} A^(l) D^{-½}) X W^(l) + b^(l)   (Eq 5)
             M(X)     = gate ⊙ Ã^(0)(X) + (1−gate) ⊙ Ã^(l)(X)      (Eq 6)

  STRGC — Spatial-Temporal Recurrent Graph Convolution (§2.4, Eq 7).  A GRU
         whose dense maps are replaced by the LM³ graph op M(·).

The Chebyshev convs reuse AGCRN's Node-Adaptive Parameter Learning (the paper
builds RGSL on the AGCRN backbone), so each W is generated from E.

Input  X : (B, T_in, N, C)
Output   : (B, horizon, N, out_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def _sym_norm(A):
    """Symmetric normalisation with self-loops: D^{-1/2}(A+I)D^{-1/2}  (Eq 4/5)."""
    N = A.shape[0]
    A = A + torch.eye(N, device=A.device, dtype=A.dtype)
    d = A.sum(1)
    dinv = d.pow(-0.5)
    dinv = torch.nan_to_num(dinv, nan=0.0, posinf=0.0, neginf=0.0)
    return dinv.view(-1, 1) * A * dinv.view(1, -1)


# ════════════════════════════════════════════════════════════════════════════
# Node-adaptive Chebyshev graph conv (AGCRN NAPL), adjacency fed in
# ════════════════════════════════════════════════════════════════════════════
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.constant_(self.bias_pool, 0.0)

    def forward(self, x, node_embeddings, support):
        """x:(B,N,in)  node_embeddings:(N,d)  support:(N,N) normalised → (B,N,out)."""
        N = node_embeddings.shape[0]
        sup = [torch.eye(N, device=x.device, dtype=x.dtype), support]
        for _ in range(2, self.cheb_k):
            sup.append(2 * support @ sup[-1] - sup[-2])
        sup = torch.stack(sup, dim=0)                          # (cheb_k, N, N)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_g = torch.einsum('knm,bmc->bknc', sup, x)            # (B,cheb_k,N,in)
        x_g = x_g.permute(0, 2, 1, 3)                          # (B,N,cheb_k,in)
        return torch.einsum('bnki,nkio->bno', x_g, weights) + bias


# ════════════════════════════════════════════════════════════════════════════
# LM³: dual-graph conv (explicit + implicit) fused by a MixUp gate (Eqs 4-6)
# ════════════════════════════════════════════════════════════════════════════
class LM3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.g_exp = AVWGCN(dim_in, dim_out, cheb_k, embed_dim)   # explicit A^(0), W^(0)
        self.g_imp = AVWGCN(dim_in, dim_out, cheb_k, embed_dim)   # implicit A^(l), W^(l)
        # MixUp gate (self-attention surrogate): input-dependent convex weight
        self.gate = nn.Linear(2 * dim_out, dim_out)
        self.last_gate = 0.5

    def forward(self, x, node_embeddings, sup_exp, sup_imp):
        h0 = self.g_exp(x, node_embeddings, sup_exp)          # Ã^(0)(X)
        hl = self.g_imp(x, node_embeddings, sup_imp)          # Ã^(l)(X)
        g = torch.sigmoid(self.gate(torch.cat([h0, hl], dim=-1)))
        self.last_gate = float(g.mean().detach())
        return g * h0 + (1 - g) * hl                          # M(X), Eq 6


# ════════════════════════════════════════════════════════════════════════════
# RGG: implicit sparse graph via Gumbel-Sigmoid (Eqs 2-3)
# ════════════════════════════════════════════════════════════════════════════
class RegularizedGraphGenerator(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.register_buffer("_eye", torch.eye(num_nodes))

    def forward(self, tau: float = 0.5):
        theta = self.emb @ self.emb.t()                       # (N,N) = logit(θ) (Eq 2)
        if self.training:
            g1 = _sample_gumbel(theta.shape, theta.device)
            g2 = _sample_gumbel(theta.shape, theta.device)
            A = torch.sigmoid((theta + g1 - g2) / tau)        # Gumbel-Sigmoid (Eq 3)
        else:
            A = torch.sigmoid(theta / tau)                    # deterministic at eval
        return A * (1.0 - self._eye)

    def edge_prob(self):
        return torch.sigmoid(self.emb @ self.emb.t()) * (1.0 - self._eye)


# ════════════════════════════════════════════════════════════════════════════
# STRGC recurrent cell (Eq 7) — GRU with LM³ graph ops
# ════════════════════════════════════════════════════════════════════════════
class RGSLCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = LM3Conv(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = LM3Conv(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, E, sup_exp, sup_imp):
        state = state.to(x.device)
        z_r = torch.sigmoid(self.gate(torch.cat((x, state), -1), E, sup_exp, sup_imp))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        hc = torch.tanh(self.update(torch.cat((x, r * state), -1), E, sup_exp, sup_imp))
        return z * state + (1 - z) * hc

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class STRGCEncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers):
        super().__init__()
        self.node_num = node_num
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [RGSLCell(node_num, dim_in, dim_out, cheb_k, embed_dim)]
            + [RGSLCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
               for _ in range(num_layers - 1)]
        )

    def forward(self, x, init_state, E, sup_exp, sup_imp):
        T = x.shape[1]
        current = x
        out_states = []
        for l, cell in enumerate(self.cells):
            state = init_state[l]
            inner = []
            for t in range(T):
                state = cell(current[:, t, :, :], state, E, sup_exp, sup_imp)
                inner.append(state)
            out_states.append(state)
            current = torch.stack(inner, dim=1)
        return current, torch.stack(out_states, dim=0)

    def init_hidden(self, batch_size):
        return torch.stack([c.init_hidden_state(batch_size) for c in self.cells], dim=0)


# ════════════════════════════════════════════════════════════════════════════
# Full RGSL forecaster
# ════════════════════════════════════════════════════════════════════════════
class RGSL(nn.Module):
    """
    Parameters
    ----------
    num_nodes  : N — products (61)
    prior_adj  : (N, N) EXPLICIT prior graph A^(0) (Spearman/CID), >=0 weights
    in_dim     : C input channels (1 demand [+ exog])
    out_dim    : output channels per node (1)
    horizon    : H forecast horizon
    rnn_units  : STRGC hidden width (paper backbone: 64)
    embed_dim  : NAPL node-embedding dim (10)
    node_dim   : RGG graph-embedding dim (10)
    num_layers : stacked cells (2)
    cheb_k     : Chebyshev order (2)
    tau        : Gumbel-Sigmoid temperature s (Eq 3)
    """

    def __init__(
        self,
        num_nodes: int,
        prior_adj: torch.Tensor,
        in_dim: int = 1,
        out_dim: int = 1,
        horizon: int = 1,
        rnn_units: int = 64,
        embed_dim: int = 10,
        node_dim: int = 10,
        num_layers: int = 2,
        cheb_k: int = 2,
        tau: float = 0.5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.out_dim = out_dim
        self.rnn_units = rnn_units
        self.tau = tau

        # explicit prior A^(0), symmetric-normalised ONCE → fixed buffer (Eq 4)
        self.register_buffer("sup_exp", _sym_norm(prior_adj.float()))

        self.napl_embed = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.gen = RegularizedGraphGenerator(num_nodes, node_dim)
        self.encoder = STRGCEncoder(num_nodes, in_dim, rnn_units, cheb_k, embed_dim, num_layers)
        self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, rnn_units), bias=True)

    def forward(self, x):
        """x:(B,T,N,C) → ŷ:(B,horizon,N,out_dim)."""
        B = x.shape[0]
        A_imp = self.gen(self.tau)                            # implicit graph (Eq 3)
        sup_imp = _sym_norm(A_imp)                            # (Eq 5)
        init_state = self.encoder.init_hidden(B).to(x.device)
        _, final = self.encoder(x, init_state, self.napl_embed, self.sup_exp, sup_imp)
        h = final[-1:].permute(1, 0, 2, 3)                   # (B,1,N,units)
        out = self.end_conv(h)                               # (B,H*out,N,1)
        out = out.squeeze(-1).reshape(B, self.horizon, self.out_dim, self.num_nodes)
        return out.permute(0, 1, 3, 2)                       # (B,H,N,out)

    # ── regularisation + graph-quality hooks ─────────────────────────────────
    def graph_reg_loss(self):
        """Sparsity regulariser on the implicit graph (mean edge probability)."""
        return self.gen.edge_prob().mean()

    @torch.no_grad()
    def learned_adjacency(self):
        """The implicit (Gumbel-Sigmoid, eval-mode) learned graph A^(l) (N, N)."""
        was = self.training
        self.eval()
        A = self.gen(self.tau)
        self.train(was)
        return A

    @torch.no_grad()
    def explicit_adjacency(self):
        """The symmetric-normalised explicit prior A^(0) actually used (N, N)."""
        return self.sup_exp.clone()

    def prior_weight(self):
        """Mean LM³ MixUp gate across convs: ~1 ⇒ leans on the prior, ~0 ⇒ learned."""
        gates = [m.last_gate for m in self.modules() if isinstance(m, LM3Conv)]
        return sum(gates) / len(gates) if gates else float("nan")
