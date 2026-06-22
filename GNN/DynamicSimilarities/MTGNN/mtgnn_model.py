"""
Faithful PyTorch implementation of MTGNN — Multivariate Time series forecasting
with Graph Neural Networks, from:

    Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi
    Zhang (2020), "Connecting the Dots: Multivariate Time Series Forecasting
    with Graph Neural Networks", KDD 2020.  arXiv:2005.11650

GLOBAL multivariate forecaster: one model over all N products jointly, producing
a per-product forecast. Like AGCRN it learns the graph end-to-end (no
Spearman/CID prior), so it is a second "learned topology" competitor to the
dissertation's Path-A graphs — but with a different inductive bias:

  * Graph learning layer (``graph_constructor``): a *directed, sparse* adjacency
    from two node-embedding dictionaries,
        A = ReLU(tanh(α (M1 M2ᵀ − M2 M1ᵀ)))   then top-k per node
    Unlike AGCRN's dense symmetric softmax(EEᵀ), MTGNN's graph is asymmetric and
    explicitly sparse (subgraph_size neighbours), which is closer in spirit to
    your top-K/kNN sparsification (SoT §3.3.2) and easier to inspect.

  * Temporal module: dilated *inception* 1-D convs (kernels {2,3,6,7}) with a
    gated tanh·sigmoid activation — captures multi-scale temporal patterns
    without recurrence (cf. Graph WaveNet).

  * Spatial module: mix-hop propagation (``mixprop``) over A and Aᵀ with an
    information-retention factor, stacked with residual + skip connections.

Interface matches AGCRN so it drops into the same pipeline:
    Input  X : (B, T_in, N, C)
    Output   : (B, horizon, N, out_dim)
Internally MTGNN works in (B, C, N, T) channels-first layout; the forward
wrapper permutes for you.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── primitive ops ────────────────────────────────────────────────────────────
class nconv(nn.Module):
    """Graph diffusion: x (B,C,N,T) × A (N,N) → (B,C,N,T)."""
    def forward(self, x, A):
        return torch.einsum('ncwl,vw->ncvl', (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, (1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    """Mix-hop propagation layer (paper §3.2): retains a fraction of the root
    state at each hop, concatenates hops, then projects."""
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        return self.mlp(ho)


class dilated_inception(nn.Module):
    """Multi-kernel dilated temporal conv (kernels {2,3,6,7})."""
    def __init__(self, cin, cout, dilation_factor=2):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        assert cout % len(self.kernel_set) == 0, "cout must be divisible by 4"
        cout = cout // len(self.kernel_set)
        self.tconv = nn.ModuleList(
            [nn.Conv2d(cin, cout, (1, k), dilation=(1, dilation_factor))
             for k in self.kernel_set]
        )

    def forward(self, x):
        xs = [conv(x) for conv in self.tconv]
        # align all branches to the shortest (largest-kernel) output length
        for i in range(len(self.kernel_set)):
            xs[i] = xs[i][..., -xs[-1].size(3):]
        return torch.cat(xs, dim=1)


class graph_constructor(nn.Module):
    """Learned directed sparse adjacency (paper §3.1)."""
    def __init__(self, nnodes, k, dim, alpha=3.0):
        super().__init__()
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.k = k
        self.alpha = alpha

    def forward(self, idx):
        v1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        v2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        a = v1 @ v2.t() - v2 @ v1.t()                          # asymmetric → directed
        adj = F.relu(torch.tanh(self.alpha * a))
        # top-k sparsification per row
        mask = torch.zeros(idx.size(0), idx.size(0), device=idx.device)
        s1, t1 = adj.topk(min(self.k, idx.size(0)), dim=1)
        mask.scatter_(1, t1, torch.ones_like(s1))
        return adj * mask


class _LayerNorm(nn.Module):
    """Per-(C,N,T) layer norm with node-indexed affine params (paper impl)."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x, idx):
        if self.elementwise_affine:
            return F.layer_norm(x, tuple(x.shape[1:]),
                                self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        return F.layer_norm(x, tuple(x.shape[1:]), None, None, self.eps)


# ── full model ────────────────────────────────────────────────────────────────
class MTGNN(nn.Module):
    """
    Parameters
    ----------
    num_nodes  : N — products (61)
    in_dim     : C — input channels (1 demand [+ exog])
    out_dim    : horizon H (end conv emits H steps; pipeline uses H=1 + recursive)
    seq_length : lookback window T (30)
    subgraph_size : top-k neighbours kept in the learned graph (paper: 20)
    node_dim   : node-embedding dim for the graph constructor (paper: 40)
    layers     : number of TC→GC blocks (paper: 3)
    gcn_depth  : mix-hop propagation depth (paper: 2)
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        out_dim: int = 1,
        seq_length: int = 30,
        subgraph_size: int = 20,
        node_dim: int = 40,
        layers: int = 3,
        gcn_depth: int = 2,
        conv_channels: int = 16,
        residual_channels: int = 16,
        skip_channels: int = 32,
        end_channels: int = 64,
        dropout: float = 0.3,
        propalpha: float = 0.05,
        tanhalpha: float = 3.0,
        dilation_exponential: int = 1,
        layer_norm_affine: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.layers = layers
        self.seq_length = seq_length
        self.horizon = out_dim

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, alpha=tanhalpha)
        self.register_buffer("idx", torch.arange(num_nodes))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_dim, residual_channels, (1, 1))

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        rf = self.receptive_field
        new_dilation = 1
        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                rf_j = int(1 + (kernel_size - 1)
                           * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_j = j * (kernel_size - 1) + 1

            self.filter_convs.append(dilated_inception(residual_channels, conv_channels, new_dilation))
            self.gate_convs.append(dilated_inception(residual_channels, conv_channels, new_dilation))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, (1, 1)))

            seg = (seq_length - rf_j + 1) if seq_length > rf else (rf - rf_j + 1)
            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, (1, seg)))
            self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.norm.append(_LayerNorm((residual_channels, num_nodes, seg),
                                        elementwise_affine=layer_norm_affine))
            new_dilation *= dilation_exponential

        if seq_length > rf:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, (1, seq_length), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, (1, seq_length - rf + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, (1, rf), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, (1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N, C)
        Returns ŷ : (B, horizon, N, 1)
        """
        # → channels-first (B, C, N, T)
        inp = x.permute(0, 3, 2, 1)
        T = inp.size(3)
        if T < self.receptive_field:
            inp = F.pad(inp, (self.receptive_field - T, 0, 0, 0))

        idx = self.idx.to(x.device)
        adp = self.gc(idx)                                     # (N, N)

        xx = self.start_conv(inp)
        skip = self.skip0(F.dropout(inp, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = xx
            filt = torch.tanh(self.filter_convs[i](xx))
            gate = torch.sigmoid(self.gate_convs[i](xx))
            xx = filt * gate
            xx = F.dropout(xx, self.dropout, training=self.training)
            skip = self.skip_convs[i](xx) + skip
            xx = self.gconv1[i](xx, adp) + self.gconv2[i](xx, adp.t())
            xx = xx + residual[:, :, :, -xx.size(3):]
            xx = self.norm[i](xx, idx)

        skip = self.skipE(xx) + skip
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)                             # (B, horizon, N, 1)
        return out

    @torch.no_grad()
    def learned_adjacency(self) -> torch.Tensor:
        """Return the current learned directed top-k adjacency (N, N) for the
        dissertation's graph-quality analysis (§3.6)."""
        idx = self.idx.to(self.gc.emb1.weight.device)
        return self.gc(idx)
