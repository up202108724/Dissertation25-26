import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Channel-independent PatchTST (Nie et al., ICLR 2023) on the target
    channel ``x[..., 0]``: RevIN, patching, Transformer encoder, flatten head.

    ``n_covariates > 0`` patches the covariate channels ``x[..., 1:1 + n_covariates]``
    with the same windows as the target, projects each patch to ``d_model`` and
    adds it to the matching target-patch embedding before the encoder (the graph
    hybrids pass their per-day embedding z here).  The projection is
    zero-initialised, so a hybrid starts out identical to the univariate
    baseline.  RevIN normalises the target only.  Any further channels are
    ignored.

    Defaults are scaled down for a 30-day lookback: patch 8 / stride 4 with
    end-padding gives 7 patches.
    """

    def __init__(self, lookback: int, horizon: int = 1, patch_len: int = 8,
                 stride: int = 4, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 64, dropout: float = 0.2,
                 revin: bool = True, n_covariates: int = 0, eps: float = 1e-5):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.revin, self.eps = revin, eps
        self.n_covariates = n_covariates
        n_patches = (lookback + stride - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if n_covariates:
            self.embed_cov = nn.Linear(patch_len * n_covariates, d_model)
            nn.init.zeros_(self.embed_cov.weight)
            nn.init.zeros_(self.embed_cov.bias)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, horizon),
        )
        if revin:
            self.rev_w = nn.Parameter(torch.ones(1))
            self.rev_b = nn.Parameter(torch.zeros(1))

    def _patch(self, s):
        """(B, L, ...) → (B, N, ..., P), end-padded by repeating the last step."""
        s = torch.cat([s, s[:, -1:].repeat_interleave(self.stride, dim=1)], dim=1)
        return s.unfold(1, self.patch_len, self.stride)

    def forward(self, x):
        y = x[..., 0]                                          # (B, L)
        if self.revin:
            mu = y.mean(dim=1, keepdim=True).detach()
            sd = torch.sqrt(y.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            y = (y - mu) / sd * self.rev_w + self.rev_b
        h = self.embed(self._patch(y)) + self.pos              # (B, N, d_model)
        if self.n_covariates:
            cov = self._patch(x[..., 1:1 + self.n_covariates])  # (B, N, C, P)
            h = h + self.embed_cov(cov.flatten(start_dim=2))
        out = self.head(self.encoder(self.drop(h)))            # (B, H)
        if self.revin:
            out = (out - self.rev_b) / (self.rev_w + self.eps) * sd + mu
        return out.unsqueeze(-1)                               # (B, H, 1)
