import torch
import torch.nn as nn


class DLinear(nn.Module):
    """DLinear (Zeng et al., AAAI 2023) on the target channel ``x[..., 0]``.

    The target window is split into a moving-average trend and a remainder,
    each mapped to the forecast by its own linear layer over the lookback.

    ``n_covariates > 0`` adds one linear map over the lookback per covariate
    channel ``x[..., 1:1 + n_covariates]`` (the graph hybrids pass their per-day
    embedding z here).  That map is zero-initialised, so a hybrid starts out
    identical to the univariate baseline.  Any further channels are ignored.

    ``kernel_size`` defaults to one week: the original 25 exceeds half of the
    30-day lookback and would make the trend nearly constant.
    """

    def __init__(self, lookback: int, horizon: int = 1, kernel_size: int = 7,
                 n_covariates: int = 0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (symmetric moving average)")
        self.kernel_size = kernel_size
        self.n_covariates = n_covariates
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)
        if n_covariates:
            self.linear_cov = nn.Linear(lookback * n_covariates, horizon)
            nn.init.zeros_(self.linear_cov.weight)
            nn.init.zeros_(self.linear_cov.bias)

    def forward(self, x):
        y = x[..., 0]                                          # (B, L)
        half = (self.kernel_size - 1) // 2
        padded = torch.cat([y[:, :1].repeat(1, half), y, y[:, -1:].repeat(1, half)], dim=1)
        trend = self.avg(padded.unsqueeze(1)).squeeze(1)       # (B, L)
        out = self.linear_trend(trend) + self.linear_seasonal(y - trend)
        if self.n_covariates:
            cov = x[..., 1:1 + self.n_covariates]              # (B, L, C)
            out = out + self.linear_cov(cov.transpose(1, 2).reshape(cov.shape[0], -1))
        return out.unsqueeze(-1)                               # (B, H, 1)
