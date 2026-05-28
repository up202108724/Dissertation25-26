"""
N-HiTS — Neural Hierarchical Interpolation for Time Series forecasting
(Challu et al., 2022, https://arxiv.org/abs/2201.12886).

Single-product direct multi-horizon forecaster.  Each block:

    1. MaxPool1d(k_l) on the lookback target -> multi-rate sampling.
    2. MLP maps the pooled lookback (concatenated with the *flattened*
       exogenous tensor) to (theta_b, theta_f) coefficients of much
       smaller dimension than (lookback, horizon).
    3. Linear interpolation expands theta_b/theta_f back to the full
       (lookback, horizon) lengths -> hierarchical interpolation.
    4. Doubly-residual: subtract backcast from the target residual, add
       forecast to the running forecast.

Stacks with larger kernel_size + smaller expressiveness ratio learn the
low-frequency component; later stacks (kernel_size=1) fit residual high
frequencies.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────────────────────────────────
class NHITSBlock(nn.Module):
    """
    One N-HiTS block: multi-rate pool -> MLP -> hierarchical interpolation.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        exog_in_size: int,
        pool_kernel_size: int,
        n_theta_backcast: int,
        n_theta_forecast: int,
        mlp_hidden: int = 512,
        n_mlp_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback         = lookback
        self.horizon          = horizon
        self.pool_kernel_size = max(1, int(pool_kernel_size))
        self.n_theta_backcast = int(n_theta_backcast)
        self.n_theta_forecast = int(n_theta_forecast)

        # MaxPool over the target lookback (ceil-mode to handle ragged divisions)
        self.pool = nn.MaxPool1d(
            kernel_size=self.pool_kernel_size,
            stride=self.pool_kernel_size,
            ceil_mode=True,
        )
        pooled_len = (lookback + self.pool_kernel_size - 1) // self.pool_kernel_size

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        in_size = pooled_len + exog_in_size

        layers = []
        prev = in_size
        for _ in range(n_mlp_layers):
            layers += [nn.Linear(prev, mlp_hidden), act(), nn.Dropout(dropout)]
            prev = mlp_hidden
        self.mlp = nn.Sequential(*layers)

        self.theta_b_head = nn.Linear(mlp_hidden, self.n_theta_backcast)
        self.theta_f_head = nn.Linear(mlp_hidden, self.n_theta_forecast)

    def _interpolate(self, theta: torch.Tensor, target_len: int) -> torch.Tensor:
        # theta: (B, n_theta) -> (B, 1, n_theta) -> interpolate -> (B, target_len)
        if theta.shape[1] == target_len:
            return theta
        return F.interpolate(
            theta.unsqueeze(1), size=target_len, mode="linear", align_corners=False,
        ).squeeze(1)

    def forward(self, y_back: torch.Tensor, exog_flat: torch.Tensor):
        """
        y_back    : (B, lookback) running residual on the target channel
        exog_flat : (B, exog_in_size) flattened exogenous features (lookback + future)
        Returns
        -------
        backcast : (B, lookback)
        forecast : (B, horizon)
        """
        # multi-rate sampling on the target residual
        pooled = self.pool(y_back.unsqueeze(1)).squeeze(1)            # (B, pooled_len)
        h = self.mlp(torch.cat([pooled, exog_flat], dim=-1))
        theta_b = self.theta_b_head(h)
        theta_f = self.theta_f_head(h)
        backcast = self._interpolate(theta_b, self.lookback)
        forecast = self._interpolate(theta_f, self.horizon)
        return backcast, forecast


# ──────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────
class NHITS(nn.Module):
    """
    N-HiTS forecaster.

    Parameters
    ----------
    lookback         : int                  – input window length L
    horizon          : int                  – forecast length H
    in_channels      : int                  – 1 (target) + n_exog channels
    pool_kernel_sizes: Sequence[int]        – one entry per stack; larger = lower freq
    n_theta_per_stack: Sequence[tuple]      – per-stack (n_theta_backcast, n_theta_forecast)
                                              defaults to (lookback, horizon) per stack
    n_blocks_per_stack : int                – number of blocks in each stack (weights NOT shared)
    mlp_hidden       : int                  – hidden width of each block's MLP
    n_mlp_layers     : int                  – MLP depth per block
    dropout          : float                – dropout in MLP
    activation       : str                  – relu / gelu / tanh
    future_exog_len  : int                  – number of *future* exog steps the model
                                              receives (= horizon when known a priori).
                                              Set to 0 to disable.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        in_channels: int,
        pool_kernel_sizes: Sequence[int] = (8, 4, 1),
        n_theta_per_stack: Sequence | None = None,
        n_blocks_per_stack: int = 1,
        mlp_hidden: int = 512,
        n_mlp_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        future_exog_len: int | None = None,
    ):
        super().__init__()
        self.lookback        = lookback
        self.horizon         = horizon
        self.in_channels     = in_channels
        self.n_exog          = max(0, in_channels - 1)
        self.future_exog_len = horizon if future_exog_len is None else int(future_exog_len)

        # Exogenous features fed to every block: flatten the lookback exog
        # plus the (optional) future-exog window.  This lets each block
        # condition on calendar / holiday features for both windows.
        exog_in_size = self.n_exog * (lookback + self.future_exog_len)

        # Default theta dims: smaller for low-freq stacks (larger pool), bigger for high-freq.
        if n_theta_per_stack is None:
            n_theta_per_stack = []
            for k in pool_kernel_sizes:
                nb = max(2, lookback // max(1, k))
                nf = max(2, horizon  // max(1, k))
                n_theta_per_stack.append((nb, nf))
        assert len(n_theta_per_stack) == len(pool_kernel_sizes)

        blocks = []
        for k_l, (n_theta_b, n_theta_f) in zip(pool_kernel_sizes, n_theta_per_stack):
            for _ in range(n_blocks_per_stack):
                blocks.append(NHITSBlock(
                    lookback=lookback,
                    horizon=horizon,
                    exog_in_size=exog_in_size,
                    pool_kernel_size=k_l,
                    n_theta_backcast=n_theta_b,
                    n_theta_forecast=n_theta_f,
                    mlp_hidden=mlp_hidden,
                    n_mlp_layers=n_mlp_layers,
                    dropout=dropout,
                    activation=activation,
                ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, future_exog: torch.Tensor | None = None) -> torch.Tensor:
        """
        x           : (B, L, C)  — column 0 is the target, columns 1: are exog
        future_exog : (B, H, n_exog) — known exog for the forecast window
                      (zeros / None when not available)

        Returns
        -------
        (B, H, 1) — forecast on the target channel
        """
        B, L, C = x.shape
        if L != self.lookback or C != self.in_channels:
            raise ValueError(
                f"Expected input (B, {self.lookback}, {self.in_channels}), "
                f"got (B, {L}, {C})."
            )

        y_back  = x[:, :, 0]                                       # (B, L) target lookback
        exog_lb = x[:, :, 1:].reshape(B, -1) if self.n_exog else x.new_zeros(B, 0)

        if self.n_exog and self.future_exog_len > 0:
            if future_exog is None:
                fut = x.new_zeros(B, self.future_exog_len, self.n_exog)
            else:
                fut = future_exog
                if fut.shape[1] != self.future_exog_len:
                    # pad / trim to expected length
                    if fut.shape[1] > self.future_exog_len:
                        fut = fut[:, : self.future_exog_len]
                    else:
                        pad = x.new_zeros(B, self.future_exog_len - fut.shape[1], self.n_exog)
                        fut = torch.cat([fut, pad], dim=1)
            exog_flat = torch.cat([exog_lb, fut.reshape(B, -1)], dim=-1)
        else:
            exog_flat = exog_lb

        residual = y_back
        forecast = x.new_zeros(B, self.horizon)
        for block in self.blocks:
            backcast, f_block = block(residual, exog_flat)
            residual = residual - backcast
            forecast = forecast + f_block

        return forecast.unsqueeze(-1)                              # (B, H, 1)
