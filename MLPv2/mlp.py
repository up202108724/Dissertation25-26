import torch
from torch import nn
import torch.nn.functional as F


_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLPForecaster(nn.Module):
    """
    Strict 1-step-ahead MLP forecaster.

    Input  : x of shape (B, L, C)  — last L observations of target + exogs.
    Output : y_hat of shape (B, 1) — scaled prediction for the next step.

    Multi-step forecasts are produced by recursive rollout (see
    `inference.recursive_inference_dynamic_exog`), NOT by a multi-step head.
    """

    def __init__(
        self,
        lookback: int,
        in_channels: int,
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback = lookback
        self.in_channels = in_channels

        act = _ACT[activation]

        layers = []
        prev = lookback * in_channels
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C) -> (B, L*C) -> (B, 1)
        b = x.size(0)
        x = x.reshape(b, -1)
        return self.net(x)


# -----------------------------------------------------------------------------
# Option 2: Time-distributed (per-step shared) projection + small MLP head.
#
# Fix for the flatten bottleneck of `MLPForecaster`: instead of collapsing
# (B, L, C) into one big vector and letting Linear(L*C, H) re-learn which slot
# corresponds to which (lag, channel), we apply a SHARED Linear(C -> d) to
# every timestep. Channel mixing is therefore time-invariant; only the head
# sees the flattened (B, L*d) vector and reasons about temporal position.
#
# `hidden_sizes` is interpreted as:
#     hidden_sizes[0]  -> per-step projection dim d
#     hidden_sizes[1:] -> head MLP hidden sizes
# Falls back to d = hidden_sizes[0] only if a single value is given.
# -----------------------------------------------------------------------------
class TimeDistributedMLPForecaster(nn.Module):
    """
    Strict 1-step-ahead forecaster with shared per-timestep channel mixing.

    Input  : x of shape (B, L, C)
    Output : y_hat of shape (B, 1)
    """

    def __init__(
        self,
        lookback: int,
        in_channels: int,
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one entry (per-step proj dim).")

        self.lookback = lookback
        self.in_channels = in_channels

        act = _ACT[activation]
        proj_dim = hidden_sizes[0]
        head_dims = hidden_sizes[1:]

        # Per-step shared projection: applied as Linear over the last dim of (B, L, C).
        self.proj = nn.Sequential(
            nn.Linear(in_channels, proj_dim),
            act(),
            nn.Dropout(dropout),
        )

        layers = []
        prev = lookback * proj_dim
        for hs in head_dims:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C) -> (B, L, d) -> (B, L*d) -> (B, 1)
        b = x.size(0)
        h = self.proj(x)
        h = h.reshape(b, -1)
        return self.head(h)


# -----------------------------------------------------------------------------
# Option 3: small causal dilated TCN.
#
# Real weight sharing across time via Conv1d. Causal (left) padding makes sure
# the prediction for the next step depends only on past observations. Two or
# three residual blocks with exponentially increasing dilation (1, 2, 4, ...)
# cover a receptive field of (kernel_size - 1) * sum(dilations) + 1, which for
# kernel=3 and 3 blocks reaches 15 -> easily covers a 30-step lookback even
# without stacking more layers.
#
# `hidden_sizes` is interpreted as the per-block channel count list:
#     hidden_sizes = (64, 64, 64)  -> 3 blocks of 64 channels, dilations 1,2,4
# The final feature map is global-average-pooled over time and mapped to (B, 1).
# -----------------------------------------------------------------------------
class _CausalConv1d(nn.Module):
    """Conv1d with left-only padding so output[t] depends only on input[<=t]."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        # x: (B, C, L)
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, act_cls):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.act1 = act_cls()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act2 = act_cls()
        self.drop2 = nn.Dropout(dropout)
        # 1x1 projection for residual when channel counts differ.
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.drop1(self.act1(self.conv1(x)))
        h = self.drop2(self.act2(self.conv2(h)))
        return h + self.res(x)


class TCNForecaster(nn.Module):
    """
    Strict 1-step-ahead causal TCN forecaster.

    Input  : x of shape (B, L, C)
    Output : y_hat of shape (B, 1)
    """

    def __init__(
        self,
        lookback: int,
        in_channels: int,
        hidden_sizes=(64, 64, 64),
        dropout: float = 0.2,
        activation: str = "relu",
        kernel_size: int = 3,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must list at least one TCN block channel count.")

        self.lookback = lookback
        self.in_channels = in_channels

        act_cls = _ACT[activation]
        blocks = []
        prev = in_channels
        for i, ch in enumerate(hidden_sizes):
            blocks.append(_TCNBlock(prev, ch, kernel_size, dilation=2 ** i,
                                    dropout=dropout, act_cls=act_cls))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        # Read out from the LAST timestep (causal -> contains full past context).
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        h = x.transpose(1, 2)
        h = self.tcn(h)            # (B, ch, L)
        h = h[:, :, -1]            # (B, ch) -- last timestep only
        return self.head(h)


# -----------------------------------------------------------------------------
# Factory used by train.py so the strategy scripts can swap models with one
# string argument and no other changes.
# -----------------------------------------------------------------------------
_FORECASTERS = {
    "mlp": MLPForecaster,
    "tdmlp": TimeDistributedMLPForecaster,
    "tcn": TCNForecaster,
}


def build_forecaster(
    model_type: str,
    lookback: int,
    in_channels: int,
    hidden_sizes,
    dropout: float = 0.2,
    activation: str = "relu",
) -> nn.Module:
    """Instantiate a forecaster by name. Valid keys: 'mlp', 'tdmlp', 'tcn'."""
    key = model_type.lower()
    if key not in _FORECASTERS:
        raise ValueError(f"Unknown model_type={model_type!r}. "
                         f"Expected one of {sorted(_FORECASTERS)}.")
    return _FORECASTERS[key](
        lookback=lookback,
        in_channels=in_channels,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
    )

