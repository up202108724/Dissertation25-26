
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

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
        
        # Build residual blocks with exponentially increasing dilation
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