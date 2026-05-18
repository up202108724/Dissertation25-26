from torch import nn


class MLPForecaster(nn.Module):
    """Flat MLP forecaster used as the no-graph baseline."""

    def __init__(
        self,
        lookback: int,
        in_channels: int,
        horizon: int,
        out_dim: int = 1,
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback    = lookback
        self.in_channels = in_channels
        self.horizon     = horizon
        self.out_dim     = out_dim

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = lookback * in_channels
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, horizon * out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C)
        b = x.size(0)
        y = self.net(x.reshape(b, -1))          # (B, H*out_dim)
        return y.reshape(b, self.horizon, self.out_dim)
