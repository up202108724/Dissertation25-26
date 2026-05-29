from torch import nn


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

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

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
