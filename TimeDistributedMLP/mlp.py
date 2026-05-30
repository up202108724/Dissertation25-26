import torch
from torch import nn
import torch.nn.functional as F


_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

class TimeDistributedMLPForecaster(nn.Module):
    """
    Applies a small MLP to each timestep independently (shared weights),
    then concatenates [last_step, mean] and projects to the forecast.

    Using only mean-pool discards temporal recency which is critical for
    1-step-ahead forecasting. Concatenating last + mean preserves both
    the most recent state and the global context.

    Params: Linear(C → H) shared across L timesteps + small head.
    ~L× fewer parameters than a flat MLPForecaster.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_sizes=(32, 16),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = in_channels
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        self.timestep_net = nn.Sequential(*layers)
        self.out = nn.Linear(prev * 2, 1)  # last + mean concatenated

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        h = self.timestep_net(x.reshape(B * L, C))  # (B*L, H)
        h = h.reshape(B, L, -1)                      # (B, L, H)
        last = h[:, -1, :]                            # (B, H) — most recent step
        mean = h.mean(dim=1)                          # (B, H) — global context
        combined = torch.cat([last, mean], dim=-1)    # (B, 2H)
        return self.out(combined).unsqueeze(1)        # (B, 1, 1)
