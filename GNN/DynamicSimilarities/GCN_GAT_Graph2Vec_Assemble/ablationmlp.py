from torch import nn
import torch


class AblationMLPForecaster(nn.Module):
    """
    Pure TimeDistributed MLP with the same forward signature as
    SimpleGCNMLPForecaster — (pyg_batch, target_node_indices, ts_seq) — but
    ignores the graph inputs entirely.

    Used for the ablate_z=True condition so that:
      - no zero-padded d_g columns waste first-layer capacity
      - parameter count matches a fair no-graph baseline
    """
    def __init__(
        self,
        ts_input_size: int,
        hidden_sizes=(32, 16),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.tdmlp = TimeDistributedMLPForecaster(
            in_channels=ts_input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        # ts_seq: (B, L, ts_input_size) — graph inputs intentionally ignored
        return self.tdmlp(ts_seq)  # (B, 1, 1)


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
