from torch import nn
import torch

class MLPForecaster(nn.Module):
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
        self.lookback = lookback
        self.in_channels = in_channels
        self.horizon = horizon
        self.out_dim = out_dim

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = lookback * in_channels
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, horizon * out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x, emb=None):
        # x: (B, L, C)  — emb ignored (no projection layer in this class)
        b = x.size(0)
        x = x.reshape(b, -1)
        y = self.net(x)
        return y.view(b, self.horizon, self.out_dim)


class MLPWithLearnedEmbedding(nn.Module):
    """MLP forecaster with a trainable linear projection on top of pre-computed
    Graph2Vec embeddings.

    The Graph2Vec embeddings are NOT concatenated as raw features.  Instead, at
    every timestep in the lookback window they pass through a small trainable
    projection layer (emb_dim → proj_dim) whose weights are learned jointly with
    the MLP's weights through the forecasting loss.  The projected sequence is
    then concatenated with the target / calendar features before the whole window
    is flattened and fed to the MLP hidden layers.

    Args:
        lookback:     Length of the input window.
        in_channels:  Feature columns *excluding* embeddings (1 target + n_exog).
        horizon:      Forecast horizon.
        emb_dim:      Dimensionality of the raw Graph2Vec embeddings.
        proj_dim:     Dimensionality of the learned embedding projection.
                      The MLP's flat input size becomes lookback*(in_channels+proj_dim).
        out_dim:      Output channels (usually 1).
        hidden_sizes: Tuple of hidden layer widths for the MLP trunk.
        dropout:      Dropout probability in the MLP trunk.
        activation:   Activation function name ('relu', 'gelu', or 'tanh').
    """

    def __init__(
        self,
        lookback: int,
        in_channels: int,
        horizon: int,
        emb_dim: int,
        proj_dim: int = 16,
        out_dim: int = 1,
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback = lookback
        self.in_channels = in_channels
        self.horizon = horizon
        self.out_dim = out_dim

        # Trainable projection: raw Graph2Vec emb → learned representation
        self.emb_projector = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
        )

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = lookback * (in_channels + proj_dim)
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, horizon * out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x, emb):
        """
        Args:
            x:   (B, L, in_channels)  — target + calendar features.
            emb: (B, L, emb_dim)      — raw Graph2Vec embeddings.
        Returns:
            (B, horizon, out_dim) forecast.
        """
        b = x.size(0)
        # Project embeddings at every timestep: (B, L, proj_dim)
        proj = self.emb_projector(emb)
        # Concatenate with time-series features then flatten
        x_aug = torch.cat([x, proj], dim=-1)       # (B, L, in_channels + proj_dim)
        x_flat = x_aug.reshape(b, -1)              # (B, L * (in_channels + proj_dim))
        y = self.net(x_flat)
        return y.view(b, self.horizon, self.out_dim)