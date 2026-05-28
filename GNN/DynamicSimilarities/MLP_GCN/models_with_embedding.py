import torch
import torch.nn as nn


class MLPWithEmbedding(nn.Module):
    """
    Flat MLP forecaster with a trainable embedding layer initialised from
    pre-trained Graph2Vec embeddings.

    The per-timestep (target + exog + embedding) features are flattened over
    the lookback window and passed through a 2-layer MLP.  The embedding
    table is trained end-to-end together with the rest of the network, so
    the Graph2Vec pre-training acts as a warm start that the forecasting
    loss can refine.

    Parameters
    ----------
    seq_length            : lookback window length (L)
    ts_input_size         : 1 + n_exog  (target + calendar features)
    emb_dim               : Graph2Vec embedding dimension
    hidden_sizes          : tuple of hidden layer widths
    dropout               : dropout probability
    num_graphs            : total number of positions in the embedding table
                            (= length of the full aligned_embeddings array)
    pretrained_embeddings : (num_graphs, emb_dim) float tensor used to
                            initialise the embedding table; if None the
                            table is initialised randomly
    freeze_embeddings     : if True the embedding weights are frozen
                            (equivalent to the old concatenation approach)
    """

    def __init__(
        self,
        seq_length: int,
        ts_input_size: int,
        emb_dim: int,
        hidden_sizes: tuple = (128, 64),
        dropout: float = 0.2,
        num_graphs: int = 1,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_graphs, emb_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                pretrained_embeddings.float().clone()
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad_(False)

        flat_size = seq_length * (ts_input_size + emb_dim)
        layers = []
        in_size = flat_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_ts: torch.Tensor,     # (B, L, ts_input_size)
        emb_idx: torch.Tensor,  # (B, L)  long
    ) -> torch.Tensor:           # (B, 1)
        emb = self.embedding(emb_idx)           # (B, L, emb_dim)
        x   = torch.cat([x_ts, emb], dim=-1)    # (B, L, ts_input_size + emb_dim)
        x   = x.view(x.size(0), -1)             # (B, flat_size)
        return self.mlp(x)
