import torch
import torch.nn as nn


class LSTMWithEmbedding(nn.Module):
    """
    LSTM forecaster with a trainable embedding layer initialised from
    pre-trained Graph2Vec embeddings.

    At every timestep t the model looks up embedding E[t] (learnable) and
    concatenates it with [target_value | exog_features] before feeding the
    combined vector into the LSTM.  The embedding table is trained
    end-to-end together with the rest of the network, so the Graph2Vec
    pre-training acts as a warm start that the forecasting loss can refine.

    Parameters
    ----------
    ts_input_size         : 1 + n_exog  (target + calendar features)
    emb_dim               : Graph2Vec embedding dimension
    hidden_size           : LSTM hidden state size
    num_layers            : number of LSTM layers
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
        ts_input_size: int,
        emb_dim: int,
        hidden_size: int = 50,
        num_layers: int = 1,
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

        lstm_input = ts_input_size + emb_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_ts: torch.Tensor,    # (B, L, ts_input_size)
        emb_idx: torch.Tensor,  # (B, L)  long
    ) -> torch.Tensor:          # (B, 1)
        emb = self.embedding(emb_idx)            # (B, L, emb_dim)
        x   = torch.cat([x_ts, emb], dim=-1)     # (B, L, ts_input_size + emb_dim)
        lstm_out, _ = self.lstm(x)
        last = self.drop(lstm_out[:, -1, :])
        return self.linear(last)

