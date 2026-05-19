import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Plain LSTM baseline. Accepts an optional `emb` argument so it can be
    used in the same training loop as LSTMWithLearnedEmbedding."""

    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, emb=None):
        lstm_out, _ = self.lstm(x)
        last = self.drop(lstm_out[:, -1, :])
        return self.linear(last)


class LSTMWithLearnedEmbedding(nn.Module):
    """LSTM with a trainable linear projection on top of pre-computed graph embeddings.

    The Graph2Vec embeddings are NOT concatenated as raw features.  Instead they
    pass through a small trainable projection layer (emb_dim → proj_dim) whose
    weights are learned jointly with the LSTM through the forecasting loss.  The
    projected embedding is then concatenated with the time-series / calendar
    features at every timestep before entering the LSTM.

    Args:
        input_size:   Number of time-varying input features *excluding* embeddings
                      (i.e. 1 target + n_exog calendar features).
        hidden_size:  LSTM hidden state size.
        num_layers:   Number of stacked LSTM layers.
        dropout:      Dropout probability applied after the last LSTM timestep.
        emb_dim:      Dimensionality of the raw Graph2Vec embeddings.
        proj_dim:     Dimensionality of the learned embedding projection.  The
                      LSTM's effective input size becomes input_size + proj_dim.
        finetune_emb: If True, passes the incoming embeddings through a learned
                      residual adapter (emb_dim → emb_dim) before projection, so
                      the raw Graph2Vec vectors are effectively fine-tuned for the
                      forecasting task.  Adds emb_dim² extra parameters — only
                      recommended when you have enough training data.
    """

    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2,
                 emb_dim=64, proj_dim=16, finetune_emb=False):
        super().__init__()

        self.finetune_emb = finetune_emb
        if finetune_emb:
            # Residual adapter: refines the raw Graph2Vec vectors end-to-end.
            # Initialised close to identity so early training is stable.
            self.emb_adapter = nn.Linear(emb_dim, emb_dim, bias=True)
            nn.init.eye_(self.emb_adapter.weight)
            nn.init.zeros_(self.emb_adapter.bias)

        # Trainable projection: (adapted) Graph2Vec emb → compact representation
        self.emb_projector = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=input_size + proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, emb):
        """
        Args:
            x:   (batch, seq_len, input_size)  — target + calendar features.
            emb: (batch, seq_len, emb_dim)     — raw Graph2Vec embeddings.
        Returns:
            (batch, 1) scalar forecast.
        """
        if self.finetune_emb:
            # Residual update: keeps original structure while allowing task-specific drift
            emb = emb + self.emb_adapter(emb)

        projected = self.emb_projector(emb)
        x_aug = torch.cat([x, projected], dim=-1)
        lstm_out, _ = self.lstm(x_aug)
        last = self.drop(lstm_out[:, -1, :])
        return self.linear(last)