from torch import nn
import torch


class AblationLSTMForecaster(nn.Module):
    """
    Pure LSTM with the same forward signature as
    SimpleGCNLSTMForecaster — (pyg_batch, target_node_indices, ts_seq) — but
    ignores the graph inputs entirely.

    Used for the ablate_z=True condition so that:
      - no zero-padded d_g columns waste the LSTM's input capacity
        (SimpleGCNLSTMForecaster with ablate_z=True still has
         input_size = lstm_input_size + d_g, with d_g columns always zero)
      - parameter count reflects a fair no-graph baseline
    """
    def __init__(
        self,
        lstm_input_size: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(p=dropout)
        self.head      = nn.Linear(lstm_hidden, horizon)

        self.horizon = horizon
        # Kept for API parity with SimpleGCNLSTMForecaster; never read here
        # since the graph branch does not exist in this model.
        self.ablate_z = True

    def forward(self, pyg_batch, target_node_indices, ts_seq):
        # ts_seq: (B, L, lstm_input_size) — graph inputs intentionally ignored
        out, _ = self.lstm(ts_seq)
        last   = self.lstm_drop(out[:, -1, :])    # (B, hidden) — last step
        pred   = self.head(last)                  # (B, horizon)
        return pred.unsqueeze(-1)                 # (B, horizon, 1)
