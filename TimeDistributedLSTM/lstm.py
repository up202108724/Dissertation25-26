import torch
import torch.nn as nn

    
# -----------------------------------------------------------------------------
# Model: one class, two heads selected by a flag
# -----------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2,
                 time_distributed=False):
        super().__init__()
        self.time_distributed = time_distributed
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                 # (B, seq, hidden)
        if self.time_distributed:
            return self.linear(self.drop(lstm_out))      # (B, seq, 1)
        return self.linear(self.drop(lstm_out[:, -1, :]))  # (B, 1)