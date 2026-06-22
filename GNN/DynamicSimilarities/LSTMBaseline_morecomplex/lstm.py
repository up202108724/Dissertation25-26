import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2,
                 bidirectional=False, pool='last', head_hidden=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.pool = pool
        if pool == 'attention':                      # learn a weight per timestep
            self.attn = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(dropout)
        if head_hidden:                              # nonlinear MLP head
            self.head = nn.Sequential(
                nn.Linear(out_dim, head_hidden), nn.LayerNorm(head_hidden),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
            )
        else:
            self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                        # (B, T, out_dim)
        if self.pool == 'mean':
            feat = out.mean(dim=1)
        elif self.pool == 'attention':
            w = torch.softmax(self.attn(out), dim=1) # (B, T, 1)
            feat = (w * out).sum(dim=1)
        else:                                        # 'last' — current behavior
            feat = out[:, -1, :]
        return self.head(self.drop(feat))