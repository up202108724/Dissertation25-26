import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionGCNLSTM(nn.Module):
    def __init__(
        self,
        gcn_model,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        gcn_embed_dim,
        horizon=1,
        dropout=0.2,
    ):
        super().__init__()

        self.gcn = gcn_model
        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        fusion_dim = lstm_hidden_size + gcn_embed_dim
        hidden_fusion = max(16, fusion_dim // 2)

        self.fc1 = nn.Linear(fusion_dim, hidden_fusion)
        self.fc2 = nn.Linear(hidden_fusion, horizon)

    def forward(self, ts_x, graph_x, graph_adj, target_node_idx):
        """
        ts_x: (B, L, F_ts)
        graph_x: (N, F_g)
        graph_adj: (N, N)
        target_node_idx: (B,)  -> one target node per sample
        """

        # Temporal branch
        lstm_out, _ = self.lstm(ts_x)
        h_last_ts = self.drop(lstm_out[:, -1, :])   # (B, H)

        # Graph branch
        h_gcn = F.relu(self.gcn.gc1(graph_x, graph_adj))
        node_embeddings = self.gcn.gc2(h_gcn, graph_adj)   # (N, Dg)

        # Gather target node embeddings for each sample in the batch
        z_i_graph = node_embeddings[target_node_idx]       # (B, Dg)

        # Late fusion
        combined = torch.cat([h_last_ts, z_i_graph], dim=-1)

        out = F.relu(self.fc1(combined))
        out = self.drop(out)
        pred = self.fc2(out)

        return pred
    
class GCN_LSTM_ConcatPerStep(nn.Module):
    def __init__(self, ts_input_dim, lstm_hidden_dim, graph_embed_dim, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=ts_input_dim + graph_embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_dim, horizon)

    def forward(self, ts_x, z_target):
        z_rep = z_target.unsqueeze(1).expand(-1, ts_x.size(1), -1)
        combined_x = torch.cat([ts_x, z_rep], dim=-1)

        lstm_out, _ = self.lstm(combined_x)
        h_last = lstm_out[:, -1, :]
        return self.fc(h_last)
    
class GCN_LSTM_InitState(nn.Module):
    def __init__(self, ts_input_dim, lstm_hidden_dim, graph_embed_dim, horizon=1):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(
            input_size=ts_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        self.h0_proj = nn.Linear(graph_embed_dim, lstm_hidden_dim)
        self.c0_proj = nn.Linear(graph_embed_dim, lstm_hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim, horizon)

    def forward(self, ts_x, z_target):
        # ts_x: (B, L, F_ts)
        # z_target: (B, D_g)

        h0 = self.h0_proj(z_target).unsqueeze(0)   # (1, B, H)
        c0 = self.c0_proj(z_target).unsqueeze(0)   # (1, B, H)

        lstm_out, _ = self.lstm(ts_x, (h0, c0))
        h_last = lstm_out[:, -1, :]
        return self.fc(h_last)
    
class GCN_LSTM_Gated(nn.Module):
    def __init__(self, ts_input_dim, lstm_hidden_dim, graph_embed_dim, horizon=1):
        super().__init__()
        self.gate = nn.Linear(graph_embed_dim, ts_input_dim)
        self.lstm = nn.LSTM(ts_input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, horizon)

    def forward(self, ts_x, z_target):
        gate = torch.sigmoid(self.gate(z_target))          # (B, F_ts)
        gated_x = ts_x * gate.unsqueeze(1)                # (B, L, F_ts)

        lstm_out, _ = self.lstm(gated_x)
        h_last = lstm_out[:, -1, :]
        return self.fc(h_last)