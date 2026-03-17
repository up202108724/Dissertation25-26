
import torch
import torch.nn as nn
import torch.nn.functional as F

class Joint_GCN_LSTM(nn.Module):
    def __init__(self, gcn_model, lstm_input_size, lstm_hidden_size, lstm_num_layers, gcn_embed_dim, dropout=0.2):
        super(Joint_GCN_LSTM, self).__init__()
        
        self.gcn = gcn_model
        
        # We increase input_size by the gcn embedding dimension since we will concatenate them
        # Alternatively, we can concatenate the GCN embedding to the LSTM output before the final linear layer.
        # Concatenating to the input at each time step is usually better for time-series.
        self.lstm = nn.LSTM(
            input_size=lstm_input_size + gcn_embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        
        # The final dense layer predicts the next time step
        self.linear = nn.Linear(lstm_hidden_size, 1)
        
        self.gcn_embed_dim = gcn_embed_dim

    def forward(self, ts_x, graph_x, graph_adj, target_node_idx):
        # 1. Forward pass through GCN to get structural embeddings for all nodes
        # We use a modified GCN forward (assuming you removed log_softmax for extracting embeddings)
        h = F.relu(self.gcn.gc1(graph_x, graph_adj))
        node_embeddings = self.gcn.gc2(h, graph_adj) 
        
        # 2. Extract the specific embedding for the node (item) we are currently forecasting
        z_i = node_embeddings[target_node_idx] # Shape: (gcn_embed_dim)
        
        # 3. Expand and concatenate the graph embedding to every time-step of the LSTM sequence input
        batch_size, seq_length, _ = ts_x.size()
        
        # Expand z_i to match batch and seq_length: (batch_size, seq_length, gcn_embed_dim)
        z_i_expanded = z_i.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        
        # Concatenate TS input with Graph Embeddings: shape (batch_size, seq_length, lstm_input_size + gcn_embed_dim)
        combined_x = torch.cat([ts_x, z_i_expanded], dim=2)
        
        # 4. Forward pass through LSTM
        lstm_out, _ = self.lstm(combined_x)
        last_hidden = self.drop(lstm_out[:, -1, :])
        
        # 5. Final Prediction
        pred = self.linear(last_hidden)
        return pred