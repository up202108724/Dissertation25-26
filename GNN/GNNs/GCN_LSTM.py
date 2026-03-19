
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

class DynamicGraphEncoder(nn.Module):
    def __init__(self, gcn_model):
        """
        A dedicated graph encoder that takes a sequence of graphs and returns their dynamic embeddings.
        """
        super(DynamicGraphEncoder, self).__init__()
        self.gcn = gcn_model

    def forward(self, graph_x_seq, graph_adj_seq, target_node_idx=None):
        """
        Encodes a sequence of graphs natively.
        If target_node_idx is provided, extracts embeddings only for that node.
        Otherwise, returns embeddings for all nodes.
        
        Outputs tensor shape:
        - If target_node_idx is not None: (seq_length, gcn_embed_dim)
        - If target_node_idx is None: (seq_length, num_nodes, gcn_embed_dim)
        """
        seq_length = len(graph_x_seq)
        dynamic_embeddings = []
        
        for t in range(seq_length):
            h_t = F.relu(self.gcn.gc1(graph_x_seq[t], graph_adj_seq[t]))
            node_embeddings_t = self.gcn.gc2(h_t, graph_adj_seq[t])
            
            if target_node_idx is not None:
                z_i_t = node_embeddings_t[target_node_idx]
                dynamic_embeddings.append(z_i_t)
            else:
                dynamic_embeddings.append(node_embeddings_t)
                
        # Stack embeddings along the sequence dimension
        return torch.stack(dynamic_embeddings, dim=0)

class Dynamic_GCN_LSTM(nn.Module):
    def __init__(self, gcn_model, lstm_input_size, lstm_hidden_size, lstm_num_layers, gcn_embed_dim, dropout=0.2):
        super(Dynamic_GCN_LSTM, self).__init__()
        
        self.encoder = DynamicGraphEncoder(gcn_model)
        
        # We concatenate the GCN embedding to the TS features at each time step
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

    def forward(self, ts_x, graph_x_seq, graph_adj_seq, target_node_idx):
        """
        ts_x: (batch_size, seq_length, lstm_input_size)
        graph_x_seq: List or Tensor of feature matrices for each time step in the sequence
        graph_adj_seq: List or Tensor of adjacency matrices for each time step in the sequence
        target_node_idx: The index of the node we are forecasting
        """
        batch_size, seq_length, _ = ts_x.size()
        
        # 1. Forward pass through Dynamic Graph Encoder to get sequence embeddings for the target node
        z_i_seq = self.encoder(graph_x_seq, graph_adj_seq, target_node_idx) # (seq_length, gcn_embed_dim)
        
        # Expand batched embeddings: (batch_size, seq_length, gcn_embed_dim)
        z_i_expanded = z_i_seq.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. Concatenate TS input with Graph Embeddings dynamically over time
        combined_x = torch.cat([ts_x, z_i_expanded], dim=2)
        
        # 3. Forward pass through LSTM
        lstm_out, _ = self.lstm(combined_x)
        last_hidden = self.drop(lstm_out[:, -1, :])
        
        # 4. Final Prediction
        pred = self.linear(last_hidden)
        return pred

class LateFusion_Discrete_GCN_LSTM(nn.Module):
    def __init__(self, gcn_model, lstm_input_size, lstm_hidden_size, lstm_num_layers, gcn_embed_dim, horizon=1, dropout=0.2):
        super(LateFusion_Discrete_GCN_LSTM, self).__init__()
        
        self.gcn = gcn_model
        
        # LSTM processes ONLY the temporal features (no graph concatenation yet)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        
        # MLP takes the combined [h_ts || z_graph] to predict the horizon
        # Combined size is lstm_hidden_size (from last LSTM output) + gcn_embed_dim
        self.fc1 = nn.Linear(lstm_hidden_size + gcn_embed_dim, (lstm_hidden_size + gcn_embed_dim) // 2)
        self.fc2 = nn.Linear((lstm_hidden_size + gcn_embed_dim) // 2, horizon)

    def forward(self, ts_x, graph_x, graph_adj, target_node_idx):
        """
        Late fusion for a *static* or single graph structure at forecasting time.
        """
        # 1. Temporal Encoding (LSTM)
        lstm_out, _ = self.lstm(ts_x)
        h_last_ts = self.drop(lstm_out[:, -1, :]) # (batch_size, lstm_hidden_size)
        
        # 2. Structural Encoding (GCN)
        # Assuming single adjacency and feature matrix at the terminal window
        h_gcn = F.relu(self.gcn.gc1(graph_x, graph_adj))
        node_embeddings = self.gcn.gc2(h_gcn, graph_adj) 
        
        # Extract target node embedding
        z_i_graph = node_embeddings[target_node_idx] # (gcn_embed_dim)
        
        # Expand z_i to match batch size
        batch_size = ts_x.size(0)
        z_i_expanded = z_i_graph.unsqueeze(0).expand(batch_size, -1) # (batch_size, gcn_embed_dim)
        
        # 3. Late Fusion Concatenation
        combined = torch.cat([h_last_ts, z_i_expanded], dim=-1)
        
        # 4. Final Prediction via MLP
        out = F.relu(self.fc1(combined))
        pred = self.fc2(out)
        
        return pred