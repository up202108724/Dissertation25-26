import torch
import torch.nn as nn
from GraphSAGE import GraphSAGEEncoder

class GraphSAGELSTM(nn.Module):
    def __init__(self, sage_in_channels, sage_hidden, sage_out, 
                 lstm_exog_features, lstm_hidden, lstm_layers, lstm_dropout=0.0):
        super(GraphSAGELSTM, self).__init__()
        
        # Spatial Encoder (GNN)
        self.graph_encoder = GraphSAGEEncoder(sage_in_channels, sage_hidden, sage_out)
        
        # Temporal Encoder (LSTM)
        # Sequence input consists of: Target(1) + Exogs + GraphSAGE embedding
        lstm_input_size = 1 + lstm_exog_features + sage_out 
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden, 
                            num_layers=lstm_layers, 
                            batch_first=True, 
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0)
        
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, batch_x, pyg_graph_batch, target_indices_map):
        """
        batch_x: [batch, seq_len, temporal_features]
        pyg_graph_batch: PyG Batch containing (batch * seq_len) graphs
        target_indices_map: List of tuples (graph_idx, local_node_idx) tracking target node indices globally
        """
        B, seq_len, _ = batch_x.shape
        
        # 1. Forward pass all graphs through GraphSAGE jointly (High Efficiency)
        all_node_embeddings = self.graph_encoder(pyg_graph_batch.x, pyg_graph_batch.edge_index)
        
        # 2. Extract ONLY the target product's embedded vector for each network step
        # pyg_graph_batch.ptr defines boundaries for each sub-graph in the mega-batch
        ptr = pyg_graph_batch.ptr
        extracted_embeddings = []
        
        for graph_idx, local_idx in target_indices_map:
            global_idx = ptr[graph_idx] + local_idx
            extracted_embeddings.append(all_node_embeddings[global_idx])
            
        # Reshape isolated embeddings back to temporal sequence shape: [Batch, Seq_len, Sage_out]
        extracted_embeddings = torch.stack(extracted_embeddings).view(B, seq_len, -1)
        
        # 3. Concatenate temporal variables and dynamic spatial embeddings
        lstm_input = torch.cat([batch_x, extracted_embeddings], dim=-1)
        
        # 4. Standard LSTM forward logic
        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :] 
        return self.fc(last_out)