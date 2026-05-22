import torch
import torch.nn as nn
from torch_geometric.data import Batch

from gnn_pyg import SimpleGNNEncoder
from mlp import MLPForecaster

class SimpleGNN_MLP_Forecaster(nn.Module):
    """
    End-To-End Forecasting Model combining a Simple GNN (2-layer GCN) for
    spatial features and an MLP for sequence forecasting.
    """
    def __init__(
        self,
        lookback: int,
        ts_dim: int,
        cal_dim: int,
        gat_in_channels: int,
        gat_hidden_channels: int,
        gat_out_channels: int,
        horizon: int,
        mlp_hidden_sizes=(64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lookback = lookback
        
        # Simple GCN Encoder to process Ego-Graphs
        self.gat = SimpleGNNEncoder(
            in_channels=gat_in_channels, 
            hidden_channels=gat_hidden_channels, 
            out_channels=gat_out_channels,
            dropout=dropout
        )
        
        # Concatenated dimension per timestep: Time Series (T) + Calendar (C) + Central Node Embedding (Z)
        self.concat_dim = ts_dim + cal_dim + gat_out_channels
        
        # MLP Forecaster for sequence prediction
        self.mlp = MLPForecaster(
            lookback=lookback,
            in_channels=self.concat_dim,
            horizon=horizon,
            out_dim=1,
            hidden_sizes=mlp_hidden_sizes,
            dropout=dropout
        )

    def forward(self, ts_seq, cal_seq, pyg_batch, target_node_indices):
        """
        Forward pass for the End-to-End model.
        
        Args:
            ts_seq: Tensor of shape (B, L, ts_dim) representing historical time-series
            cal_seq: Tensor of shape (B, L, cal_dim) representing calendar features
            pyg_batch: A PyG Batch object containing B x L subgraphs batched together
            target_node_indices: Tensor of shape (B*L) indicating the index of the central 
                                 target node within each subgraph in the batch.
                                 
        Returns:
            predictions: Tensor of shape (B, horizon, 1)
        """
        B, L, _ = ts_seq.shape
        
        # 1. Process all graphs through Simple GCN
        # Pass edge_attr if present
        ea = pyg_batch.edge_attr if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.shape[0] > 0) else None
        # node_embeddings shape: (total_nodes_in_batch, gat_out_channels)
        node_embeddings = self.gat(pyg_batch.x, pyg_batch.edge_index, edge_attr=ea)
        
        # 2. Extract strictly the central node's embedding for each graph in the sequence
        z_target = node_embeddings[target_node_indices] # shape: (B*L, gat_out_channels)
        
        # Reshape to sequence format (B, L, gat_out_channels)
        z_target = z_target.view(B, L, -1)
        
        # 3. Concatenate T, C, and z_target (Spatial-Temporal fusion)
        # Sequence shape: (B, L, concat_dim)
        combined_seq = torch.cat([ts_seq, cal_seq, z_target], dim=-1)
        
        # 4. Pass through MLP for sequence mapping -> horizon prediction
        predictions = self.mlp(combined_seq)
        
        return predictions

