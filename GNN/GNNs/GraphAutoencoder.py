import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import torch

# 2. Define a Graph Autoencoder wrapper around your existing GCN
class GraphAutoencoder(nn.Module):
    def __init__(self, encoder):
        super(GraphAutoencoder, self).__init__()
        self.encoder = encoder

    def forward(self, x, adj):
        # We skip the specific log_softmax in your original GCN
        # and use the raw outputs of the layers for our embeddings
        hidden = F.relu(self.encoder.gc1(x, adj))
        z = self.encoder.gc2(hidden, adj) # 'z' are the node embeddings
        
        # Reconstruct the adjacency matrix via inner product (Decoder)
        adj_reconstructed = torch.matmul(z, z.t())
        return z, adj_reconstructed