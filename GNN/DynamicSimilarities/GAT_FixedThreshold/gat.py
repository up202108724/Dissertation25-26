import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.2):
        """
        Extrator de Features Espaciais com 2 Camadas GAT.
        A primeira camada permite a partilha de "mensagens" vizinho-vizinho, 
        e a segunda agrega-as para produzir o embedding final.
        """
        super().__init__()
        self.dropout = dropout
        
        # Camada 1: Expande as features e aplica Múltiplas Cabeças de Atenção (Multi-Head)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Camada 2: Agrega as heads no embedding final.
        # concat=False aplica a média entre as heads para estabilizar e obter a dimensão exata 'out_channels'.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index):
        # Passagem pela 1ª Camada GAT
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        
        # Passagem pela 2ª Camada GAT
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Retorna o tensor `x` de *todos* os nós atualizados pela vizinhança
        return x
