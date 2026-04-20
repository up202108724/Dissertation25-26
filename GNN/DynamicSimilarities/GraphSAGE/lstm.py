import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

# --- Classe de Exemplo do Extrator de Features GraphSAGE ---
# Você precisará definir esta classe ou importar uma semelhante.
# Ela usa SAGEConv para processar o grafo de forma indutiva.
class SimpleSAGEFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        # Primeira camada
        self.convs.append(SAGEConv(in_channels, hidden_channels, project=True)) # 'project=True' para agregação indutiva
        # Camadas ocultas (se houver)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, project=True))
        # Última camada
        self.convs.append(SAGEConv(hidden_channels if num_layers > 1 else in_channels, out_channels, project=True))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        # Propagação pelas arestas e agregação dos vizinhos
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

# --- Definição da Classe SAGE_LSTM adaptada ---
class SAGE_LSTM(nn.Module):
    def __init__(self, sage_in_channels=1, sage_hidden=16, sage_out=32, sage_num_layers=1,
                 exog_size=0, lstm_hidden=50, num_layers=1, dropout=0.2):
        """
        Arquitetura GNN-LSTM Híbrida: spatial extractions por GraphSAGE (SAGEConv) e temporal state por LSTM.
        Adaptada da classe GAT_LSTM para usar embeddings do GraphSAGE.
        """
        super().__init__()
        
        # 1. Extrator Espacial (GraphSAGE com 1 ou mais Camadas)
        # O SAGEConv é indutivo e aprende funções de agregação que podem ser
        # aplicadas a grafos totalmente novos.
        self.sage = SimpleSAGEFeatureExtractor(
            in_channels=sage_in_channels,
            hidden_channels=sage_hidden,
            out_channels=sage_out,
            num_layers=sage_num_layers,
            dropout=dropout
        )
        
        # 2. Dimensão de Entrada da LSTM (Output do Nó Central + Exógenas)
        # O output do GraphSAGE agora alimenta a LSTM.
        lstm_input_size = sage_out + exog_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        
        # 3. Cabeça de Previsão (MLP Final) - Inalterada
        self.linear = nn.Linear(lstm_hidden, 1)

    def forward(self, graphs_seq, exog_features=None):
        """
        graphs_seq: Opcionalmente uma lista de len = seq_len onde cada elemento é um batch do PyTorch Geometric. 
                    Em cada t da sequência de entrada da LSTM, corremos 1 vez a rede espacial SAGE.
        exog_features: [batch_size, seq_len, exog_features_size]
        """
        seq_embeddings = []
        
        # Iterar no tempo para fazer a "ponte" Espacial -> Temporal
        for t in range(len(graphs_seq)):
            graph_t = graphs_seq[t]
            
            # Parte Espacial: Propagação pelas arestas e Vizinhos com a SAGEConv
            # O SAGEConv agrega as características dos vizinhos e gera o embedding estrutural.
            node_embeddings = self.sage(graph_t.x, graph_t.edge_index)
            
            # Necessitamos extrair APENAS o vetor do NÓ CENTRAL para fornecer à LSTM
            # graph_t.target_mask deve ser uma máscara bool/índice mapeando quem é o nó central prevísivel no batch
            target_embs_t = node_embeddings[graph_t.target_mask]
            
            seq_embeddings.append(target_embs_t)
            
        # Converter para formato (Batch Size, Sequence Length, Embedding Size) 
        # lstm_in é de tamanho [batch_size, seq_len, sage_out]
        lstm_in = torch.stack(seq_embeddings, dim=1) 
        
        # Concatenação com as variáveis exógenas se fornecidas - Inalterada
        if exog_features is not None:
             lstm_in = torch.cat([lstm_in, exog_features], dim=-1)
             
        # Parte Temporal - Inalterada
        lstm_out, _ = self.lstm(lstm_in)
        last = self.drop(lstm_out[:, -1, :])
        
        # Parte Final (Regressão para Vendas) - Inalterada
        return self.linear(last)