import torch
import torch.nn as nn
from gat import GATFeatureExtractor

class GAT_LSTM(nn.Module):
    def __init__(self, gat_in_channels=1, gat_hidden=16, gat_out=32, gat_heads=2,
                 exog_size=0, lstm_hidden=50, num_layers=1, dropout=0.2):
        """
        Arquitetura GNN-LSTM Híbrida: spatial extractions por GAT e temporal state por LSTM.
        """
        super().__init__()
        
        # 1. Extrator Espacial (2 Camadas de GAT)
        self.gat = GATFeatureExtractor(
            in_channels=gat_in_channels,
            hidden_channels=gat_hidden,
            out_channels=gat_out,
            heads=gat_heads,
            dropout=dropout
        )
        
        # 2. Dimensão de Entrada da LSTM (Output do Nó Central + Exógenas)
        lstm_input_size = gat_out + exog_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        
        # 3. Cabeça de Previsão (MLP Final)
        self.linear = nn.Linear(lstm_hidden, 1)

    def forward(self, graphs_seq, exog_features=None):
        """
        graphs_seq: Opcionalmente uma lista de len = seq_len onde cada elemento é um batch do PyTorch Geometric. 
                    Em cada t da sequência de entrada da LSTM, corremos 1 vez a rede espacial GAT.
        exog_features: [batch_size, seq_len, exog_features_size]
        """
        seq_embeddings = []
        
        # Iterar no tempo para fazer a "ponte" Espacial -> Temporal
        # Supondo que graphs_seq é uma lista T onde array de entrada tem Dataloader PyG "Batch"
        for t in range(len(graphs_seq)):
            graph_t = graphs_seq[t]
            
            # Parte Espacial: Propagação pelas arestas e Vizinhos com a GAT
            # graph_t.x é o tensor [num_nos_no_batch, features_janela]
            # graph_t.edge_index são as conexões naquele dia
            node_embeddings = self.gat(graph_t.x, graph_t.edge_index)
            
            # Necessitamos extrair APENAS o vetor do NÓ CENTRAL para fornecer à LSTM
            # graph_t.target_mask deve ser uma máscara bool/índice mapeando quem é o nó central prevísivel no batch
            # ou assumimos um tensor em que iteram os id's dos produtos target
            # Ex: targets [node_id1_batch, node_id2_batch, ...] 
            target_embs_t = node_embeddings[graph_t.target_mask]
            
            seq_embeddings.append(target_embs_t)
            
        # Converter para formato (Batch Size, Sequence Length, Embedding Size)    
        # target_embs_t é de tamanho [batch_size, gat_out]
        lstm_in = torch.stack(seq_embeddings, dim=1) 
        
        # Concatenação com as variáveis exógenas se fornecidas
        if exog_features is not None:
             lstm_in = torch.cat([lstm_in, exog_features], dim=-1)
             
        # Parte Temporal
        lstm_out, _ = self.lstm(lstm_in)
        last = self.drop(lstm_out[:, -1, :])
        
        # Parte Final (Regressão para Vendas)
        return self.linear(last)