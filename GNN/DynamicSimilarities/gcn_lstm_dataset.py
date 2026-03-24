import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import networkx as nx
import time
from collections import defaultdict
import random

class GraphBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 1. Agrupar os índices das amostras pelo seu graph_idx
        self.graph_to_indices = defaultdict(list)
        
        for idx in range(len(dataset)):
            # No teu SingleItemGraphDataset as amostras estão gravadas como (t, graph_idx)
            _, graph_idx = dataset.samples[idx] 
            self.graph_to_indices[graph_idx].append(idx)
            
    def __iter__(self):
        batches = []
        
        # 2. Para cada grafo, dividir as suas amostras em batches
        for graph_idx, indices in self.graph_to_indices.items():
            if self.shuffle:
                random.shuffle(indices) # Baralhar as amostras dentro do mesmo grafo
                
            # Criar os blocos de tamanho "batch_size" para este graph_idx
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                batches.append(batch)
                
        # 3. Baralhar a ordem dos lotes (batches) no treino
        # Isto garante que o modelo não treina todas as janelas de Janeiro e só depois as de Fevereiro
        if self.shuffle:
            random.shuffle(batches)
            
        return iter(batches)
        
    def __len__(self):
        # Retorna o número total de batches
        total_batches = 0
        for indices in self.graph_to_indices.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches

class SingleItemGraphDataset(Dataset):
    def __init__(self, data, exog_data, seq_length, dates, window_info, item_node_idx):
        self.data = np.array(data)
        self.exog_data = np.array(exog_data) if exog_data is not None else None
        self.seq_length = seq_length
        self.dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        self.item_node_idx = item_node_idx
        
        self.graph_end_dates = pd.to_datetime([w["end_date"] for w in window_info])
        self.samples = []
        
        # Start at exactly seq_length so t - seq_length is 0 for the first item
        for t in range(seq_length, len(data) - 1): # horizon = 1
            last_observed_date = self.dates[t-1]
            valid_graph_idx = np.where(self.graph_end_dates <= last_observed_date)[0]
            if len(valid_graph_idx) == 0:
                continue
            
            graph_idx = valid_graph_idx[-1]
            self.samples.append((t, graph_idx))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        t, graph_idx = self.samples[idx]
        
        # Now t starts at exactly seq_length. 
        # For target history: [0 : seq_length]
        seq_univariate = self.data[t - self.seq_length : t]
        
        if self.exog_data is not None:
            # For exog history shifted 1 step forward: [1 : seq_length + 1]
            exog_seq = self.exog_data[t - self.seq_length + 1 : t + 1]
            seq = np.column_stack((seq_univariate, exog_seq))
        else:
            seq = seq_univariate.reshape(-1, 1)
            
        y = self.data[t]
        
        return {
            "ts_x": torch.tensor(seq, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "graph_idx": torch.tensor(graph_idx, dtype=torch.long),
            "target_node_idx": torch.tensor(self.item_node_idx, dtype=torch.long)
        }

def get_adj_matrix(adj_list):
    # adj_list is dict {node_id: [(neighbor, weight, sim), ...]}
    nodes = list(adj_list.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.float32)
    for u, edges in adj_list.items():
        if u not in node_to_idx:
            continue
        u_idx = node_to_idx[u]
        for v, weight, sim in edges:
            if v in node_to_idx:
                v_idx = node_to_idx[v]
                adj[u_idx, v_idx] = weight
    return adj