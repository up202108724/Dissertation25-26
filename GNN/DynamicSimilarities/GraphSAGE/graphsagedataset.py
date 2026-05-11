import torch
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.data import Batch

class JointSAGETimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length, pyg_graphs_list, target_item_name):
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.pyg_graphs_list = pyg_graphs_list
        self.target_item_name = target_item_name

    def __len__(self):
        return len(self.target_data) - self.seq_length
    
    def __getitem__(self, idx):
        # 1. Extract Temporal Data (target + exog)
        target_seq = self.target_data[idx : idx + self.seq_length]
        y = self.target_data[idx + self.seq_length]
        
        if self.exog_data is not None:
            # Exogs align with target + offset by 1 for prediction features
            exog_seq = self.exog_data[idx + 1 : idx + self.seq_length + 1]
            x_seq = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x_seq = target_seq.reshape(-1, 1)
            
        x_tensor = torch.FloatTensor(x_seq)
        y_tensor = torch.FloatTensor([y])

        # 2. Extract the Sequence of PyG Graphs (30 graphs for the 30 sequence days)
        graphs_seq = self.pyg_graphs_list[idx : idx + self.seq_length]
        
        # Determine the target product's ID location inside each graph's node structure
        target_indices = []
        for g in graphs_seq:
            if hasattr(g, 'node_mapping') and self.target_item_name in g.node_mapping:
                target_indices.append(g.node_mapping[self.target_item_name])
            else:
                target_indices.append(0) # Fallback if node becomes disconnected

        return x_tensor, graphs_seq, target_indices, y_tensor

def joint_collate_fn(batch):
    x_seqs = []
    y_vals = []
    all_graphs = []
    all_target_indices = []
    
    graph_offset = 0 # Tracks graph order in the mega-batch
    
    for item in batch:
        x_tensor, graphs_seq, target_indices, y_tensor = item
        
        x_seqs.append(x_tensor)
        y_vals.append(y_tensor)
        
        for g, t_idx in zip(graphs_seq, target_indices):
            all_graphs.append(g)
            # Store tuple: (Which graph in the mega-batch, local index of target node)
            all_target_indices.append((graph_offset, t_idx))
            graph_offset += 1
            
    # Standard PyTorch batching for sequences
    batch_x = torch.stack(x_seqs) # Shape: [Batch_size, Seq_length, 1 + Exogs]
    batch_y = torch.stack(y_vals) # Shape: [Batch_size, 1]
    
    # Powerful PyG native batching for graphs
    batch_graphs = Batch.from_data_list(all_graphs)
    
    return batch_x, batch_graphs, all_target_indices, batch_y
