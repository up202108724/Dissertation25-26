import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length, embeddings=None, graph_window_size=7):
        """
        Dataset for time series with optional exogenous variables and graph embeddings.
        
        Args:
            target_data: Scaled target variable data
            exog_data: Scaled exogenous variables data (can be None)
            seq_length: Length of input sequences
            embeddings: Node embeddings data (can be None). If you pass the embeddings 
                        of the sliding windows, this class will automatically pad the 
                        first `graph_window_size` days with zeros, ensuring the embedding 
                        generated from days 0..(graph_window_size-1) is only active starting 
                        on day `graph_window_size`.
            graph_window_size: The window size used to construct the graphs.
        """
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.has_exog = exog_data is not None
        self.has_embeddings = embeddings is not None

        if self.has_embeddings:
            # If the embeddings array only contains the valid sliding windows (e.g. N - graph_window_size)
            # We zero-pad the first `graph_window_size` days so that embeddings[t] represents 
            # the graph from the graph_window_size days prior to t.
            if len(embeddings) < len(target_data):
                emb_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
                zero_pad = np.zeros((graph_window_size, emb_dim))
                self.embeddings = np.vstack([zero_pad, embeddings])
            else:
                self.embeddings = embeddings
        else:
            self.embeddings = None
    
    def __len__(self):
        return len(self.target_data) - self.seq_length
    
    def __getitem__(self, idx):
        # Target sequence and label
        target_seq = self.target_data[idx:idx+self.seq_length]
        y = self.target_data[idx+self.seq_length]
        
        if self.has_exog:
            # Combine target with exogenous variables
            exog_seq = self.exog_data[idx+1:idx+self.seq_length+1]  # Align exog with target sequence
            # Stack target and exog features: shape (seq_length, 1 + n_exog)
            x = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x = target_seq.reshape(-1, 1)
            
        if self.has_embeddings:
            # Fetch embeddings corresponding to the sequence days
            # Because of the padding in __init__, emb_seq[i] is the graph of the graph_window_size days before day (idx + 1 + i)
            emb_seq = self.embeddings[idx:idx+self.seq_length] 
            # Stack the target/exog features with the embeddings: shape (seq_length, 1 + n_exog? + emb_dim)
            x = np.column_stack([x, emb_seq])
            
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor([y])
        
        return x_tensor, y_tensor
