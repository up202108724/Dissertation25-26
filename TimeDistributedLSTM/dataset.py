import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length):
        """
        Dataset for time series with optional exogenous variables.
        
        Args:
            target_data: Scaled target variable data
            exog_data: Scaled exogenous variables data (can be None)
            seq_length: Length of input sequences
        """
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.has_exog = exog_data is not None
    
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
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

# -----------------------------------------------------------------------------
# Dataset: returns per-step targets so the time-distributed head can be
# supervised at every step. The seq-to-one head simply uses y_seq[:, -1].
# -----------------------------------------------------------------------------
class SeqTargetDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length):
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.has_exog = exog_data is not None

    def __len__(self):
        return len(self.target_data) - self.seq_length

    def __getitem__(self, idx):
        target_seq = self.target_data[idx:idx + self.seq_length]
        # Each position k predicts the next value -> aligned with the exog at k.
        y_seq = self.target_data[idx + 1:idx + self.seq_length + 1]

        if self.has_exog:
            exog_seq = self.exog_data[idx + 1:idx + self.seq_length + 1]
            x = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x = target_seq.reshape(-1, 1)

        return torch.FloatTensor(x), torch.FloatTensor(y_seq.reshape(-1, 1))