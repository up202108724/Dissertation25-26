import torch
from torch.utils.data import Dataset
import numpy as np

# =============================================================================
# Dataset: multivariate sequences with per-step targets
# =============================================================================
class MultivariateSeqDataset(Dataset):
    """Sliding-window dataset over a (T, K) matrix of K aligned series.

    Each window of length ``seq_length`` predicts the *next* value of every
    series, so position k is supervised by step k+1 (aligned with the exog at
    k+1, as in the univariate ``SeqTargetDataset``).
    """

    def __init__(self, series_data, exog_data, seq_length):
        self.series_data = series_data          # (T, K) scaled
        self.exog_data = exog_data              # (T, E) scaled or None
        self.seq_length = seq_length
        self.has_exog = exog_data is not None

    def __len__(self):
        return len(self.series_data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.series_data[idx:idx + self.seq_length]                 # (seq, K)
        y_seq = self.series_data[idx + 1:idx + self.seq_length + 1]       # (seq, K)

        if self.has_exog:
            exog_seq = self.exog_data[idx + 1:idx + self.seq_length + 1]  # (seq, E)
            x = np.column_stack([seq, exog_seq])
        else:
            x = seq

        return torch.FloatTensor(x), torch.FloatTensor(y_seq)