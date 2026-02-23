import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length):
        self.seq_length = seq_length

        # Convert once
        self.target_data = torch.as_tensor(target_data, dtype=torch.float32)

        if exog_data is not None:
            self.exog_data = torch.as_tensor(exog_data, dtype=torch.float32)
            self.has_exog = True

            if len(self.exog_data) != len(self.target_data):
                raise ValueError("target_data and exog_data must have the same length")
        else:
            self.exog_data = None
            self.has_exog = False

    def __len__(self):
        return len(self.target_data) - self.seq_length

    def __getitem__(self, idx):
        # target_seq: (seq_length,)
        target_seq = self.target_data[idx:idx + self.seq_length]
        y = self.target_data[idx + self.seq_length]  # scalar tensor

        if self.has_exog:
            # exog_seq: (seq_length, n_exog)
            exog_seq = self.exog_data[idx:idx + self.seq_length]

            # target_seq.unsqueeze(1): (seq_length, 1)
            # x: (seq_length, 1 + n_exog)
            x = torch.cat([target_seq.unsqueeze(1), exog_seq], dim=1)
        else:
            x = target_seq.unsqueeze(1)  # (seq_length, 1)

        return x, y.unsqueeze(0)  # y shape: (1,)