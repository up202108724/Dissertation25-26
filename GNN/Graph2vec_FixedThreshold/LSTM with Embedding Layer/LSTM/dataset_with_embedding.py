import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesEmbIdxDataset(Dataset):
    """
    Time-series dataset where graph embeddings are represented as INTEGER
    INDICES into a trainable nn.Embedding table, rather than raw float vectors.

    Each sample returns a 3-tuple:
        x_ts      : (seq_length, 1 + n_exog)   float32  — target + exog features
        emb_idx   : (seq_length,)               int64    — absolute time indices
                                                           into the embedding table
        y         : (1,)                        float32  — next-step target value

    The absolute index for dataset sample ``k`` at window position ``j`` is:
        emb_idx[j] = idx_offset + k + j

    where ``idx_offset`` is the absolute start of this split in the full
    time series (e.g. ``train_start_idx`` for the training set,
    ``val_start_idx`` for the validation set).

    Parameters
    ----------
    target_data    : (T,)           scaled target values for this split
    exog_data      : (T, n_exog) or None   scaled exogenous features
    seq_length     : int            lookback window length
    idx_offset     : int            absolute position in the full time series
                                    where this split starts
    emb_table_size : int            total size of the embedding table (used
                                    to clamp out-of-bounds indices safely)
    """

    def __init__(
        self,
        target_data: np.ndarray,
        exog_data,
        seq_length: int,
        idx_offset: int = 0,
        emb_table_size: int = None,
    ):
        self.target_data    = target_data
        self.exog_data      = exog_data
        self.seq_length     = seq_length
        self.idx_offset     = idx_offset
        self.emb_table_size = emb_table_size
        self.has_exog       = exog_data is not None

    def __len__(self):
        return len(self.target_data) - self.seq_length

    def __getitem__(self, idx):
        target_seq = self.target_data[idx : idx + self.seq_length]
        y          = self.target_data[idx + self.seq_length]

        if self.has_exog:
            # Shift exog forward by 1 to match training convention used in
            # the rest of the codebase (exog at t+1 predicts target at t+1)
            exog_seq = self.exog_data[idx + 1 : idx + self.seq_length + 1]
            x_ts     = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x_ts = target_seq.reshape(-1, 1)

        # Absolute indices in the embedding table
        emb_idx = np.arange(
            self.idx_offset + idx,
            self.idx_offset + idx + self.seq_length,
            dtype=np.int64,
        )
        if self.emb_table_size is not None:
            emb_idx = np.clip(emb_idx, 0, self.emb_table_size - 1)

        return (
            torch.FloatTensor(x_ts),    # (L, 1 + n_exog)
            torch.LongTensor(emb_idx),  # (L,)
            torch.FloatTensor([y]),     # (1,)
        )
