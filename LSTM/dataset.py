import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
class TimeSeriesDataset(Dataset):
    """Dataset for LSTM time series forecasting with fixed lookback window.
    
    Creates sequences: [t-lookback:t] -> [t+1]
    """
    
    def __init__(self, data: pd.DataFrame, lookback: int = 30, forecast_horizon: int = None, 
                 store_id: int = None, item_id: int = None,
                 include_features: List[str] = None):

        # Filter data if store/item specified
        if store_id is not None and item_id is not None:
            data = data[(data['store_id'] == store_id) & (data['item_id'] == item_id)].copy()
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Extract target (sales values) - univariate only
        self.values = data['value'].values
        self.lookback = lookback
        
        # Create fixed-length sequences: [t-lookback:t] -> [t+1]
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self):
        
        X, y = [], []
        
        # Start from lookback position (need at least lookback points)
        for i in range(self.lookback, len(self.values)):
            x_seq = self.values[i-self.lookback:i]  # Fixed lookback window
            y_val = self.values[i]                   # Next value to predict
            X.append(x_seq)
            y.append(y_val)
        
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        logger.info(f"Sequence creation complete:")
        logger.info(f"  Total time series length: {len(self.values)}")
        logger.info(f"  Lookback window: {self.lookback}")
        logger.info(f"  Sequences created: {len(X_array)}")
        logger.info(f"  X shape: {X_array.shape} (num_sequences, lookback)")
        logger.info(f"  y shape: {y_array.shape} (num_sequences,)")
        
        return X_array, y_array
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """Return fixed-length sequence and target value"""
        x_seq = self.X[idx]  # Shape: (lookback,)
        y_val = self.y[idx]  # Shape: scalar
        
        # Convert to tensors and reshape for LSTM
        x_tensor = torch.FloatTensor(x_seq).unsqueeze(-1)  # (lookback,) -> (lookback, 1)
        y_tensor = torch.FloatTensor([y_val])  # (1,)
        
        return x_tensor, y_tensor





