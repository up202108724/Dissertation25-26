import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Dataset for recursive autoregressive LSTM time series forecasting.
    
    Uses all available history up to each point:
    [all_history_from_start_to_t] -> [t+1]
    """
    
    def __init__(self, data: pd.DataFrame, lookback: int = None, forecast_horizon: int = None, 
                 store_id: int = None, item_id: int = None,
                 include_features: List[str] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns: date, store_id, item_id, value, and optional features
        lookback : int, optional
            Ignored in this mode (all history is used)
        forecast_horizon : int, optional
            Ignored in this mode
        store_id, item_id : int, optional
            If provided, filter to specific store-item pair
        include_features : List[str], optional
            Not used in univariate mode
        """
        # Filter data if store/item specified
        if store_id is not None and item_id is not None:
            data = data[(data['store_id'] == store_id) & (data['item_id'] == item_id)].copy()
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Extract target (sales values) - univariate only
        self.values = data['value'].values
        
        # Create variable-length sequences: all history up to t -> t+1
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self):
        """Create pairs where input is all history from start to t, output is t+1"""
        X, y = [], []
        
        # For each point, use all history from start to that point
        for i in range(1, len(self.values)):
            x_seq = self.values[:i]  # All history from start to current point
            y_val = self.values[i]    # Next value to predict
            
            X.append(x_seq)
            y.append(y_val)
        
        return X, np.array(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """Return variable-length sequence and its target value"""
        x_seq = self.X[idx]  # Variable length array
        y_val = self.y[idx]
        return torch.FloatTensor(x_seq), torch.FloatTensor([y_val])





