import pandas as pd
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

class DynamicGraphTimeSeriesDataset(Dataset):
    def __init__(
        self, 
        target_data, 
        exog_data, 
        date_data,
        seq_length, 
        dynamic_graphs, 
        dynamic_features, 
        graph_window_info
    ):
        """
        Dataset for time series that intelligently fetches the most recent valid graph context.
        
        Args:
            target_data: Target variable data (e.g. sales)
            exog_data: Exogenous variables data (can be None)
            date_data: Array or series of dates corresponding to each time step
            seq_length: Length of LSTM input sequences (e.g. 28)
            dynamic_graphs: List of adjacency matrices/edge_indices for each graph window
            dynamic_features: List of computed node feature tensors for each graph window
            graph_window_info: List of dicts with 'start_date' and 'end_date' for each window
        """
        self.target_data = target_data
        self.exog_data = exog_data
        self.date_data = pd.to_datetime(date_data)  # Ensure datetime format for exact comparisons
        self.seq_length = seq_length
        self.has_exog = exog_data is not None
        
        self.dynamic_graphs = dynamic_graphs
        self.dynamic_features = dynamic_features
        self.graph_window_info = graph_window_info
        
        # Pre-process window dates for rapid lookup during __getitem__
        self.window_end_dates = pd.to_datetime([info['end_date'] for info in graph_window_info])
        
    def __len__(self):
        return len(self.target_data) - self.seq_length
    
    def _find_latest_valid_graph_index(self, target_date):
        """
        Finds the index of the most recent graph that was completed on or before `target_date`.
        """
        # Find all windows that ended before or on the current target date
        valid_indices = np.where(self.window_end_dates <= target_date)[0]
        
        if len(valid_indices) == 0:
            # If we ask for a date so early that NO graph window has finished yet,
            # fallback to the very first available graph (or handle as 0 padding)
            return 0
            
        # Return the index of the most recently finished window
        return valid_indices[-1]
    
    def __getitem__(self, idx):
        # 1. Temporal sequence preparation (Standard LSTM logic)
        target_seq = self.target_data[idx : idx + self.seq_length]
        
        # The exact day we are predicting sales for
        forecast_horizon_idx = idx + self.seq_length
        y = self.target_data[forecast_horizon_idx]
        
        # The last day the LSTM has access to (the day before the prediction)
        last_observed_date = self.date_data.iloc[forecast_horizon_idx - 1] if isinstance(self.date_data, pd.Series) else self.date_data[forecast_horizon_idx - 1]
        
        # Compile temporal features X
        if self.has_exog:
            exog_seq = self.exog_data[idx + 1 : forecast_horizon_idx + 1] 
            x_ts = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x_ts = target_seq.reshape(-1, 1)
            
        # 2. Structural Dynamic Graph preparation
        # Find which graph snapshot was structurally valid as of `last_observed_date`
        graph_idx = self._find_latest_valid_graph_index(last_observed_date)
        
        # Extract the correctly synched graph matrices
        graph_adj = self.dynamic_graphs[graph_idx]
        graph_x = self.dynamic_features[graph_idx]
        
        return (
            torch.FloatTensor(x_ts), 
            torch.FloatTensor([y]),
            graph_adj,  # The adjacency structure valid for this exact timestamp
            graph_x     # The node features valid for this exact timestamp
        )
