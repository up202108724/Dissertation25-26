import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AGCRNDataset(Dataset):
    def __init__(self, df, seq_len=30, horizon=1, target_col='sales', feature_cols=None):
        """
        df: Pandas dataframe containing all 61 products. Must be sorted by time!
        Assumes columns: ['date', 'product_id', target_col] + feature_cols
        """
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 1. Pivot the dataframe to align all products chronologically
        # Shape: (Total_Timesteps, Num_Nodes)
        target_df = df.pivot(index='date', columns='product_id', values=target_col).fillna(0)
        self.y_data = target_df.values
        
        # 2. Handle exogenous features (if any, e.g., promotions, day-of-week)
        self.features = feature_cols if feature_cols else [target_col]
        self.X_data = np.stack([
            df.pivot(index='date', columns='product_id', values=f).fillna(0).values 
            for f in self.features
        ], axis=-1) # Shape: (Total_Timesteps, Num_Nodes, Num_Features)
        
        self.num_samples = len(self.X_data) - seq_len - horizon + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: (Seq_Len, Num_Nodes, Features)
        X = self.X_data[idx : idx + self.seq_len]
        
        # Target: (Horizon, Num_Nodes, 1)
        Y = self.y_data[idx + self.seq_len : idx + self.seq_len + self.horizon]
        Y = np.expand_dims(Y, axis=-1) if Y.ndim == 2 else Y
        
        return torch.FloatTensor(X), torch.FloatTensor(Y)

def create_agcrn_dataloaders(train_df, val_df, test_df, feature_cols, seq_len=30, horizon=1, batch_size=64):
    """Creates the train, val, and test DataLoaders."""
    
    
    
    train_dataset = AGCRNDataset(train_df, seq_len, horizon, 'sales', feature_cols)
    val_dataset = AGCRNDataset(val_df, seq_len, horizon, 'sales', feature_cols)
    test_dataset = AGCRNDataset(test_df, seq_len, horizon, 'sales', feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader