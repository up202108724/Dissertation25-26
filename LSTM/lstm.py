import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import TimeSeriesDataset

class LSTMForecaster(nn.Module):
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, 
                 forecast_horizon: int = None, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer: predict single next value
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out).squeeze(-1)  
        
        return output
