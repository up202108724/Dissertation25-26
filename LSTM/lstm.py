import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import TimeSeriesDataset

class LSTMForecaster(nn.Module):
    """LSTM model for recursive autoregressive forecasting.
    
    Predicts one step ahead: [t] -> [t+1]
    This model is trained on single-step predictions and used recursively during inference.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2, 
                 forecast_horizon: int = None, dropout: float = 0.2):
        """
        Parameters
        ----------
        input_size : int
            Number of input features (should be 1 for univariate autoregressive)
        hidden_size : int
            LSTM hidden size
        num_layers : int
            Number of LSTM layers
        forecast_horizon : int, optional
            Ignored in autoregressive mode (included for compatibility)
        dropout : float
            Dropout rate
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, 1, input_size) for autoregressive mode
        
        Returns
        -------
        torch.Tensor
            Shape: (batch_size,) - single next value prediction per sample
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)
        
        # Predict next single value
        output = self.fc(last_hidden).squeeze(-1)  # Shape: (batch_size,)
        
        return output
