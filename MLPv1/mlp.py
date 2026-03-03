from torch import nn

class MLP(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.flatten_dim = seq_length * input_size
        
        layers = []
        # Input layer
        layers.append(nn.Linear(self.flatten_dim, hidden_size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)
        # Flatten input: (batch, seq_len * input_size)
        x_flat = x.view(batch_size, -1)
        return self.network(x_flat)
