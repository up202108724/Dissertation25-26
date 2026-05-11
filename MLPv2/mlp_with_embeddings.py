from torch import nn
import torch

class MLPForecasterWithCategoricalEmbeddings(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_continuous_features: int,
        embedding_sizes: dict, # e.g., {'day_of_week': (7, 4), 'month': (12, 4)}
        horizon: int,
        out_dim: int = 1,      # often 1 for a single target variable
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.out_dim = out_dim

        act_dict = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act = act_dict.get(activation.lower(), nn.ReLU)

        # Create ModuleDict for categorical embeddings
        self.embeddings = nn.ModuleDict({
            cat_name: nn.Embedding(num_embeddings=num_cats, embedding_dim=emb_dim)
            for cat_name, (num_cats, emb_dim) in embedding_sizes.items()
        })
        
        # Calculate the total feature dimension per time step
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_sizes.values())
        features_per_step = n_continuous_features + total_emb_dim
        
        layers = []
        prev = lookback * features_per_step
        
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
            
        layers += [nn.Linear(prev, horizon * out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        # x_cont: (B, L, n_continuous_features)
        # x_cat: dict of categorical tensors, each (B, L)
        b = x_cont.size(0)
        
        embs = []
        for cat_name, emb_layer in self.embeddings.items():
            cat_tensor = x_cat[cat_name].long()
            emb = emb_layer(cat_tensor) # (B, L, emb_dim)
            embs.append(emb)
            
        if embs:
            x_cat_combined = torch.cat(embs, dim=-1) # (B, L, total_emb_dim)
            x_combined = torch.cat([x_cont, x_cat_combined], dim=-1) # (B, L, total_features)
        else:
            x_combined = x_cont
            
        x_flat = x_combined.reshape(b, -1)  # flatten temporal dimension
        y = self.net(x_flat)    # (B, H*out_dim)
        
        return y.view(b, self.horizon, self.out_dim)