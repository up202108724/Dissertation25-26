import torch
import numpy as np
from torch import nn
import pandas as pd
import os
class TrainConfig:
    lookback: int = 30
    horizon: int = 153
    batch_size: int = 32
    train_size: int = 455
    val_size : int = 153
    lr: float = 1e-4
    epochs: int = 30
    weight_decay: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
class MLPForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        in_channels: int,
        horizon: int,
        out_dim: int = 1,                 # often 1 for a single target variable
        hidden_sizes=(64, 32),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lookback = lookback
        self.in_channels = in_channels
        self.horizon = horizon
        self.out_dim = out_dim

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = lookback * in_channels
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act(), nn.Dropout(dropout)]
            prev = hs
        layers += [nn.Linear(prev, horizon * out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C)
        b = x.size(0)
        x = x.reshape(b, -1)  # flatten - changed from .view to .reshape
        y = self.net(x)    # (B, H*out_dim)
        return y.view(b, self.horizon, self.out_dim)
    
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, L, C)
        self.y = torch.from_numpy(y)  # (N, H, C)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]  # (T, 1)

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    X = np.zeros((N, lookback, C), dtype=np.float32)
    y = np.zeros((N, horizon, C), dtype=np.float32)

    exog_indices = [idx for idx in range(C) if idx != target_channel]

    for i in range(N):
        X[i] = series[i : i + lookback].copy()
        
        # Shift exogenous variables forward by 1 so the model sees the target day's exog features
        # X[i] target corresponds to steps i ... i+lookback-1
        # X[i] exog corresponds to steps i+1 ... i+lookback
        if len(exog_indices) > 0:
            X_i = X[i]
            X_i[:, exog_indices] = series[i + 1 : i + lookback + 1, exog_indices]
            
        y[i] = series[i + lookback : i + lookback + horizon]

    return X, y

def train_mlp_forecaster(
    df: pd.DataFrame, 
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,    
    scaler,
    target_channel,  
    val_ratio,
    hidden_sizes,
    target_col,
    exog_cols,
    test_size
):

    
    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values
    
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - cfg.val_size
    train_end_idx = val_start_idx 
    
    train_data = data[:train_end_idx]
    

    # scaler is passed as argument
    train_scaled = train_data.copy()
    train_target = train_data[:, target_channel:target_channel+1]
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(train_target)
    
    val_context_start_idx = val_start_idx
    val_context_data = data[val_context_start_idx : val_end_idx]
    
    val_scaled = val_context_data.copy()
    val_target = val_context_data[:, target_channel:target_channel+1]
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(val_target)

    X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
    X_val, y_val_full = make_windows(val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)

    y_train = y_train_full[:, :, target_channel : target_channel + 1]
    y_val = y_val_full[:, :, target_channel : target_channel + 1]

    C_in = X_train.shape[2]
    
    print(f"Train/Val Split Indices:")
    print(f"  Train End: {train_end_idx}")
    print(f"  Val Range:     {val_start_idx} to {val_end_idx}")
    print(f"  Test Range:    {test_start_idx} to End")
    print(f"  Val Targets:   {y_val.shape[0]} windows created")
    print(f"Input Shape:  (Batch, {cfg.lookback}, {C_in})")
    print(f"Output Shape: (Batch, {cfg.horizon}, 1)")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
        
    model = MLPForecaster(
        lookback=cfg.lookback,
        in_channels=C_in,
        horizon=cfg.horizon,
        out_dim=1, 
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'mse', 'mae', or 'huber'.")

    best_val = float("inf")
    best_state = None
    
    train_losses = []
    val_losses = []

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f'{model_dir}/mlp_product_{product_id}.pth'
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                
        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_model_path)
                best_epoch = epoch
                print (f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} (New Best)")
        else:
            print("WARNING: Validation set is too small to form windows!")

    if hasattr(val_loader.dataset, "__len__") and len(val_loader.dataset) > 0:
        print(f"Best Val {loss_type.upper()}: {best_val:.6f} (Epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch

def recursive_inference_mlp(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,  
    future_exog: np.ndarray,     
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Recursively forecasts `horizon` steps using a 1-step-ahead model.
    Uses known `future_exog` for the next step's input.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    horizon = len(future_exog)
    
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    # Identificar índices exógenos de forma segura
    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Escalar o histórico recente (apenas o target, assumindo exogs já escaladas)
    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )
    
    preds_scaled = []
    model = model.to(device).eval()

    # Alinhar janela inicial com a mesma lógica do make_windows
    # X[t] = target[t], exog[t+1]
    input_window = current_x_scaled.copy()
    if len(exog_indices) > 0:
        # Fazer shift das exógenas
        input_window[:-1, exog_indices] = current_x_scaled[1:, exog_indices]
        # Inserir exógena do dia que vamos prever no último step
        input_window[-1, exog_indices] = future_exog[0]
    
    with torch.no_grad(): # BOA PRÁTICA: Desligar gradientes na inferência
        for i in range(horizon):

            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device) # (1, L, C_in)
            
            y_pred = model(x_tensor)
            val_pred = y_pred.item()
            preds_scaled.append(val_pred)

            # --- Shift da Janela ---
            input_window = np.roll(input_window, -1, axis=0)
            
            # 1. O novo target no último step é a previsão que acabámos de fazer
            input_window[-1, target_channel] = val_pred
            
            # 2. Inserir as exógenas do PRÓXIMO dia no último step
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]
            
    # Reshape para fazer inverse_transform
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds