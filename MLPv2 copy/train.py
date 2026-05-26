


from dataclasses import dataclass
from time import time
import time
from typing import Tuple, Optional, List
from mlp import MLPForecaster
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from dataset import make_windows, WindowDataset
@dataclass
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

    # MLPForecaster is strict 1-step: collapse (N, 1, 1) -> (N, 1) so the loss matches model output.
    assert cfg.horizon == 1, "MLPForecaster is strict 1-step; cfg.horizon must be 1."
    y_train = y_train_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)
    y_val = y_val_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)

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

def train_model_best_train_loss(
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
    """
    Strategy 1: Records the epoch of the best validation model, but saves the model 
    with the best training loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values
    
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - cfg.val_size
    train_end_idx = val_start_idx 
    
    train_data = data[:train_end_idx]
    
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

    assert cfg.horizon == 1, "MLPForecaster is strict 1-step; cfg.horizon must be 1."
    y_train = y_train_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)
    y_val = y_val_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)

    C_in = X_train.shape[2]

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f'{model_dir}/mlp_product_{product_id}_best_train.pth'

    start_train_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Save model based on best training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                outputs = model(xb)
                val_loss += criterion(outputs, yb).item() * xb.size(0)

        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    train_time = time.time() - start_train_time
    
    # Load the best train loss state before returning
    model.load_state_dict(torch.load(best_model_path))

    return model, scaler, train_losses, val_losses, best_val_epoch


def train_model_combined(
    df: pd.DataFrame, 
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,    
    scaler,
    target_channel,  
    val_ratio, # Unused here but kept for signature consistency
    hidden_sizes,
    target_col,
    exog_cols,
    test_size
):
    """
    Strategy 2: Train an existing model further (such as when combining Train + Val).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values
    
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    # Combine Train and Val splits
    train_end_idx = test_start_idx 
    
    train_data = data[:train_end_idx]
    
    train_scaled = train_data.copy()
    train_target = train_data[:, target_channel:target_channel+1]
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(train_target)
    
    X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
    assert cfg.horizon == 1, "MLPForecaster is strict 1-step; cfg.horizon must be 1."
    y_train = y_train_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)

    C_in = X_train.shape[2]

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        
    model = MLPForecaster(
        lookback=cfg.lookback,
        in_channels=C_in,
        horizon=cfg.horizon,
        out_dim=1, 
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = f'{model_dir}/mlp_product_{product_id}_combined.pth'

    start_train_time = time.time()
    train_losses = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"Epoch {epoch+1}/{cfg.epochs} | Combined Train Loss: {avg_train_loss:.6f}")

    # Save the retrained model
    torch.save(model.state_dict(), final_model_path)
        
    train_time = time.time() - start_train_time

    return model, scaler, train_losses, train_time

def train_model_expanding_window(
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
    """
    Strategy 3: Expanding Window (Growing Window) Training using TrainConfig parameters.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values
    
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    total_len = len(data[:test_start_idx])
    
    # We define a sensible val_step_size and initial_train_size
    val_step_size = cfg.val_size
    initial_train_size = total_len - val_step_size # The first expanding window stops right before validation
    
    current_train_end = initial_train_size
    
    model = MLPForecaster(
        lookback=cfg.lookback,
        in_channels=len(cols),
        horizon=cfg.horizon,
        out_dim=1, 
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = f'{model_dir}/mlp_product_{product_id}_expanding.pth'

    start_train_time = time.time()
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

    window_idx = 0
    patience = 10
    
    # Walk through the data by expanding the train window limit repeatedly
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Expanding Window {window_idx}: Train 0->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")
        
        train_data = data[:current_train_end]
        val_start = current_train_end - cfg.lookback
        val_end = current_train_end + val_step_size
        val_data = data[val_start:val_end]
        
        train_scaled = train_data.copy()
        train_target = train_data[:, target_channel:target_channel+1]
        train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(train_target)
        
        val_scaled = val_data.copy()
        val_target = val_data[:, target_channel:target_channel+1]
        val_scaled[:, target_channel:target_channel+1] = scaler.transform(val_target)
        
        X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
        X_val, y_val_full = make_windows(val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
        
        # If there's insufficient validation data to make windows, break or continue
        if len(X_val) == 0 or len(X_train) == 0:
             print("Skipping window due to insufficient data for seq_length.")
             current_train_end += val_step_size
             continue

        assert cfg.horizon == 1, "MLPForecaster is strict 1-step; cfg.horizon must be 1."
        y_train = y_train_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)
        y_val = y_val_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
        
        patience_counter = 0
        
        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                
            avg_train_loss = epoch_loss / len(train_loader.dataset)
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                    outputs = model(xb)
                    val_loss += criterion(outputs, yb).item() * xb.size(0)
            
            if len(val_loader.dataset) > 0:
                val_loss /= len(val_loader.dataset)
            
                # Keep globally lowest validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_overall_epoch = epoch + 1
                    torch.save(model.state_dict(), final_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} in window {window_idx}.")
                break
                
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(val_loss)
                
        current_train_end += val_step_size
        window_idx += 1
        
    train_time = time.time() - start_train_time
    # Restore best overall weights
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))

    return model, scaler, all_train_losses, all_val_losses, best_overall_epoch


def train_model_sliding_window(
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
    """
    Strategy 4: Sliding Window Training using TrainConfig parameters.
    The model trains progressively on a sliding dataset window, validating on the immediate next block.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values
    
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    total_len = len(data[:test_start_idx])
    
    val_step_size = cfg.val_size
    initial_train_size = total_len - val_step_size
    
    current_train_start = 0
    current_train_end = initial_train_size
    
    model = MLPForecaster(
        lookback=cfg.lookback,
        in_channels=len(cols),
        horizon=cfg.horizon,
        out_dim=1, 
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = f'{model_dir}/mlp_product_{product_id}_sliding.pth'

    start_train_time = time.time()
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

    window_idx = 0
    patience = 10
    
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Sliding Window {window_idx}: Train {current_train_start}->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")
        
        train_data = data[current_train_start:current_train_end]
        val_start = current_train_end - cfg.lookback
        val_end = current_train_end + val_step_size
        val_data = data[val_start:val_end]
        
        train_scaled = train_data.copy()
        train_target = train_data[:, target_channel:target_channel+1]
        train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(train_target)
        
        val_scaled = val_data.copy()
        val_target = val_data[:, target_channel:target_channel+1]
        val_scaled[:, target_channel:target_channel+1] = scaler.transform(val_target)
        
        X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
        X_val, y_val_full = make_windows(val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
        
        if len(X_val) == 0 or len(X_train) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_start += val_step_size
            current_train_end += val_step_size
            continue

        assert cfg.horizon == 1, "MLPForecaster is strict 1-step; cfg.horizon must be 1."
        y_train = y_train_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)
        y_val = y_val_full[:, :, target_channel : target_channel + 1].reshape(-1, 1)

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
        
        patience_counter = 0
        
        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                
            avg_train_loss = epoch_loss / len(train_loader.dataset)
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                    outputs = model(xb)
                    val_loss += criterion(outputs, yb).item() * xb.size(0)
            
            if len(val_loader.dataset) > 0:
                val_loss /= len(val_loader.dataset)
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_overall_epoch = epoch + 1
                    torch.save(model.state_dict(), final_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} in window {window_idx}.")
                break
                
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(val_loss)
                
        current_train_start += val_step_size
        current_train_end += val_step_size
        window_idx += 1
        
    train_time = time.time() - start_train_time
    
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))

    return model, scaler, all_train_losses, all_val_losses, best_overall_epoch

