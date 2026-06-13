


from dataclasses import dataclass
import time
from typing import Tuple, Optional, List
from mlp import TimeDistributedMLPForecaster
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from dataset import make_windows, WindowDataset
from losses import get_loss 
@dataclass
class TrainConfig:
    lookback: int = 30
    horizon: int = 1
    batch_size: int = 32
    train_size: int = 455
    val_size : int = 153
    lr: float = 1e-4
    epochs: int = 1000
    weight_decay: float = 1e-3
    patience: int = 150
    hidden_sizes: tuple = (64, 32)
    dropout: float = 0.2
    # Walk-forward CV fold geometry for expanding/sliding window strategies.
    # Pool = train+val. initial=365 -> first fold sees a full seasonal year;
    # step=81 tiles the remaining ~243 days into 3 folds whose last validation
    # block ends at the test boundary (no recent data wasted).
    cv_initial_train_size: int = 365
    cv_val_step_size: int = 81
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _make_criterion(loss_type):
    if loss_type.lower() == 'mse':
        return nn.MSELoss()
    elif loss_type.lower() == 'mae':
        return nn.L1Loss()
    elif loss_type.lower() == 'huber':
        return nn.HuberLoss()
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _scale_and_window(raw, scaler, fit, cfg, target_channel):
    """Scale the target channel (fit or transform) and build (X, y) windows."""
    scaled = raw.copy()
    tgt = raw[:, target_channel:target_channel + 1]
    scaled[:, target_channel:target_channel + 1] = scaler.fit_transform(tgt) if fit else scaler.transform(tgt)
    X, y_full = make_windows(scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
    y = y_full[:, :, target_channel:target_channel + 1]
    return X, y


def _train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch and return the sample-weighted average loss."""
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / len(loader.dataset)


def _eval_loss(model, loader, criterion, device):
    """Return the sample-weighted validation loss over the loader."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    return val_loss / len(loader.dataset)


def _mean_curves(fold_train_curves, fold_val_curves):
    """Average per-fold train/val curves epoch-by-epoch (truncated to the shortest fold)."""
    n = min(len(c) for c in fold_val_curves)
    mean_train = [float(np.mean([c[e] for c in fold_train_curves])) for e in range(n)]
    mean_val = [float(np.mean([c[e] for c in fold_val_curves])) for e in range(n)]
    return mean_train, mean_val


def _select_epochs_from_mean_curve(mean_val_curve, patience):
    """Early-stopping selection on the mean fold curve. Returns (best_epoch, best_mean_val)."""
    best_mean = float('inf')
    best_epoch = 0
    counter = 0
    for e, mv in enumerate(mean_val_curve):
        if mv < best_mean:
            best_mean = mv
            best_epoch = e + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return best_epoch, best_mean


def train_mlp_forecaster(
    df: pd.DataFrame, 
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,    
    scaler,
    target_channel,  
    val_ratio,          # unused: val split is determined by cfg.val_size and test_size
    target_col=None,
    exog_cols=None,
    test_size=None,
    model_type: str = "mlp",
):
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
    

    # scaler is passed as argument
    train_scaled = train_data.copy()
    train_target = train_data[:, target_channel:target_channel+1]
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(train_target)
    
    # Prepend `lookback` rows of train context so make_windows produces `val_size`
    # windows. Without this, val windows = val_size - lookback (= 0 when equal).
    val_context_start_idx = val_start_idx - cfg.lookback
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
    print("Output Shape: (Batch, 1)")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
        
    model = TimeDistributedMLPForecaster(
        in_channels=C_in,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
        activation="relu",
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        loss_fn = nn.HuberLoss()
    elif loss_type.lower() == 'tweedie':
        loss_fn = get_loss('tweedie')
    elif loss_type.lower() == 'quantile':
        loss_fn = get_loss('quantile')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'mse', 'mae', 'huber', 'tweedie', or 'quantile'.")

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f'{model_dir}/mlp_item{product_id}.pth'
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
                patience_counter = 0
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} (New Best)")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs).")
                    break
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
    test_size,
    model_type: str = "mlp",
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
    
    # Prepend `lookback` rows of train context so make_windows produces `val_size`
    # windows. Without this, val windows = val_size - lookback (= 0 when equal).
    val_context_start_idx = val_start_idx - cfg.lookback
    val_context_data = data[val_context_start_idx : val_end_idx]
    
    val_scaled = val_context_data.copy()
    val_target = val_context_data[:, target_channel:target_channel+1]
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(val_target)

    X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)
    X_val, y_val_full = make_windows(val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel)

    y_train = y_train_full[:, :, target_channel : target_channel + 1]
    y_val = y_val_full[:, :, target_channel : target_channel + 1]

    C_in = X_train.shape[2]

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
        
    model = TimeDistributedMLPForecaster(
        in_channels=C_in,
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
    test_size,
    model_type: str = "mlp",
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
    y_train = y_train_full[:, :, target_channel : target_channel + 1]

    C_in = X_train.shape[2]

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        
    model = TimeDistributedMLPForecaster(
        in_channels=C_in,
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
    test_size,
    model_type: str = "mlp",
):
    """
    Strategy 3: Expanding Window (Growing Window) Cross-Validation.

    The train window grows by cfg.cv_val_step_size each fold; the next block is the validation
    set. Each fold is INDEPENDENT (fresh weights + fresh optimizer per fold) and runs the full
    cfg.epochs budget, recording its per-epoch validation curve. Model selection: the per-fold
    val curves are averaged epoch-by-epoch and E* is picked by early-stopping (cfg.patience) on
    that MEAN curve. A single fresh model is then refit on ALL train+val data for E* epochs ->
    this is the deployed model. The passed scaler is left fit on the full refit data so the
    caller's inference scaling matches the deployed model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values

    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    total_len = len(data[:test_start_idx])

    val_step_size = cfg.cv_val_step_size
    initial_train_size = cfg.cv_initial_train_size

    model = TimeDistributedMLPForecaster(
        in_channels=len(cols),
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)
    # Pristine init captured once -> every fold starts identically and independently.
    initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    criterion = _make_criterion(loss_type)

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = f'{model_dir}/mlp_product_{product_id}_expanding.pth'

    start_train_time = time.time()

    fold_train_curves = []
    fold_val_curves = []

    current_train_end = initial_train_size
    window_idx = 0
    # Walk through the data by expanding the train window limit repeatedly
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Expanding Window {window_idx}: Train 0->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")

        train_data = data[:current_train_end]
        val_start = current_train_end - cfg.lookback
        val_end = current_train_end + val_step_size
        val_data = data[val_start:val_end]

        X_train, y_train = _scale_and_window(train_data, scaler, True, cfg, target_channel)
        X_val, y_val = _scale_and_window(val_data, scaler, False, cfg, target_channel)

        if len(X_val) == 0 or len(X_train) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_end += val_step_size
            continue

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

        # Independent fold: reset weights and start a fresh optimizer.
        model.load_state_dict(initial_model_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Run the full epoch budget so every fold's curve is aligned for averaging.
        train_curve, val_curve = [], []
        for epoch in range(cfg.epochs):
            train_curve.append(_train_one_epoch(model, train_loader, criterion, optimizer, cfg.device))
            val_curve.append(_eval_loss(model, val_loader, criterion, cfg.device))

        fold_train_curves.append(train_curve)
        fold_val_curves.append(val_curve)

        current_train_end += val_step_size
        window_idx += 1

    if len(fold_val_curves) == 0:
        return model, scaler, [], [], 0

    # Average across folds and pick E* on the mean validation curve.
    mean_train_curve, mean_val_curve = _mean_curves(fold_train_curves, fold_val_curves)
    best_overall_epoch, best_mean = _select_epochs_from_mean_curve(mean_val_curve, cfg.patience)
    print(f"\\n[Expanding CV] {len(fold_val_curves)} folds | E* = {best_overall_epoch} epochs | mean val loss = {best_mean:.6f}")

    # Refit one fresh model on ALL train+val data for E* epochs -> deployed model.
    X_ref, y_ref = _scale_and_window(data[:total_len], scaler, True, cfg, target_channel)
    refit_loader = DataLoader(WindowDataset(X_ref, y_ref), batch_size=cfg.batch_size, shuffle=False)
    model.load_state_dict(initial_model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(best_overall_epoch):
        avg_train_loss = _train_one_epoch(model, refit_loader, criterion, optimizer, cfg.device)
        if (epoch + 1) % 10 == 0 or epoch == best_overall_epoch - 1:
            print(f"  [refit] Epoch {epoch+1}/{best_overall_epoch} | Train Loss: {avg_train_loss:.6f}")
    torch.save(model.state_dict(), final_model_path)

    train_time = time.time() - start_train_time
    return model, scaler, mean_train_curve, mean_val_curve, best_overall_epoch


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
    test_size,
    model_type: str = "mlp",
):
    """
    Strategy 4: Sliding Window Cross-Validation.

    The train window has a FIXED size (cfg.cv_initial_train_size) and slides forward by
    cfg.cv_val_step_size each fold; the next block is the validation set. Each fold is
    INDEPENDENT (fresh weights + fresh optimizer per fold) and runs the full cfg.epochs budget,
    recording its per-epoch validation curve. Model selection: the per-fold val curves are
    averaged epoch-by-epoch and E* is picked by early-stopping (cfg.patience) on that MEAN
    curve. A single fresh model is then refit for E* epochs on the MOST RECENT window only (the
    last cfg.cv_initial_train_size days), keeping the fixed-window philosophy -> this is the
    deployed model. The passed scaler is left fit on that refit window so the caller's inference
    scaling matches the deployed model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values

    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    total_len = len(data[:test_start_idx])

    val_step_size = cfg.cv_val_step_size
    initial_train_size = cfg.cv_initial_train_size

    model = TimeDistributedMLPForecaster(
        in_channels=len(cols),
        hidden_sizes=hidden_sizes,
        dropout=0.2,
        activation="relu",
    ).to(cfg.device)
    # Pristine init captured once -> every fold starts identically and independently.
    initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    criterion = _make_criterion(loss_type)

    model_dir = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = f'{model_dir}/mlp_product_{product_id}_sliding.pth'

    start_train_time = time.time()

    fold_train_curves = []
    fold_val_curves = []

    current_train_start = 0
    current_train_end = initial_train_size
    window_idx = 0

    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Sliding Window {window_idx}: Train {current_train_start}->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")

        train_data = data[current_train_start:current_train_end]
        val_start = current_train_end - cfg.lookback
        val_end = current_train_end + val_step_size
        val_data = data[val_start:val_end]

        X_train, y_train = _scale_and_window(train_data, scaler, True, cfg, target_channel)
        X_val, y_val = _scale_and_window(val_data, scaler, False, cfg, target_channel)

        if len(X_val) == 0 or len(X_train) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_start += val_step_size
            current_train_end += val_step_size
            continue

        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

        # Independent fold: reset weights and start a fresh optimizer.
        model.load_state_dict(initial_model_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Run the full epoch budget so every fold's curve is aligned for averaging.
        train_curve, val_curve = [], []
        for epoch in range(cfg.epochs):
            train_curve.append(_train_one_epoch(model, train_loader, criterion, optimizer, cfg.device))
            val_curve.append(_eval_loss(model, val_loader, criterion, cfg.device))

        fold_train_curves.append(train_curve)
        fold_val_curves.append(val_curve)

        current_train_start += val_step_size
        current_train_end += val_step_size
        window_idx += 1

    if len(fold_val_curves) == 0:
        return model, scaler, [], [], 0

    # Average across folds and pick E* on the mean validation curve.
    mean_train_curve, mean_val_curve = _mean_curves(fold_train_curves, fold_val_curves)
    best_overall_epoch, best_mean = _select_epochs_from_mean_curve(mean_val_curve, cfg.patience)
    print(f"\\n[Sliding CV] {len(fold_val_curves)} folds | E* = {best_overall_epoch} epochs | mean val loss = {best_mean:.6f}")

    # Refit one fresh model for E* epochs on the MOST RECENT window only -> deployed model.
    refit_start = max(0, total_len - initial_train_size)
    X_ref, y_ref = _scale_and_window(data[refit_start:total_len], scaler, True, cfg, target_channel)
    refit_loader = DataLoader(WindowDataset(X_ref, y_ref), batch_size=cfg.batch_size, shuffle=False)
    model.load_state_dict(initial_model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(best_overall_epoch):
        avg_train_loss = _train_one_epoch(model, refit_loader, criterion, optimizer, cfg.device)
        if (epoch + 1) % 10 == 0 or epoch == best_overall_epoch - 1:
            print(f"  [refit] Epoch {epoch+1}/{best_overall_epoch} | Train Loss: {avg_train_loss:.6f}")
    torch.save(model.state_dict(), final_model_path)

    train_time = time.time() - start_train_time
    return model, scaler, mean_train_curve, mean_val_curve, best_overall_epoch

