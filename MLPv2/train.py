


from dataclasses import dataclass
from time import time
import time
from typing import Tuple, Optional, List

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
    target_channel: int = 0,    
    val_ratio: float = 0.2,
    hidden_sizes=(16, 8),
    target_col: str = TARGET_COL,
    exog_cols: Optional[List[str]] = EXOG_COLS,
    test_size: Optional[int] = None,
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

    X_train, y_train_full = make_windows(train_scaled, cfg.lookback, cfg.horizon)
    X_val, y_val_full = make_windows(val_scaled, cfg.lookback, cfg.horizon)

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

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
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

def train_model_best_train_loss(seed, epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=10):
    """
    Strategy 1: Records the epoch of the best validation model, but saves the model 
    with the best training loss.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    start_train_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Save model based on best training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)

        # Validation
        model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
                all_outputs.append(outputs)
                all_targets.append(batch_y)

        all_outputs = torch.cat(all_outputs, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        
        val_loss = criterion2(all_outputs, all_targets).item()
        val_losses.append(val_loss)
        
        if scheduler is not None:
            scheduler.step(val_loss)

        # Track early stopping and epochs based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            patience_counter = 0  # Reset patience on improvement
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    train_time = time.time() - start_train_time

    # Return best_val_epoch as requested
    return model, train_losses, val_losses, best_val_epoch, train_time


def train_model_combined(seed, epochs_to_train, model, combined_loader, criterion, optimizer, device, final_model_path):
    """
    Strategy 2: Train an existing model further (such as when combining Train + Val).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    start_train_time = time.time()
    train_losses = []

    for epoch in range(epochs_to_train):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(combined_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(combined_loader)
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == epochs_to_train - 1:
            print(f"Epoch {epoch+1}/{epochs_to_train} | Combined Train Loss: {avg_train_loss:.6f}")

    # Save the retrained model
    torch.save(model.state_dict(), final_model_path)
        
    train_time = time.time() - start_train_time

    return model, train_losses, train_time

def train_model_expanding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 3: Expanding Window (Growing Window) Training.
    The model trains progressively on an expanding dataset array, validating on the immediate next block.
    After each window, the model retains weights to fine-tune across expanding windows, simulating real longitudinal training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)
    current_train_end = initial_train_size
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

    window_idx = 0
    # Walk through the data by expanding the train window limit repeatedly
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Expanding Window {window_idx}: Train 0->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")
        
        train_data = full_train_scaled[:current_train_end]
        
        val_start = current_train_end - seq_length
        val_end = current_train_end + val_step_size
        val_data = full_train_scaled[val_start:val_end]
        
        exog_train = exog_scaled[:current_train_end] if exog_scaled is not None else None
        exog_val = exog_scaled[val_start:val_end] if exog_scaled is not None else None
        
        train_dataset = dataset_class(train_data, exog_train, seq_length)
        val_dataset = dataset_class(val_data, exog_val, seq_length)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_end += val_step_size
            continue
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation Step
            model.eval()
            all_outputs = []
            all_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
                    all_outputs.append(outputs)
                    all_targets.append(batch_y)
            
            all_outputs = torch.cat(all_outputs, dim=0).view(-1)
            all_targets = torch.cat(all_targets, dim=0).view(-1)
            val_loss = criterion2(all_outputs, all_targets).item()
            
            if scheduler is not None:
                scheduler.step(val_loss)
                
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
    return model, all_train_losses, all_val_losses, best_overall_epoch, train_time


def train_model_sliding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 4: Sliding Window Training.
    The model trains progressively on a sliding dataset window, validating on the immediate next block.
    The training window size remains fixed, but slides forward on each iteration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)
    
    current_train_start = 0
    current_train_end = initial_train_size
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

    window_idx = 0
    
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Sliding Window {window_idx}: Train {current_train_start}->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")
        
        train_data = full_train_scaled[current_train_start:current_train_end]
        
        val_start = current_train_end - seq_length
        val_end = current_train_end + val_step_size
        val_data = full_train_scaled[val_start:val_end]
        
        exog_train = exog_scaled[current_train_start:current_train_end] if exog_scaled is not None else None
        exog_val = exog_scaled[val_start:val_end] if exog_scaled is not None else None
        
        train_dataset = dataset_class(train_data, exog_train, seq_length)
        val_dataset = dataset_class(val_data, exog_val, seq_length)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_start += val_step_size
            current_train_end += val_step_size
            continue
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation Step
            model.eval()
            all_outputs = []
            all_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    if outputs.dim() == 3: outputs = outputs.view(outputs.size(0), -1)
                    all_outputs.append(outputs)
                    all_targets.append(batch_y)
            
            all_outputs = torch.cat(all_outputs, dim=0).view(-1)
            all_targets = torch.cat(all_targets, dim=0).view(-1)
            val_loss = criterion2(all_outputs, all_targets).item()
            
            if scheduler is not None:
                scheduler.step(val_loss)
                
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
                
        current_train_start += val_step_size
        current_train_end += val_step_size
        window_idx += 1
        
    train_time = time.time() - start_train_time
    return model, all_train_losses, all_val_losses, best_overall_epoch, train_time

