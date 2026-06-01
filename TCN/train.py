import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader
import copy
@dataclass
class TrainConfig:
    lookback: int = 30
    horizon: int = 153
    batch_size: int = 32
    train_size: int = 455
    val_size : int = 153
    lr: float = 1e-4
    epochs: int = 300       # Or leave at 1000, but patience is what matters
    weight_decay: float = 1e-2  # <--- Increase from 1e-3 to give AdamW more teeth
    patience: int = 30          # <--- Decrease from 150. Stop training when val loss flatlines!
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    


def train_tcn_forecaster(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    device: str, 
    epochs: int = 300, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-2,  # Aggressive weight decay for small datasets
    patience: int = 30           # Fast early stopping
):
    model = model.to(device)
    
    # Use AdamW to properly decouple the weight decay penalty
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        epoch_train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            
            loss.backward()
            # Clip gradients to prevent exploding weights
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * xb.size(0)
            
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                epoch_val_loss += criterion(pred, yb).item() * xb.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Print only when we find a new best to keep logs clean
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} (New Best)")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {best_epoch}.")
            break

    # Restore the absolute best version of the model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_epoch