
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
from mlp import TimeDistributedMLPForecaster
from GNN.DynamicSimilarities.GAT.TimedistributedMLP_GCN.gat_mlpdataset import make_windows, WindowDataset
@dataclass
class TrainConfig:
    lookback: int = 30
    horizon: int = 1
    batch_size: int = 32
    train_size: int = 455
    val_size: int = 153
    lr: float = 1e-4
    epochs: int = 1000
    weight_decay: float = 1e-3
    dropout: float = 0.2
    patience: int = 150
    hidden_sizes: tuple = (32, 16)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_mlp_forecaster(
    df: pd.DataFrame, 
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,    
    scaler,
    target_channel: int = 0,
    target_col=None,
    exog_cols=None,
    graph_embeddings=None,
    test_size=None
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
    
    # Prepend `lookback` rows of train context so val windows cover the full
    # val split. Without this, val windows = val_size - lookback - horizon + 1
    # which collapses when val_size is small relative to lookback.
    val_context_start_idx = val_start_idx - cfg.lookback
    val_context_data = data[val_context_start_idx : val_end_idx]

    val_scaled = val_context_data.copy()
    val_target = val_context_data[:, target_channel:target_channel+1]
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(val_target)

    # Extract embeddings for Train and Val (if provided).
    # val_scaled starts at (val_start_idx - lookback), so the embedding start
    # must match that same data position, not len(train_scaled).
    if graph_embeddings is not None:
        emb_train = graph_embeddings[:len(train_scaled)]
        val_emb_start = len(train_scaled) - cfg.lookback  # = val_context_start_idx (positive)
        emb_val = graph_embeddings[val_emb_start : val_emb_start + len(val_scaled)]
    else:
        emb_train = None
        emb_val = None

    # Create windows with targets, exogenous features, and embeddings
    X_train, y_train_full = make_windows(
        train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel, embeddings=emb_train, graph_window_size=15
    )
    X_val, y_val_full = make_windows(
        val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel, embeddings=emb_val, graph_window_size=15
    )

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
    patience_counter = 0
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
                if cfg.patience is not None and patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs; best epoch={best_epoch}).")
                    break
        else:
            print("WARNING: Validation set is too small to form windows!")

    if hasattr(val_loader.dataset, "__len__") and len(val_loader.dataset) > 0:
        print(f"Best Val {loss_type.upper()}: {best_val:.6f} (Epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch