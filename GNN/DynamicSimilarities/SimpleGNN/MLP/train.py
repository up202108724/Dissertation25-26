
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
from mlp import MLPForecaster
from GNN.DynamicSimilarities.SimpleGNN.MLP.gnndataset import make_windows, WindowGraphDataset
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
    target_col=None,
    exog_cols=None,
    graphs=None, # Replaced graph_embeddings with graphs
    test_size=None,
    gnn_in_channels=1,
    gnn_hidden_channels=32,
    gnn_out_channels=16,
):
    from GNN.DynamicSimilarities.SimpleGNN.MLP.gnndataset import py_geometric_collate
    from GNN.DynamicSimilarities.SimpleGNN.MLP.gnn_mlp import SimpleGNN_MLP_Forecaster

    use_graphs = graphs is not None

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

    # Extract graphs for Train and Val (instead of embeddings)
    if use_graphs:
        graphs_train = graphs[:len(train_scaled)]
        graphs_val = graphs[len(train_scaled): len(train_scaled) + len(val_scaled)]
    else:
        graphs_train = None
        graphs_val = None

    # Create windows with targets, exogenous features, and graphs
    X_train, y_train_full, graphs_train_seq = make_windows(
        train_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel, graphs=graphs_train, graph_window_size=gnn_in_channels
    )
    X_val, y_val_full, graphs_val_seq = make_windows(
        val_scaled, cfg.lookback, cfg.horizon, target_channel=target_channel, graphs=graphs_val, graph_window_size=gnn_in_channels
    )

    y_train = y_train_full[:, :, target_channel : target_channel + 1]
    y_val = y_val_full[:, :, target_channel : target_channel + 1]

    C_in = X_train.shape[2]
    ts_dim = 1
    cal_dim = C_in - ts_dim
    
    print(f"Train/Val Split Indices:")
    print(f"  Train End: {train_end_idx}")
    print(f"  Val Range:     {val_start_idx} to {val_end_idx}")
    print(f"  Test Range:    {test_start_idx} to End")
    print(f"  Val Targets:   {y_val.shape[0]} windows created")
    print(f"Input Shape:  (Batch, {cfg.lookback}, {C_in})")
    print(f"Output Shape: (Batch, {cfg.horizon}, 1)")
    print(f"Mode: {'SimpleGNN+MLP' if use_graphs else 'Pure MLP (no embeddings)'}")

    if use_graphs:
        train_loader = DataLoader(
            WindowGraphDataset(X_train, y_train, graphs_train_seq), 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            collate_fn=py_geometric_collate
        )
        val_loader = DataLoader(
            WindowGraphDataset(X_val, y_val, graphs_val_seq), 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            collate_fn=py_geometric_collate
        )

        model = SimpleGNN_MLP_Forecaster(
            lookback=cfg.lookback,
            ts_dim=ts_dim,
            cal_dim=cal_dim,
            gat_in_channels=gnn_in_channels,
            gat_hidden_channels=gnn_hidden_channels,
            gat_out_channels=gnn_out_channels,
            horizon=cfg.horizon,
            mlp_hidden_sizes=hidden_sizes,
            dropout=0.2,
        ).to(cfg.device)
    else:
        # Pure MLP baseline: just ts + cal, no graph encoder
        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model = MLPForecaster(
            lookback=cfg.lookback,
            in_channels=C_in,
            horizon=cfg.horizon,
            out_dim=1,
            hidden_sizes=hidden_sizes,
            dropout=0.2,
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
        for batch in train_loader:
            if use_graphs:
                xb, yb, pyg_batch = batch
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pyg_batch = pyg_batch.to(cfg.device)
                ts_seq = xb[:, :, :1]
                cal_seq = xb[:, :, 1:]
                target_node_indices = pyg_batch.ptr[:-1]
                pred = model(ts_seq, cal_seq, pyg_batch, target_node_indices)
            else:
                xb, yb = batch
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
            for batch in val_loader:
                if use_graphs:
                    xb, yb, pyg_batch = batch
                    xb = xb.to(cfg.device)
                    yb = yb.to(cfg.device)
                    pyg_batch = pyg_batch.to(cfg.device)
                    ts_seq = xb[:, :, :1]
                    cal_seq = xb[:, :, 1:]
                    target_node_indices = pyg_batch.ptr[:-1]
                    pred = model(ts_seq, cal_seq, pyg_batch, target_node_indices)
                else:
                    xb, yb = batch
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