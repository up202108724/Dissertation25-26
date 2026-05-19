
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os


@dataclass
class TrainConfig:
    lookback: int = 30
    horizon: int = 1
    batch_size: int = 32
    train_size: int = 455
    val_size: int = 153
    lr: float = 1e-4
    epochs: int = 30
    weight_decay: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Pure GraphSAGE training  (one graph per sample, linear head only)
# ---------------------------------------------------------------------------

def train_pure_sage(
    df: pd.DataFrame,
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,
    scaler,
    target_channel: int = 0,
    hidden_sizes=(64, 32),
    target_col=None,
    exog_cols=None,
    graphs=None,
    test_size=None,
    graph_window_size: int = 15,
    sage_hidden_channels: int = 32,
    sage_out_channels: int = 16,
    dropout: float = 0.2,
    include_cal_lookback: bool = False,
    node_features: list = None,
):
    """
    Train either:
      • PureGraphSAGEForecaster  – when graphs is not None
      • MLPForecaster (baseline)  – when graphs is None

    Returns
    -------
    model, scaler, train_losses, val_losses, best_epoch
    """
    from graphsagedataset import (SingleGraphDataset, single_graph_collate,
                                   make_single_windows, make_xy_windows)
    from graphsage_pyg import PureGraphSAGEForecaster
    from mlp import MLPForecaster

    use_graphs = graphs is not None

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values                         # (T, 1 + cal_dim)

    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx    = test_start_idx
    val_start_idx  = val_end_idx - cfg.val_size
    train_end_idx  = val_start_idx

    train_data = data[:train_end_idx]
    train_scaled = train_data.copy()
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(
        train_data[:, target_channel:target_channel+1]
    )

    val_data   = data[val_start_idx:val_end_idx]
    val_scaled = val_data.copy()
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(
        val_data[:, target_channel:target_channel+1]
    )

    C_in    = train_scaled.shape[1]
    cal_dim = C_in - 1       # all columns except the target

    exog_col_indices = [i for i in range(C_in) if i != target_channel]

    print(f"Train/Val Split Indices:")
    print(f"  Train End: {train_end_idx}")
    print(f"  Val Range: {val_start_idx} to {val_end_idx}")
    print(f"  Test Range: {test_start_idx} to end")
    print(f"Mode: {'Pure GraphSAGE' if use_graphs else 'Pure MLP (no graph)'}")

    # ------------------------------------------------------------------ #
    #  Build loaders                                                       #
    # ------------------------------------------------------------------ #
    if use_graphs:
        graphs_train = graphs[:len(train_scaled)]
        graphs_val   = graphs[len(train_scaled): len(train_scaled) + len(val_scaled)]

        train_ts  = train_scaled[:, target_channel:target_channel+1]   # (T, 1)
        train_cal = (train_scaled[:, exog_col_indices]
                     if exog_col_indices else np.zeros((len(train_scaled), 0), dtype=np.float32))

        val_ts    = val_scaled[:, target_channel:target_channel+1]
        val_cal   = (val_scaled[:, exog_col_indices]
                     if exog_col_indices else np.zeros((len(val_scaled), 0), dtype=np.float32))

        y_train, g_train = make_single_windows(
            train_ts, train_cal, cfg.lookback, cfg.horizon,
            target_channel=0, graphs=graphs_train,
            graph_window_size=graph_window_size,
            include_cal_lookback=include_cal_lookback,
            node_features=node_features,
        )
        y_val, g_val = make_single_windows(
            val_ts, val_cal, cfg.lookback, cfg.horizon,
            target_channel=0, graphs=graphs_val,
            graph_window_size=graph_window_size,
            include_cal_lookback=include_cal_lookback,
            node_features=node_features,
        )

        train_loader = DataLoader(
            SingleGraphDataset(y_train, g_train),
            batch_size=cfg.batch_size, shuffle=True,
            collate_fn=single_graph_collate,
        )
        val_loader = DataLoader(
            SingleGraphDataset(y_val, g_val),
            batch_size=cfg.batch_size, shuffle=False,
            collate_fn=single_graph_collate,
        )

        if include_cal_lookback:
            sage_in_channels = cfg.lookback * (1 + cal_dim) + cal_dim + 8
        else:
            sage_in_channels = cfg.lookback + cal_dim + 8
        model = PureGraphSAGEForecaster(
            in_channels=sage_in_channels,
            hidden_channels=sage_hidden_channels,
            out_channels=sage_out_channels,
            horizon=cfg.horizon,
            dropout=dropout,
        ).to(cfg.device)

        print(f"  SAGE node feature dim: {sage_in_channels}  "
              f"(lookback={cfg.lookback} + cal_dim={cal_dim} + stats=8)")

    else:
        # Pure MLP baseline
        X_train, y_train = make_xy_windows(train_scaled, cfg.lookback, cfg.horizon, target_channel)
        X_val,   y_val   = make_xy_windows(val_scaled,   cfg.lookback, cfg.horizon, target_channel)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=cfg.batch_size, shuffle=False,
        )

        model = MLPForecaster(
            lookback=cfg.lookback,
            in_channels=C_in,
            horizon=cfg.horizon,
            out_dim=1,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(cfg.device)

    # ------------------------------------------------------------------ #
    #  Loss / optimiser                                                    #
    # ------------------------------------------------------------------ #
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type: '{loss_type}'. Choose 'mse', 'mae', or 'huber'.")

    best_val   = float("inf")
    best_state = None
    best_epoch = 0
    train_losses, val_losses = [], []

    model_dir       = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f'{model_dir}/sage_product_{product_id}.pth'

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    for epoch in range(1, cfg.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if use_graphs:
                yb, pyg_batch = batch
                yb        = yb.to(cfg.device)
                pyg_batch = pyg_batch.to(cfg.device)
                target_node_indices = pyg_batch.ptr[:-1]
                pred = model(pyg_batch, target_node_indices)
            else:
                xb, yb = batch
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                pred = model(xb)

            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * yb.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_graphs:
                    yb, pyg_batch = batch
                    yb        = yb.to(cfg.device)
                    pyg_batch = pyg_batch.to(cfg.device)
                    target_node_indices = pyg_batch.ptr[:-1]
                    pred = model(pyg_batch, target_node_indices)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                    pred = model(xb)

                val_loss += loss_fn(pred, yb).item() * yb.size(0)

        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save(best_state, best_model_path)
                best_epoch = epoch
                print(f"Epoch {epoch}: train={train_loss:.6f}  val={val_loss:.6f}  (new best)")
        else:
            print("WARNING: Validation set too small to form windows!")

    print(f"Best val {loss_type.upper()}: {best_val:.6f}  (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch