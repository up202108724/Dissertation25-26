
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from mlp import MLPForecaster, WindowDataset, make_windows
from gnndataset import (SingleGraphDataset, single_graph_collate,
                             make_single_windows)
from gnn_lstm_pyg import SimpleGNNLSTMForecaster, generate_node_features
   
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
# GCN+LSTM training  (one graph per sample, GCN initialises LSTM h₀/c₀)
# ---------------------------------------------------------------------------

def train_gnn_lstm(
    df: pd.DataFrame,
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,
    scaler,
    target_channel: int = 0,
    hidden_sizes=(32, 16),          # (gcn_hidden_channels, gcn_out_channels)
    target_col=None,
    exog_cols=None,
    graphs=None,
    test_size=None,
    graph_window_size: int = 15,
    dropout: float = 0.2,
    include_cal_lookback: bool = False,
    node_features: list = None,
    cal_columns: list = None,
    lstm_hidden: int = 64,
    lstm_layers: int = 1,
    patience: int = 150,
):
    """
    Train SimpleGNNLSTMForecaster (GCN → h₀/c₀ init → LSTM).

    Returns
    -------
    model, scaler, train_losses, val_losses, best_epoch
    """

    gcn_hidden = hidden_sizes[0] if len(hidden_sizes) > 0 else 32
    gcn_out    = hidden_sizes[1] if len(hidden_sizes) > 1 else 16

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values

    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx    = test_start_idx
    val_start_idx  = val_end_idx - cfg.val_size
    train_end_idx  = val_start_idx

    train_data   = data[:train_end_idx]
    train_scaled = train_data.copy()
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(
        train_data[:, target_channel:target_channel+1]
    )
    val_data   = data[val_start_idx:val_end_idx]
    val_scaled = val_data.copy()
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(
        val_data[:, target_channel:target_channel+1]
    )

    C_in             = train_scaled.shape[1]
    cal_dim          = C_in - 1
    exog_col_indices = [i for i in range(C_in) if i != target_channel]

    print(f"[train_pure_sage] Train end={train_end_idx}  Val [{val_start_idx},{val_end_idx})  "
          f"cal_dim={cal_dim}")

    graphs_train = (graphs[:len(train_scaled)] if graphs is not None else None)
    graphs_val   = (graphs[len(train_scaled): len(train_scaled) + len(val_scaled)]
                    if graphs is not None else None)

    train_ts  = train_scaled[:, target_channel:target_channel+1]
    train_cal = (train_scaled[:, exog_col_indices]
                 if exog_col_indices else np.zeros((len(train_scaled), 0), dtype=np.float32))
    val_ts    = val_scaled[:, target_channel:target_channel+1]
    val_cal   = (val_scaled[:, exog_col_indices]
                 if exog_col_indices else np.zeros((len(val_scaled), 0), dtype=np.float32))

    y_train, g_train, ts_train = make_single_windows(
        train_ts, train_cal, cfg.lookback, cfg.horizon,
        target_channel=0, graphs=graphs_train,
        graph_window_size=graph_window_size,
        include_cal_lookback=include_cal_lookback,
        node_features=node_features, cal_columns=cal_columns,
    )
    y_val, g_val, ts_val = make_single_windows(
        val_ts, val_cal, cfg.lookback, cfg.horizon,
        target_channel=0, graphs=graphs_val,
        graph_window_size=graph_window_size,
        include_cal_lookback=include_cal_lookback,
        node_features=node_features, cal_columns=cal_columns,
    )

    train_loader = DataLoader(
        SingleGraphDataset(y_train, g_train, ts_train),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=single_graph_collate,
    )
    val_loader = DataLoader(
        SingleGraphDataset(y_val, g_val, ts_val),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=single_graph_collate,
    )

    # Dynamic node feature dim
    _dummy_ts  = np.zeros(cfg.lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((cfg.lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    gcn_in_channels = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))
    lstm_input_size = 1 + cal_dim

    model = SimpleGNNLSTMForecaster(
        in_channels=gcn_in_channels,
        hidden_channels=gcn_hidden,
        out_channels=gcn_out,
        lstm_input_size=lstm_input_size,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        horizon=cfg.horizon,
        dropout=dropout,
    ).to(cfg.device)

    print(f"  GCN feat_dim={gcn_in_channels} | "
          f"LSTM input={lstm_input_size} hidden={lstm_hidden} layers={lstm_layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    best_val   = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []

    model_dir       = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f'gnn_lstm_product_{product_id}.pth')

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            yb, pyg_batch, ts_batch = batch
            yb        = yb.to(cfg.device)
            pyg_batch = pyg_batch.to(cfg.device)
            ts_batch  = ts_batch.to(cfg.device)
            pred = model(pyg_batch, pyg_batch.ptr[:-1], ts_batch)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * yb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                yb, pyg_batch, ts_batch = batch
                yb        = yb.to(cfg.device)
                pyg_batch = pyg_batch.to(cfg.device)
                ts_batch  = ts_batch.to(cfg.device)
                pred = model(pyg_batch, pyg_batch.ptr[:-1], ts_batch)
                val_loss += loss_fn(pred, yb).item() * yb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_path)
            best_epoch = epoch
            no_improve = 0
            print(f"Epoch {epoch}: train={train_loss:.6f}  val={val_loss:.6f}  (new best)")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Best val {loss_type.upper()}: {best_val:.6f}  (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch


# ---------------------------------------------------------------------------
# GCN-at-every-step + LSTM training  (L graphs per sample)
# ---------------------------------------------------------------------------

def train_gnn_lstm_sequence(
    df: pd.DataFrame,
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id: str,
    scaler,
    target_channel: int = 0,
    hidden_sizes=(32, 16),          # (gcn_hidden_channels, gcn_out_channels)
    target_col=None,
    exog_cols=None,
    graphs=None,
    test_size=None,
    graph_window_size: int = 15,
    dropout: float = 0.2,
    lstm_hidden: int = 64,
    lstm_layers: int = 1,
    patience: int = 150,
):
    """
    Train SimpleGNNSeqLSTMForecaster (GCN at every lookback step → LSTM).

    At each LSTM step t the GCN processes the ego-graph whose 15-day window
    ends at t; the target-node embedding z_t is concatenated with (value_t,
    cal_{t+1}) as LSTM input.

    Returns
    -------
    model, scaler, train_losses, val_losses, best_epoch
    """
    from gnndataset import (SequenceGraphDataset, sequence_graph_collate,
                             make_sequence_windows)
    from gnn_lstm_pyg import SimpleGNNSeqLSTMForecaster

    gcn_hidden = hidden_sizes[0] if len(hidden_sizes) > 0 else 32
    gcn_out    = hidden_sizes[1] if len(hidden_sizes) > 1 else 16

    cols = [target_col] + (exog_cols if exog_cols else [])
    data = df[cols].values

    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx    = test_start_idx
    val_start_idx  = val_end_idx - cfg.val_size
    train_end_idx  = val_start_idx

    train_data   = data[:train_end_idx]
    train_scaled = train_data.copy()
    train_scaled[:, target_channel:target_channel+1] = scaler.fit_transform(
        train_data[:, target_channel:target_channel+1]
    )
    val_data   = data[val_start_idx:val_end_idx]
    val_scaled = val_data.copy()
    val_scaled[:, target_channel:target_channel+1] = scaler.transform(
        val_data[:, target_channel:target_channel+1]
    )

    C_in             = train_scaled.shape[1]
    cal_dim          = C_in - 1
    exog_col_indices = [i for i in range(C_in) if i != target_channel]

    print(f"[train_gnn_lstm_sequence] Train end={train_end_idx}  "
          f"Val [{val_start_idx},{val_end_idx})  cal_dim={cal_dim}")

    graphs_train = (graphs[:len(train_scaled)] if graphs is not None else None)
    graphs_val   = (graphs[len(train_scaled): len(train_scaled) + len(val_scaled)]
                    if graphs is not None else None)

    train_ts  = train_scaled[:, target_channel:target_channel+1]
    train_cal = (train_scaled[:, exog_col_indices]
                 if exog_col_indices else np.zeros((len(train_scaled), 0), dtype=np.float32))
    val_ts    = val_scaled[:, target_channel:target_channel+1]
    val_cal   = (val_scaled[:, exog_col_indices]
                 if exog_col_indices else np.zeros((len(val_scaled), 0), dtype=np.float32))

    y_train, g_train, ts_train = make_sequence_windows(
        train_ts, train_cal, cfg.lookback, cfg.horizon,
        target_channel=0, graphs=graphs_train,
        graph_window_size=graph_window_size,
    )
    y_val, g_val, ts_val = make_sequence_windows(
        val_ts, val_cal, cfg.lookback, cfg.horizon,
        target_channel=0, graphs=graphs_val,
        graph_window_size=graph_window_size,
    )

    train_loader = DataLoader(
        SequenceGraphDataset(y_train, g_train, ts_train),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=sequence_graph_collate,
    )
    val_loader = DataLoader(
        SequenceGraphDataset(y_val, g_val, ts_val),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=sequence_graph_collate,
    )

    # node feature dim: graph_window_size + cal_dim + 8
    gcn_in_channels = graph_window_size + cal_dim + 8
    lstm_ts_input   = 1 + cal_dim        # (value_t, cal_{t+1})

    model = SimpleGNNSeqLSTMForecaster(
        in_channels=gcn_in_channels,
        hidden_channels=gcn_hidden,
        out_channels=gcn_out,
        lstm_ts_input=lstm_ts_input,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        horizon=cfg.horizon,
        dropout=dropout,
    ).to(cfg.device)

    print(f"  GCN in={gcn_in_channels} hidden={gcn_hidden} out={gcn_out} | "
          f"LSTM input={lstm_ts_input + gcn_out} hidden={lstm_hidden} layers={lstm_layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if loss_type.lower() == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type.lower() == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type.lower() == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    best_val   = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []

    model_dir       = f'best_models/seed_{seed}/{loss_type}'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f'gnn_seq_lstm_product_{product_id}.pth')

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            yb, pyg_batch, B, L, ts_batch = batch
            yb        = yb.to(cfg.device)
            pyg_batch = pyg_batch.to(cfg.device)
            ts_batch  = ts_batch.to(cfg.device)
            pred = model(ts_batch, pyg_batch, B, L)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * yb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                yb, pyg_batch, B, L, ts_batch = batch
                yb        = yb.to(cfg.device)
                pyg_batch = pyg_batch.to(cfg.device)
                ts_batch  = ts_batch.to(cfg.device)
                pred = model(ts_batch, pyg_batch, B, L)
                val_loss += loss_fn(pred, yb).item() * yb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_path)
            best_epoch = epoch
            no_improve = 0
            print(f"Epoch {epoch}: train={train_loss:.6f}  val={val_loss:.6f}  (new best)")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Best val {loss_type.upper()}: {best_val:.6f}  (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch
