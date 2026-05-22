"""
train.py — Training loop for the Static-GCN + LSTM forecaster.

What changes vs the per-sample ego-graph approach:
  • The global graph is built ONCE before training (in graph_builder.py).
  • Each DataLoader batch is (ts_seq, target_node_idx, y) — no PyG Batch needed.
  • The model forward call receives the same global_x / edge_index / edge_weight
    on every step; only target_node_indices differ per batch.
  • GCN and LSTM parameters are optimised jointly end-to-end.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from gcn_lstm_pyg import StaticGCNLSTMForecaster
from dataset import StaticGCNDataset, make_windows
from graph_builder import build_ego_graph, NODE_FEAT_DIM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lookback    : int   = 30
    horizon     : int   = 1
    batch_size  : int   = 32
    train_size  : int   = 455
    val_size    : int   = 153
    lr          : float = 1e-3
    weight_decay: float = 1e-3
    epochs      : int   = 1000
    device      : str   = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_static_gcn_lstm(
    df: pd.DataFrame,
    cfg: TrainConfig,
    seed: int,
    loss_type: str,
    product_id,
    scaler,
    target_col: str = 'value',
    exog_cols: Optional[List[str]] = None,
    # --- graph ---
    df_wide=None,           # pd.DataFrame (items × dates) for graph construction
    all_item_ids=None,      # ordered list of all item IDs (defines node order)
    target_item_id=None,    # item ID of the product being forecast
    metric: str = 'spearman',
    threshold: float = 0.82,
    train_end_idx_global: int = None,  # column index in df_wide up to which training data ends
    # --- model hypers ---
    hidden_sizes=(32, 16),  # (gcn_hidden, gcn_out)
    lstm_hidden: int = 64,
    lstm_layers: int = 1,
    dropout: float = 0.2,
    patience: int = 150,
    test_size: int = None,
    include_2hop: bool = False,
    gcn_layers: int = 2,
    graph_conditioning: str = 'init',
) -> tuple:
    """
    Train StaticGCNLSTMForecaster end-to-end.

    Returns
    -------
    model, scaler, train_losses, val_losses, best_epoch,
    graph_data, item_to_node
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.device)

    exog_cols = exog_cols or []
    cols      = [target_col] + exog_cols
    data      = df[cols].values.astype(np.float32)

    # ── Data splits ────────────────────────────────────────────────────────
    test_start_idx = -test_size if test_size is not None else -cfg.horizon
    val_end_idx    = test_start_idx
    val_start_idx  = val_end_idx - cfg.val_size
    train_end_idx  = val_start_idx

    train_data = data[:train_end_idx]
    val_data   = data[val_start_idx:val_end_idx]

    # Scale target channel
    train_scaled = train_data.copy()
    train_scaled[:, 0:1] = scaler.fit_transform(train_data[:, 0:1])

    val_scaled = val_data.copy()
    val_scaled[:, 0:1] = scaler.transform(val_data[:, 0:1])

    train_ts  = train_scaled[:, 0:1]
    train_cal = train_scaled[:, 1:] if len(exog_cols) > 0 else np.zeros((len(train_scaled), 0), dtype=np.float32)
    val_ts    = val_scaled[:, 0:1]
    val_cal   = val_scaled[:, 1:]   if len(exog_cols) > 0 else np.zeros((len(val_scaled), 0), dtype=np.float32)

    cal_dim = train_cal.shape[1]
    print(f"[train] train={len(train_ts)} val={len(val_ts)} cal_dim={cal_dim}")

    # ── Build the ego-graph around the target item (once) ─────────────────
    graph_data, item_to_node = build_ego_graph(
        target_item_id  = target_item_id,
        df_wide         = df_wide,
        item_ids        = all_item_ids,
        metric          = metric,
        threshold       = threshold,
        train_end_idx   = train_end_idx_global,
        include_2hop    = include_2hop,
    )
    graph_data = graph_data.to(device)

    target_node_idx = item_to_node[target_item_id]
    print(f"[train] target item {target_item_id} → node {target_node_idx} | "
          f"graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} directed edges")

    # ── Build sliding-window datasets ──────────────────────────────────────
    ts_train, idx_train, y_train = make_windows(
        train_ts, train_cal, cfg.lookback, cfg.horizon, target_node_idx
    )
    ts_val, idx_val, y_val = make_windows(
        val_ts, val_cal, cfg.lookback, cfg.horizon, target_node_idx
    )

    train_loader = DataLoader(
        StaticGCNDataset(ts_train, idx_train, y_train),
        batch_size=cfg.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        StaticGCNDataset(ts_val, idx_val, y_val),
        batch_size=cfg.batch_size, shuffle=False,
    )

    # ── Build model ────────────────────────────────────────────────────────
    gcn_hidden = hidden_sizes[0] if len(hidden_sizes) > 0 else 32
    gcn_out    = hidden_sizes[1] if len(hidden_sizes) > 1 else 16

    model = StaticGCNLSTMForecaster(
        in_channels        = NODE_FEAT_DIM,
        gcn_hidden         = gcn_hidden,
        gcn_out            = gcn_out,
        lstm_input_size    = 1 + cal_dim,
        lstm_hidden        = lstm_hidden,
        lstm_layers        = lstm_layers,
        horizon            = cfg.horizon,
        dropout            = dropout,
        gcn_layers         = gcn_layers,
        graph_conditioning = graph_conditioning,
    ).to(device)

    print(f"  GCN in={NODE_FEAT_DIM} hidden={gcn_hidden} out={gcn_out} "
          f"layers={gcn_layers} conditioning={graph_conditioning!r} | "
          f"LSTM input={1+cal_dim} hidden={lstm_hidden} layers={lstm_layers}")

    # ── Optimiser & loss ───────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = {
        'mse'  : nn.MSELoss(),
        'mae'  : nn.L1Loss(),
        'huber': nn.HuberLoss(),
    }.get(loss_type.lower())
    if loss_fn is None:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    # Persistent graph tensors (moved to device once)
    gx = graph_data.x
    ei = graph_data.edge_index
    ew = graph_data.edge_weight if graph_data.edge_weight.numel() > 0 else None

    best_val   = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []

    model_dir       = os.path.join(os.path.dirname(__file__), f'best_models/seed_{seed}/{loss_type}')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f'static_gcn_lstm_{product_id}.pth')

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for ts_b, idx_b, y_b in train_loader:
            ts_b  = ts_b.to(device)
            idx_b = idx_b.to(device)
            y_b   = y_b.to(device)

            pred = model(gx, ei, ew, idx_b, ts_b)
            loss = loss_fn(pred, y_b)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * y_b.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ts_b, idx_b, y_b in val_loader:
                ts_b  = ts_b.to(device)
                idx_b = idx_b.to(device)
                y_b   = y_b.to(device)
                pred  = model(gx, ei, ew, idx_b, ts_b)
                val_loss += loss_fn(pred, y_b).item() * y_b.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_path)
            best_epoch = epoch
            no_improve = 0
            print(f"Epoch {epoch:4d}: train={train_loss:.6f}  val={val_loss:.6f}  (new best)")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stop at epoch {epoch} ({patience} epochs no improvement)")
                break

    print(f"Best val {loss_type.upper()}: {best_val:.6f} (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_losses, val_losses, best_epoch, graph_data, item_to_node
