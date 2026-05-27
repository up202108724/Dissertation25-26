"""Shared (global) GCN pipeline.

ONE `GCNMLPForecaster` is trained jointly on the ego-graphs of ALL products,
then used in lock-step at inference for every product.

Why shared
----------
A GCN layer is `H' = sigma(A_hat H W)`. The weights `W` don't depend on
*which* node is the center, *which* product the center represents, *which*
neighbors are picked, or *how many* neighbors there are. So a single model
can be trained on (product, time) pairs drawn from the entire catalog and
applied to any product's ego-graph at inference.

Pipeline
--------
1. `build_shared_training_data` -> for every product:
     * synthesize the per-pid df (item-level aggregated series + EXOG_COLS),
     * fit per-pid target/exog scalers on the train slice,
     * build per-pid training graphs via `neighbourhood_graph`,
     * call `make_single_windows` to materialize (y, graph, ts_seq) windows,
     * append to a global pool.
   Per-pid scalers and the last-lookback snapshot at val-end are cached in
   `contexts[pid]` for downstream inference.

2. `train_shared_gcn` -> one `GCNMLPForecaster` trained on the combined pool,
   with train/val split = per-pid first portion / per-pid last portion. The
   trained model is saved to a single .pth on disk and reused on rerun.

3. `infer_shared_gcn_lockstep` -> same lock-step semantic as
   `bulk_pipeline.infer_all_products_lockstep`, but every product points at
   the SAME `model` object.

4. `evaluate_all_products_metrics` from `bulk_pipeline` is reused for the
   metrics CSV.

Notes
-----
* Heterogeneity in per-pid magnitudes is normalized away by the per-pid
  MinMax scalers BEFORE the data hits the GCN, so the shared model sees a
  comparable [0,1] scale across products.
* No product-id embedding is added (would require changing
  `GCNMLPForecaster`). All product-specific information enters through the
  ego-graph + ts_seq + cal features.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from train import TrainConfig  # noqa: F401 (kept for parity)
from gnn_pyg import GCNMLPForecaster, generate_node_features
from gnndataset import (SingleGraphDataset, single_graph_collate,
                        make_single_windows)
from utils import (
    generate_exogenous_features,
    neighbourhood_graph,
    compute_similarities_1vsAll,
    compute_distances_1vsAll,
    parse_dynamic_exog_cols,
)
from gnninference import build_dynamic_graph_with_calculated_threshold
from bulk_pipeline import build_pid_df, _infer_metric_type


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class SharedGCNConfig:
    """Single fixed graph + training configuration used for the shared GCN."""
    metric: str = 'spearman'
    threshold: Optional[float] = 0.85
    percentile: Optional[float] = None
    window_size: int = 30
    step_size: int = 1
    enable_edges_within_star: bool = False
    enable_second_degree: bool = False
    use_residuals: bool = False
    # Training budget (one pass over the FULL combined pool).
    epochs: int = 30
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    # GCN architecture
    gcn_hidden_channels: int = 32
    gcn_out_channels: int = 64
    ts_proj_dim: int = 64
    mlp_hidden_sizes: tuple = (256, 64)
    dropout: float = 0.2

    def fingerprint(self) -> str:
        h = hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return h[:10]


# -----------------------------------------------------------------------------
# Stage 1: build training pool (per-pid graphs + windows -> concatenated)
# -----------------------------------------------------------------------------
def build_shared_training_data(
    df_wide_global: pd.DataFrame,
    product_minmax_scalers: dict,
    cat_labels_dict,
    seed: int,
    shared_cfg: SharedGCNConfig,
    lookback: int,
    val_size: int,
    train_size: int,
    forecast_horizon: int,
    target_col: str,
    exog_cols: list,
    date_col: str,
    node_features: list,
    cal_columns: list,
    include_cal_lookback: bool,
    skip_pids=None,
):
    """Return (pool, contexts).

    pool       : dict with keys
        'y_train' (N_tr, H, 1), 'g_train' (List[Data]), 'ts_train' (N_tr, L, 1+cal),
        'y_val',   'g_val',   'ts_val',
        'feature_dim', 'cal_dim'
    contexts   : dict[pid] -> per-pid state needed by infer:
        target_scaler, exog_scaler, target_window_scaled (L,),
        cal_window (L, cal_dim) or None, target_buffer_unscaled (list),
        future_exog_unscaled (H_test, cal_dim) or None
    """
    metric_type  = _infer_metric_type(shared_cfg.metric)
    compute_func = (compute_distances_1vsAll if metric_type == 'distance'
                    else compute_similarities_1vsAll)
    skip_pids    = set(skip_pids or [])
    pids         = [p for p in df_wide_global.index if p not in skip_pids]

    L            = df_wide_global.shape[1]
    test_start   = L - forecast_horizon
    val_start    = test_start - val_size
    train_start  = max(0, val_start - train_size)

    y_tr_list, g_tr_list, ts_tr_list = [], [], []
    y_vl_list, g_vl_list, ts_vl_list = [], [], []
    contexts: Dict = {}
    n_ok = n_skipped = 0
    t_start = time.time()

    feature_dim = None
    cal_dim_global = None

    print(f"[shared-gcn] Building training pool: pids={len(pids)} "
          f"(metric={shared_cfg.metric}, win={shared_cfg.window_size})")

    for j, pid in enumerate(pids, 1):
        # --- per-pid df ---
        df_pid = build_pid_df(pid, df_wide_global, date_col, target_col, exog_cols)
        target_full = df_pid[target_col].values.astype(np.float64)

        train_tgt = target_full[train_start:val_start]
        if (len(train_tgt) < lookback + 1
                or np.sum(np.abs(train_tgt)) == 0
                or pid not in product_minmax_scalers):
            n_skipped += 1
            continue

        # --- per-pid scalers ---
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_tgt.reshape(-1, 1))

        if exog_cols:
            exog_train_raw = df_pid[exog_cols].values[train_start:val_start]
            exog_val_raw   = df_pid[exog_cols].values[val_start:test_start]
            exog_test_raw  = df_pid[exog_cols].values[test_start:]
            exog_scaler    = MinMaxScaler()
            exog_scaler.fit(exog_train_raw)
            cols_idx = df_pid.columns.get_indexer(exog_cols)
            df_pid.iloc[train_start:val_start, cols_idx] = exog_scaler.transform(exog_train_raw)
            df_pid.iloc[val_start:test_start,  cols_idx] = exog_scaler.transform(exog_val_raw)
            df_pid.iloc[test_start:,            cols_idx] = exog_scaler.transform(exog_test_raw)
        else:
            exog_scaler   = None
            exog_test_raw = None

        # --- training graphs (over train slice) ---
        try:
            graphs_list, _thr = neighbourhood_graph(
                product_id=pid, df=df_wide_global, metric=shared_cfg.metric,
                metric_type=metric_type, window_size=shared_cfg.window_size,
                compute_func=compute_func,
                threshold=shared_cfg.threshold, percentile=shared_cfg.percentile,
                step_size=shared_cfg.step_size,
                cat_labels=cat_labels_dict,
                residuals=shared_cfg.use_residuals,
                enable_edges_within_star=shared_cfg.enable_edges_within_star,
                enable_second_degree=shared_cfg.enable_second_degree,
                train_end_idx=val_start,
                node_features=node_features,
                node_scalers=product_minmax_scalers,
            )
        except Exception as e:
            print(f"[shared-gcn] pid={pid} graph build failed: {e}; skipped")
            n_skipped += 1
            continue
        if not graphs_list:
            n_skipped += 1
            continue

        # --- scaled cols for window building ---
        cols     = [target_col] + (exog_cols if exog_cols else [])
        data     = df_pid[cols].values
        train_d  = data[:val_start].copy()
        val_d    = data[val_start:test_start].copy()
        train_d[:, 0:1] = target_scaler.transform(train_d[:, 0:1])
        val_d[:,   0:1] = target_scaler.transform(val_d[:,   0:1])

        cal_dim = train_d.shape[1] - 1
        if cal_dim_global is None:
            cal_dim_global = cal_dim
        elif cal_dim != cal_dim_global:
            # All pids must yield the same exog dimensionality; otherwise the
            # shared model can't be one tensor. Skip mismatched pids.
            n_skipped += 1
            continue

        gtr = graphs_list[:len(train_d)]
        gvl = (graphs_list[len(train_d): len(train_d) + len(val_d)]
               if len(graphs_list) >= len(train_d) + len(val_d)
               else graphs_list[len(train_d):])

        try:
            y_tr, g_tr, ts_tr = make_single_windows(
                train_d[:, 0:1], train_d[:, 1:] if cal_dim > 0 else np.zeros((len(train_d), 0), dtype=np.float32),
                lookback, 1, target_channel=0, graphs=gtr,
                graph_window_size=shared_cfg.window_size,
                include_cal_lookback=include_cal_lookback,
                node_features=node_features, cal_columns=cal_columns,
            )
            y_vl, g_vl, ts_vl = make_single_windows(
                val_d[:, 0:1], val_d[:, 1:] if cal_dim > 0 else np.zeros((len(val_d), 0), dtype=np.float32),
                lookback, 1, target_channel=0, graphs=gvl,
                graph_window_size=shared_cfg.window_size,
                include_cal_lookback=include_cal_lookback,
                node_features=node_features, cal_columns=cal_columns,
            )
        except Exception as e:
            print(f"[shared-gcn] pid={pid} window build failed: {e}; skipped")
            n_skipped += 1
            continue

        if feature_dim is None and len(g_tr) > 0:
            feature_dim = g_tr[0].x.shape[1]

        y_tr_list.append(y_tr); g_tr_list.extend(g_tr); ts_tr_list.append(ts_tr)
        y_vl_list.append(y_vl); g_vl_list.extend(g_vl); ts_vl_list.append(ts_vl)

        # --- per-pid inference context ---
        val_tgt_unsc = target_full[val_start:test_start]
        recent_tgt_unsc = val_tgt_unsc[-lookback:]
        recent_tgt_sc   = target_scaler.transform(
            recent_tgt_unsc.reshape(-1, 1)
        ).astype(np.float32).flatten()
        if exog_cols:
            recent_exog_sc = df_pid[exog_cols].values[
                val_start + len(val_tgt_unsc) - lookback : val_start + len(val_tgt_unsc)
            ].astype(np.float32)
        else:
            recent_exog_sc = None
        history_unsc_full = target_full[:test_start]

        contexts[pid] = dict(
            target_scaler=target_scaler,
            exog_scaler=exog_scaler,
            target_window_scaled=recent_tgt_sc,
            cal_window=recent_exog_sc,
            target_buffer_unscaled=list(history_unsc_full[-max(lookback, 28):]
                                        .astype(np.float64)),
            future_exog_unscaled=(exog_test_raw.astype(np.float64).copy()
                                   if exog_test_raw is not None else None),
        )
        n_ok += 1
        if j % 50 == 0 or j == len(pids):
            print(f"[shared-gcn] built {j}/{len(pids)} | ok={n_ok} skipped={n_skipped} "
                  f"({time.time() - t_start:.1f}s)", end='\r')

    print(f"\n[shared-gcn] Pool ready. ok={n_ok} skipped={n_skipped} "
          f"feature_dim={feature_dim} cal_dim={cal_dim_global}")

    if n_ok == 0 or feature_dim is None:
        raise RuntimeError("No usable training data for the shared GCN.")

    pool = dict(
        y_train  = np.concatenate(y_tr_list,  axis=0),
        g_train  = g_tr_list,
        ts_train = np.concatenate(ts_tr_list, axis=0),
        y_val    = np.concatenate(y_vl_list,  axis=0) if y_vl_list else np.zeros((0, 1, 1), dtype=np.float32),
        g_val    = g_vl_list,
        ts_val   = np.concatenate(ts_vl_list, axis=0) if ts_vl_list else np.zeros((0, lookback, 1 + (cal_dim_global or 0)), dtype=np.float32),
        feature_dim = feature_dim,
        cal_dim     = cal_dim_global or 0,
    )
    return pool, contexts


# -----------------------------------------------------------------------------
# Stage 2: train a single shared GCN on the combined pool
# -----------------------------------------------------------------------------
def train_shared_gcn(
    pool: dict,
    seed: int,
    shared_cfg: SharedGCNConfig,
    lookback: int,
    device,
    save_dir: str,
    loss_type: str = 'mse',
):
    os.makedirs(save_dir, exist_ok=True)
    fp = shared_cfg.fingerprint()
    model_path = os.path.join(save_dir, f'shared_gcn_{fp}.pth')

    cal_dim     = pool['cal_dim']
    feature_dim = pool['feature_dim']

    torch.manual_seed(seed)
    model = GCNMLPForecaster(
        in_channels=feature_dim,
        hidden_channels=shared_cfg.gcn_hidden_channels,
        out_channels=shared_cfg.gcn_out_channels,
        ts_input_size=lookback * (1 + cal_dim),
        mlp_hidden_sizes=list(shared_cfg.mlp_hidden_sizes),
        horizon=1,
        dropout=shared_cfg.dropout,
        ts_proj_dim=shared_cfg.ts_proj_dim,
    ).to(device)

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[shared-gcn] Loaded cached weights: {model_path}")
            return model.eval(), [], []
        except Exception as e:
            print(f"[shared-gcn] Cache load failed ({e}); retraining.")

    train_loader = DataLoader(
        SingleGraphDataset(pool['y_train'], pool['g_train'], pool['ts_train']),
        batch_size=shared_cfg.batch_size, shuffle=True,
        collate_fn=single_graph_collate,
    )
    val_loader = DataLoader(
        SingleGraphDataset(pool['y_val'], pool['g_val'], pool['ts_val']),
        batch_size=shared_cfg.batch_size, shuffle=False,
        collate_fn=single_graph_collate,
    ) if len(pool['y_val']) > 0 else None

    if loss_type == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=shared_cfg.lr,
                            weight_decay=shared_cfg.weight_decay)

    best_val   = float('inf')
    best_state = None
    best_epoch = 0
    patience_ct = 0
    train_losses, val_losses = [], []
    t_start = time.time()
    print(f"[shared-gcn] Training: train_windows={len(pool['y_train'])} "
          f"val_windows={len(pool['y_val'])} epochs={shared_cfg.epochs} fp={fp}")

    for epoch in range(1, shared_cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for yb, pyg_batch, ts_seq in train_loader:
            yb = yb.to(device); pyg_batch = pyg_batch.to(device); ts_seq = ts_seq.to(device)
            tgt_idx = pyg_batch.ptr[:-1]
            pred = model(pyg_batch, tgt_idx, ts_seq)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * yb.size(0)
        tr_loss /= max(1, len(train_loader.dataset))
        train_losses.append(tr_loss)

        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for yb, pyg_batch, ts_seq in val_loader:
                    yb = yb.to(device); pyg_batch = pyg_batch.to(device); ts_seq = ts_seq.to(device)
                    tgt_idx = pyg_batch.ptr[:-1]
                    pred = model(pyg_batch, tgt_idx, ts_seq)
                    vl += loss_fn(pred, yb).item() * yb.size(0)
            vl /= max(1, len(val_loader.dataset))
            val_losses.append(vl)
            improved = vl < best_val
            if improved:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_ct = 0
            else:
                patience_ct += 1
            print(f"[shared-gcn] epoch {epoch:3d}  train={tr_loss:.6f}  "
                  f"val={vl:.6f}{'  (best)' if improved else ''}")
            if shared_cfg.patience is not None and patience_ct >= shared_cfg.patience:
                print(f"[shared-gcn] Early stop @ epoch {epoch} "
                      f"(no improvement for {shared_cfg.patience})")
                break
        else:
            print(f"[shared-gcn] epoch {epoch:3d}  train={tr_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    print(f"[shared-gcn] Done. best_val={best_val:.6f} @ epoch {best_epoch} "
          f"(elapsed={time.time()-t_start:.1f}s) -> {model_path}")
    return model.eval(), train_losses, val_losses


# -----------------------------------------------------------------------------
# Stage 3: lock-step inference using the SAME shared model for every pid
# -----------------------------------------------------------------------------
def infer_shared_gcn_lockstep(
    shared_model,
    contexts: Dict,
    df_wide_global: pd.DataFrame,
    cat_labels_dict,
    shared_cfg: SharedGCNConfig,
    lookback: int,
    forecast_horizon: int,
    exog_cols: list,
    node_features: list,
    cal_columns: list,
    include_cal_lookback: bool,
    product_minmax_scalers: dict,
    device,
):
    """Same semantics as bulk_pipeline.infer_all_products_lockstep, but ONE
    model object processes every pid's ego-graph.
    """
    from torch_geometric.data import Batch

    metric_type  = _infer_metric_type(shared_cfg.metric)  # noqa: F841 (parity)
    df_wide_dyn  = df_wide_global.copy()
    all_dates    = list(df_wide_global.columns)
    L            = df_wide_global.shape[1]
    test_start   = L - forecast_horizon
    past_dates   = all_dates[:test_start]
    future_dates = all_dates[test_start:]
    horizon      = min(forecast_horizon, len(future_dates))

    preds_per_pid = {pid: [] for pid in contexts}

    # Feature dim is identical across pids (we enforced this at build time).
    any_ctx = next(iter(contexts.values()))
    cal_dim = 0 if any_ctx['cal_window'] is None else any_ctx['cal_window'].shape[1]
    _dummy_ts  = np.zeros(lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    feature_dim = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))

    if exog_cols:
        exog_cols = list(exog_cols)
        lag_cols_d, roll_cols_d = parse_dynamic_exog_cols(exog_cols)
        col_idx = {c: i for i, c in enumerate(exog_cols)}
    else:
        lag_cols_d, roll_cols_d, col_idx = {}, {}, {}

    pids_sorted = list(contexts.keys())
    print(f"[shared-infer] Lock-step inference: pids={len(pids_sorted)} horizon={horizon}")
    t_start = time.time()
    shared_model = shared_model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            f_date_i     = future_dates[i]
            t_idx_global = len(past_dates) - 1 + i
            date_start   = max(0, t_idx_global - shared_cfg.window_size + 1)
            window_dates = all_dates[date_start : t_idx_global + 1]

            step_outputs = {}

            for pid in pids_sorted:
                ctx = contexts[pid]
                target_preds_for_graph = (
                    df_wide_dyn.loc[pid, window_dates].values.astype(float).copy()
                )
                for f_idx, f_date in enumerate(future_dates[:i]):
                    if f_date in window_dates:
                        w_idx = list(window_dates).index(f_date)
                        target_preds_for_graph[w_idx] = preds_per_pid[pid][f_idx]

                try:
                    G_data = build_dynamic_graph_with_calculated_threshold(
                        target_id=pid,
                        target_preds=target_preds_for_graph,
                        df_wide=df_wide_dyn,
                        cat_labels=cat_labels_dict,
                        date_cols=window_dates,
                        metric=shared_cfg.metric,
                        fixed_threshold=shared_cfg.threshold,
                        enable_edges_within_star=shared_cfg.enable_edges_within_star,
                        enable_second_degree=shared_cfg.enable_second_degree,
                        node_features=node_features,
                        node_scalers=product_minmax_scalers,
                    )
                except Exception:
                    fallback = (preds_per_pid[pid][-1]
                                if len(preds_per_pid[pid]) > 0 else 0.0)
                    step_outputs[pid] = (None, fallback, None)
                    continue

                n_nodes = G_data.x.shape[0]
                x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

                if exog_cols and ctx['future_exog_unscaled'] is not None \
                        and ctx['exog_scaler'] is not None:
                    row_unscaled = ctx['future_exog_unscaled'][i].copy()
                    buf = ctx['target_buffer_unscaled']
                    for col, k in lag_cols_d.items():
                        idx = col_idx[col]
                        row_unscaled[idx] = (buf[-k] if k <= len(buf) else 0.0)
                    for col, w in roll_cols_d.items():
                        idx = col_idx[col]
                        vals = (buf[-w:] if w <= len(buf) else buf)
                        row_unscaled[idx] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                    cal_next = ctx['exog_scaler'].transform(
                        row_unscaled.reshape(1, -1)
                    ).ravel().astype(np.float32)
                elif cal_dim > 0:
                    cal_next = np.zeros(cal_dim, dtype=np.float32)
                else:
                    cal_next = None

                cal_lb = ctx['cal_window'] if include_cal_lookback else None
                x_new[0] = torch.tensor(
                    generate_node_features(
                        ctx['target_window_scaled'],
                        cal_next=cal_next, cal_lookback=cal_lb,
                        selected_features=node_features, cal_columns=cal_columns,
                    ),
                    dtype=torch.float32,
                )
                _dummy_cal_next = (np.zeros(cal_dim, dtype=np.float32)
                                   if cal_dim > 0 else None)
                for node_idx in range(1, n_nodes):
                    if hasattr(G_data, 'node_ts') and G_data.node_ts is not None:
                        neighbor_ts = G_data.node_ts[node_idx].numpy()
                    else:
                        orig_feat   = G_data.x[node_idx].numpy()
                        neighbor_ts = orig_feat[:shared_cfg.window_size]
                        n_min, n_max = neighbor_ts.min(), neighbor_ts.max()
                        n_range = n_max - n_min
                        neighbor_ts = ((neighbor_ts - n_min) / n_range
                                       if n_range > 1e-8
                                       else np.zeros_like(neighbor_ts))
                    x_new[node_idx] = torch.tensor(
                        generate_node_features(
                            neighbor_ts, cal_next=_dummy_cal_next,
                            selected_features=node_features,
                            is_neighbor=True, pad_ts_to=lookback,
                            cal_columns=cal_columns,
                        ),
                        dtype=torch.float32,
                    )
                G_data.x = x_new

                pyg_batch = Batch.from_data_list([G_data]).to(device)
                tgt_idx   = torch.tensor([0], dtype=torch.long, device=device)
                _ts_t = torch.tensor(ctx['target_window_scaled'],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                if cal_dim > 0 and ctx['cal_window'] is not None:
                    _ts_c = torch.tensor(ctx['cal_window'],
                                         dtype=torch.float32).unsqueeze(0)
                    ts_seq = torch.cat([_ts_t, _ts_c], dim=-1).to(device)
                else:
                    ts_seq = _ts_t.to(device)

                y_pred = shared_model(pyg_batch, tgt_idx, ts_seq)
                val_pred = float(y_pred.view(-1)[0].item())
                unscaled = float(ctx['target_scaler']
                                 .inverse_transform([[val_pred]])[0, 0])
                step_outputs[pid] = (val_pred, unscaled, cal_next)

            # --- atomic commit ---
            for pid in pids_sorted:
                val_pred, unscaled, cal_next = step_outputs[pid]
                ctx = contexts[pid]
                preds_per_pid[pid].append(unscaled)
                ctx['target_buffer_unscaled'].append(float(unscaled))
                if val_pred is not None:
                    ctx['target_window_scaled'] = np.roll(ctx['target_window_scaled'], -1)
                    ctx['target_window_scaled'][-1] = val_pred
                    if include_cal_lookback and ctx['cal_window'] is not None \
                            and cal_next is not None:
                        ctx['cal_window'] = np.roll(ctx['cal_window'], -1, axis=0)
                        ctx['cal_window'][-1] = cal_next
                if f_date_i in df_wide_dyn.columns:
                    df_wide_dyn.at[pid, f_date_i] = float(unscaled)

            if (i + 1) % max(1, horizon // 10) == 0 or i == horizon - 1:
                print(f"[shared-infer] step {i+1}/{horizon} "
                      f"({time.time() - t_start:.1f}s)")

    return ({pid: np.asarray(v, dtype=np.float64)
             for pid, v in preds_per_pid.items()},
            df_wide_dyn)
