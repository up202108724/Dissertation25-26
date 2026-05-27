"""Bulk per-product GCN training and lock-step inference.

This module implements the "one GCN per product" path:
  1. `train_gcn_mlps_for_all_products` trains (or loads from disk) ONE
     `train_gcn_mlp` model per product in `df_wide_global`. Each model is
     trained on that product's own item-level aggregated series (sum across
     stores) with exogenous features regenerated from the same series so the
     per-pid pipeline is self-consistent.
  2. `infer_all_products_lockstep` runs all per-product GCNs forward in a
     SHARED autoregressive loop: at each step every product computes its
     next-day forecast from the *current* state of `df_wide_dyn`; only AFTER
     every product has produced its step-i prediction is the dyn frame
     updated. The next step then sees the previous step's predictions from
     every product — i.e. the dynamic graph is rebuilt every step from a
     fully predicted future state.
  3. `evaluate_all_products_metrics` writes per-pid RMSE/MAE/BIAS/R2/POCID.

Caveats
-------
* Training ~len(products) GCN models is heavy. Disk-cache to
  `best_models/seed_{seed}/per_product_gcn/gcn_pid_{pid}.pth` and a config
  fingerprint guards correctness across runs.
* A single (metric, threshold/percentile, window, step, edges, 2nd) tuple is
  used for every product — no grid search at this scale.
* Per-pid exog is regenerated from the item-level aggregated series via
  `generate_exogenous_features`, so lag/rolling columns are consistent with
  what the GCN sees.
"""

from __future__ import annotations

import os
import csv
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports (resolved via main.py sys.path setup)
from train import TrainConfig, train_gcn_mlp
from utils import (
    generate_exogenous_features,
    neighbourhood_graph,
    compute_similarities_1vsAll,
    compute_distances_1vsAll,
    parse_dynamic_exog_cols,
)
from gnninference import build_dynamic_graph_with_calculated_threshold
from gnn_pyg import generate_node_features


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class BulkGCNConfig:
    """Single fixed graph + training configuration used for every product."""
    metric: str = 'spearman'
    threshold: Optional[float] = 0.85
    percentile: Optional[float] = None
    window_size: int = 30
    step_size: int = 1
    enable_edges_within_star: bool = False
    enable_second_degree: bool = False
    use_residuals: bool = False
    # Training budget per product (lower than the per-target grid search runs)
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32

    def fingerprint(self) -> str:
        h = hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return h[:10]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _infer_metric_type(metric: str) -> str:
    distance_metrics = {'euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                        'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                        'lorentzian', 'sbd', 'msm', 'edr', 'lcss'}
    return 'distance' if metric in distance_metrics else 'similarity'


def build_pid_df(pid, df_wide_global, date_col, target_col, exog_cols):
    """Synthesize a per-product DataFrame (with EXOG_COLS) from the
    item-level aggregated series in df_wide_global."""
    dates = pd.to_datetime(df_wide_global.columns)
    series = df_wide_global.loc[pid].values.astype(float)
    df_pid = pd.DataFrame({
        date_col:   dates,
        'item_id':  pid,
        'store_id': 0,
        target_col: series,
    })
    df_pid = generate_exogenous_features(
        df_pid, exog_cols=list(exog_cols),
        date_col=date_col, target_col=target_col,
        group_cols=['item_id'],
    )
    for c in exog_cols:
        if c not in df_pid.columns:
            df_pid[c] = 0.0
    return df_pid


# -----------------------------------------------------------------------------
# Training: one GCN per product
# -----------------------------------------------------------------------------
def train_gcn_mlps_for_all_products(
    df_wide_global,
    product_minmax_scalers,
    cat_labels_dict,
    seed,
    bulk_cfg: BulkGCNConfig,
    lookback,
    val_size,
    train_size,
    forecast_horizon,
    target_col,
    exog_cols,
    date_col,
    node_features,
    cal_columns,
    include_cal_lookback,
    hidden_sizes,
    gcn_hidden_channels,
    gcn_out_channels,
    ts_proj_dim,
    dropout,
    device,
    save_dir,
    skip_pids=None,
):
    """For every pid in df_wide_global, train (or load) a `train_gcn_mlp`
    model. Returns a dict[pid] -> context dict with everything needed to run
    inference later.
    """
    os.makedirs(save_dir, exist_ok=True)
    fp = bulk_cfg.fingerprint()
    metric_type   = _infer_metric_type(bulk_cfg.metric)
    compute_func  = (compute_distances_1vsAll if metric_type == 'distance'
                     else compute_similarities_1vsAll)

    skip_pids = set(skip_pids or [])
    pids      = [p for p in df_wide_global.index if p not in skip_pids]
    print(f"[bulk-gcn] Preparing {len(pids)} per-product GCN models "
          f"(metric={bulk_cfg.metric}, win={bulk_cfg.window_size}, "
          f"epochs={bulk_cfg.epochs}, fp={fp}, dir={save_dir})")

    contexts = {}
    n_loaded = n_trained = n_skipped = 0
    t_start  = time.time()

    L           = df_wide_global.shape[1]
    test_start  = L - forecast_horizon
    val_start   = test_start - val_size
    train_start = max(0, val_start - train_size)

    for j, pid in enumerate(pids, 1):
        # ---- 1. Build pid-level df with EXOG_COLS ----
        df_pid = build_pid_df(pid, df_wide_global, date_col, target_col, exog_cols)
        target_full = df_pid[target_col].values.astype(np.float64)

        train_tgt = target_full[train_start:val_start]
        if (len(train_tgt) < lookback + 1
                or np.sum(np.abs(train_tgt)) == 0
                or pid not in product_minmax_scalers):
            n_skipped += 1
            continue

        # ---- 2. Per-pid scalers ----
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
            exog_val_raw  = None

        # ---- 3. Train or load ----
        model_path = os.path.join(save_dir, f'gcn_pid_{pid}_{fp}.pth')
        cfg = TrainConfig(
            lookback=lookback, horizon=1, batch_size=bulk_cfg.batch_size,
            train_size=train_size, val_size=val_size,
            lr=bulk_cfg.lr, epochs=bulk_cfg.epochs,
            weight_decay=bulk_cfg.weight_decay, patience=bulk_cfg.patience,
            device=str(device),
        )

        # Build per-pid graphs over the train period (one set per pid).
        try:
            graphs_list, _thr = neighbourhood_graph(
                product_id=pid, df=df_wide_global, metric=bulk_cfg.metric,
                metric_type=metric_type, window_size=bulk_cfg.window_size,
                compute_func=compute_func,
                threshold=bulk_cfg.threshold, percentile=bulk_cfg.percentile,
                step_size=bulk_cfg.step_size,
                cat_labels=cat_labels_dict,
                residuals=bulk_cfg.use_residuals,
                enable_edges_within_star=bulk_cfg.enable_edges_within_star,
                enable_second_degree=bulk_cfg.enable_second_degree,
                train_end_idx=val_start,
                node_features=node_features,
                node_scalers=product_minmax_scalers,
            )
        except Exception as e:
            print(f"[bulk-gcn] pid={pid} graph build failed: {e}; skipped")
            n_skipped += 1
            continue
        if not graphs_list:
            n_skipped += 1
            continue

        torch.manual_seed(seed)
        try:
            model, _, _t_loss, _v_loss, _best = train_gcn_mlp(
                df=df_pid, cfg=cfg, seed=seed, loss_type='mse',
                product_id=f"bulk_{pid}", scaler=target_scaler, target_channel=0,
                hidden_sizes=hidden_sizes, target_col=target_col, exog_cols=exog_cols,
                graphs=graphs_list, test_size=forecast_horizon,
                graph_window_size=bulk_cfg.window_size,
                gcn_hidden_channels=gcn_hidden_channels,
                gcn_out_channels=gcn_out_channels,
                ts_proj_dim=ts_proj_dim, dropout=dropout,
                include_cal_lookback=include_cal_lookback,
                node_features=node_features, cal_columns=cal_columns,
            )
        except Exception as e:
            print(f"[bulk-gcn] pid={pid} train failed: {e}; skipped")
            n_skipped += 1
            continue

        # Load cached weights if present (skips re-training cost on reruns).
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                n_loaded += 1
            except Exception:
                torch.save(model.state_dict(), model_path)
                n_trained += 1
        else:
            torch.save(model.state_dict(), model_path)
            n_trained += 1

        model = model.to(device).eval()

        # ---- 4. Cache the per-pid context needed at inference ----
        # Build the rolling lookback window from the END of the val slice (i.e.
        # the last `lookback` observations BEFORE the test horizon).
        val_tgt_unsc = target_full[val_start:test_start]
        recent_target_unsc = val_tgt_unsc[-lookback:]
        recent_target_sc   = target_scaler.transform(
            recent_target_unsc.reshape(-1, 1)).astype(np.float32).flatten()

        if exog_cols:
            recent_exog_sc = df_pid[exog_cols].values[
                val_start + len(val_tgt_unsc) - lookback : val_start + len(val_tgt_unsc)
            ].astype(np.float32)
        else:
            recent_exog_sc = None

        history_unsc_full = target_full[:test_start]

        contexts[pid] = dict(
            model=model,
            target_scaler=target_scaler,
            exog_scaler=exog_scaler,
            target_window_scaled=recent_target_sc,             # (L,)
            cal_window=recent_exog_sc,                          # (L, cal_dim) or None
            target_buffer_unscaled=list(history_unsc_full[-max(lookback, 28):]
                                        .astype(np.float64)),
            future_exog_unscaled=(exog_test_raw.astype(np.float64).copy()
                                   if exog_test_raw is not None else None),
        )

        if j % 25 == 0 or j == len(pids):
            dt = time.time() - t_start
            print(f"[bulk-gcn] {j}/{len(pids)} | ready={len(contexts)} "
                  f"loaded={n_loaded} trained={n_trained} skipped={n_skipped} "
                  f"({dt:.1f}s)", end='\r')

    print(f"\n[bulk-gcn] Done. ready={len(contexts)} loaded={n_loaded} "
          f"trained={n_trained} skipped={n_skipped} "
          f"(elapsed={time.time()-t_start:.1f}s)")
    return contexts


# -----------------------------------------------------------------------------
# Lock-step inference: all products advance one step in parallel.
# -----------------------------------------------------------------------------
def infer_all_products_lockstep(
    contexts,
    df_wide_global,
    cat_labels_dict,
    bulk_cfg: BulkGCNConfig,
    lookback,
    forecast_horizon,
    val_size,
    exog_cols,
    node_features,
    cal_columns,
    include_cal_lookback,
    product_minmax_scalers,
    device,
):
    """Returns dict[pid] -> np.ndarray of length `forecast_horizon` (unscaled
    forecasts) and the final dyn frame for inspection.

    Semantics: at step i, every pid computes its prediction from the SAME
    df_wide_dyn snapshot (which already contains steps 0..i-1 predictions for
    all pids). Only after all pids are done is df_wide_dyn updated with step
    i predictions. Then i+1 starts.
    """
    from torch_geometric.data import Batch

    metric_type  = _infer_metric_type(bulk_cfg.metric)
    df_wide_dyn  = df_wide_global.copy()
    all_dates    = list(df_wide_global.columns)
    L            = df_wide_global.shape[1]
    test_start   = L - forecast_horizon
    past_dates   = all_dates[:test_start]
    future_dates = all_dates[test_start:]
    horizon      = min(forecast_horizon, len(future_dates))

    preds_per_pid = {pid: [] for pid in contexts}

    # Pre-compute feature_dim per pid (depends on cal_dim, lookback, node_features)
    feature_dims = {}
    for pid, ctx in contexts.items():
        cal_dim = 0 if ctx['cal_window'] is None else ctx['cal_window'].shape[1]
        _dummy_ts  = np.zeros(lookback, dtype=np.float32)
        _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
        _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                      if include_cal_lookback and cal_dim > 0 else None)
        feature_dims[pid] = len(generate_node_features(
            _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
            selected_features=node_features, cal_columns=cal_columns,
        ))

    # Dynamic exog metadata
    if exog_cols:
        exog_cols = list(exog_cols)
        lag_cols_d, roll_cols_d = parse_dynamic_exog_cols(exog_cols)
        col_idx = {c: i for i, c in enumerate(exog_cols)}
    else:
        lag_cols_d, roll_cols_d, col_idx = {}, {}, {}

    pids_sorted = list(contexts.keys())
    print(f"[bulk-infer] Lock-step inference: pids={len(pids_sorted)} horizon={horizon}")
    t_start = time.time()

    with torch.no_grad():
        for i in range(horizon):
            f_date_i     = future_dates[i]
            t_idx_global = len(past_dates) - 1 + i
            date_start   = max(0, t_idx_global - bulk_cfg.window_size + 1)
            window_dates = all_dates[date_start : t_idx_global + 1]

            # Stage per-step predictions, commit AFTER the pid loop.
            step_outputs = {}

            for pid in pids_sorted:
                ctx = contexts[pid]
                cal_dim = 0 if ctx['cal_window'] is None else ctx['cal_window'].shape[1]

                # Baseline target row for the graph; overwrite past predicted
                # cells with this pid's own predictions for steps 0..i-1.
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
                        metric=bulk_cfg.metric,
                        fixed_threshold=bulk_cfg.threshold,
                        enable_edges_within_star=bulk_cfg.enable_edges_within_star,
                        enable_second_degree=bulk_cfg.enable_second_degree,
                        node_features=node_features,
                        node_scalers=product_minmax_scalers,
                    )
                except Exception:
                    # If graph build fails for this pid this step, fall back
                    # to the previous prediction (or 0 at step 0).
                    fallback = (preds_per_pid[pid][-1]
                                if len(preds_per_pid[pid]) > 0 else 0.0)
                    step_outputs[pid] = (None, fallback, None)
                    continue

                n_nodes = G_data.x.shape[0]
                x_new   = torch.zeros((n_nodes, feature_dims[pid]),
                                      dtype=torch.float32)

                # ---- Build cal_next for this pid (dynamic exog if available) ----
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
                        cal_next=cal_next,
                        cal_lookback=cal_lb,
                        selected_features=node_features,
                        cal_columns=cal_columns,
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
                        neighbor_ts = orig_feat[:bulk_cfg.window_size]
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

                # ---- Forward pass for this pid ----
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

                y_pred = ctx['model'](pyg_batch, tgt_idx, ts_seq)
                val_pred = float(y_pred.view(-1)[0].item())
                unscaled = float(ctx['target_scaler']
                                 .inverse_transform([[val_pred]])[0, 0])
                step_outputs[pid] = (val_pred, unscaled, cal_next)

            # ---- COMMIT step i atomically across all pids ----
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
                dt = time.time() - t_start
                print(f"[bulk-infer] step {i+1}/{horizon} done ({dt:.1f}s)")

    return {pid: np.asarray(v, dtype=np.float64) for pid, v in preds_per_pid.items()}, df_wide_dyn


# -----------------------------------------------------------------------------
# Per-product evaluation CSV
# -----------------------------------------------------------------------------
def evaluate_all_products_metrics(
    forecasts,
    df_wide_global,
    forecast_horizon,
    seed,
    bulk_cfg: BulkGCNConfig,
    out_csv_path,
):
    L          = df_wide_global.shape[1]
    test_start = L - forecast_horizon
    header     = ["seed", "fp", "product_id", "n_test", "rmse", "mae", "bias", "r2", "pocid", "score"]
    fp         = bulk_cfg.fingerprint()
    file_exists = os.path.exists(out_csv_path)
    n_rows = 0
    with open(out_csv_path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if not file_exists:
            w.writerow(header)
        for pid, preds in forecasts.items():
            actual = df_wide_global.loc[pid].values[test_start:].astype(np.float64)
            preds  = np.asarray(preds, dtype=np.float64)
            n      = min(len(actual), len(preds))
            if n == 0:
                continue
            a, p = actual[:n], preds[:n]
            mask = ~np.isnan(a)
            a, p = a[mask], p[mask]
            if len(a) == 0:
                continue
            rmse = float(np.sqrt(mean_squared_error(a, p)))
            mae  = float(mean_absolute_error(a, p))
            bias = float(np.mean(p - a))
            try:
                r2 = float(r2_score(a, p))
            except Exception:
                r2 = float('nan')
            if len(a) > 1:
                d_a, d_p = np.diff(a), np.diff(p)
                pocid = float(((d_a * d_p) > 0).sum() / len(d_a))
            else:
                pocid = float('nan')
            score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
            w.writerow([seed, fp, pid, len(a), rmse, mae, bias, r2, pocid, score])
            n_rows += 1
    print(f"[bulk-eval] seed={seed} fp={fp}: wrote {n_rows} per-pid rows -> "
          f"{os.path.basename(out_csv_path)}")
