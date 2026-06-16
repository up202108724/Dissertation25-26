"""
Sequential scalability benchmark for the GCN / GAT / Graph2Vec assemble.

Goal
----
Measure, for each model variant, how long it takes to BUILD (graphs/embeddings)
and TRAIN over an increasing number of products — i.e. answer "how does each
variant scale with the catalogue size?".

It runs strictly sequentially (no subprocess workers) and VARIANT-GROUPED, as
requested: it runs one variant over the first N products, records the time,
then moves to the next variant — for N in PRODUCT_COUNTS = (1, 10, 20, 50, 100, 200, 500).

What is timed (per product, then accumulated):
    build_seconds : neighbourhood-graph construction (GCN/GAT) or Graph2Vec
                    embedding fit+infer (graph2vec_*).  ~0 for the baselines.
    train_seconds : the model's training loop (early-stopped, EPOCHS/PATIENCE
                    from the base module) — the headline "training time".
    total_seconds : build + train.

Recursive inference is intentionally NOT run here: this benchmark isolates the
cost that scales with training, exactly as asked ("record the training time").

The heavy lifting (data prep helpers, datasets, models, training functions, the
Graph2Vec embedding pipeline) is imported from the integrated runner module so
this file never duplicates that logic — it only re-orders the loops and times
them.

Output: ``scalability_results.csv`` + a printed table.
"""

from __future__ import annotations

import os
import csv
import time
import random
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# The integrated runner — import (does NOT run main(), which is __main__-guarded)
# and reuse all its constants, helpers, datasets, models and training functions.
import graph2vec_gcn_gat_assmble_parallelized_time_elapsed as base

# ── Benchmark configuration ──────────────────────────────────────────────────
PRODUCT_COUNTS = [1, 10, 20, 50, 100, 200, 500]
BENCH_SEED     = 42
# Variant run order (baselines first, then the graph variants), as requested.
BENCH_VARIANTS = [
    'lstm_baseline', 'mlp_baseline',
    'graph2vec_lstm', 'graph2vec_mlp',
    'gcn_lstm', 'gcn_mlp', 'gat_lstm', 'gat_mlp',
]
# Single similarity/threshold config taken from the runner's grid.
_CFG          = base.grid_configs[0]
METRIC        = _CFG['metric']
_THRESHOLDS   = _CFG.get('thresholds', [None])
IS_THRESHOLD  = _THRESHOLDS not in (None, [None])
PARAM_VAL     = (_THRESHOLDS[0] if IS_THRESHOLD else _CFG.get('percentiles', [None])[0])
WINDOW_SIZE   = base.window_sizes[0]
STEP_SIZE     = base.step_sizes[0]
ENABLE_EDGES  = base.enable_edges_opts[0]
ENABLE_2ND    = base.enable_second_degree_opts[0]
NODE_FEAT_MODE = base.node_feature_modes[0]
RESULTS_CSV   = os.path.join(base.SCRIPT_DIR, "scalability_results.csv")

_DISTANCE_METRICS = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                     'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                     'lorentzian', 'sbd', 'msm', 'edr', 'lcss']


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Shared, product-independent setup (mirrors the runner's main()) ──────────
def prepare_shared():
    print(f"Loading data from {base.DATA_PATH}...")
    df = pd.read_feather(base.DATA_PATH)
    if base.DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if base.DATE_COL in df.columns else df.reset_index()
    if df.index.name == base.DATE_COL:
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df[base.DATE_COL] = pd.to_datetime(df[base.DATE_COL])
    df = df.sort_values([base.DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = base.generate_exogenous_features(df, date_col=base.DATE_COL, exog_cols=base.EXOG_COLS)
    full_df = df.copy()

    # Consider ALL products of the dataset (not just the top subset); the sweep
    # caps at max(PRODUCT_COUNTS)=500 eligible products in main().
    products = (
        full_df[["item_id", "store_id"]]
        .drop_duplicates()
        .sort_values(["item_id", "store_id"])
        .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
        .tolist()
    )

    cat_labels_dict = (
        full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in full_df.columns else {}
    )
    df_wide_global = full_df.pivot_table(
        index='item_id', columns=base.DATE_COL, values=base.TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    Lcols = len(df_wide_global.columns)
    global_train_start_idx = max(0, Lcols - base.forecast_horizon - base.val_size - base.train_size)
    global_val_start_idx   = Lcols - base.forecast_horizon - base.val_size

    product_scalers = {}
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled = df_wide_global.copy()
    for item_id_iter in df_wide_global.index:
        z = StandardScaler()
        z.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        product_scalers[item_id_iter] = z
        df_wide_scaled.loc[item_id_iter] = z.transform(
            df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        ).flatten()

    return dict(
        full_df=full_df, products=products, cat_labels_dict=cat_labels_dict,
        df_wide_global=df_wide_global, df_wide_scaled=df_wide_scaled,
        product_scalers=product_scalers, global_val_start_idx=global_val_start_idx,
    )


# ── Per-product data slice (mirrors the runner's per-product block) ──────────
def prepare_product(shared, product_id, store_id):
    full_df = shared['full_df']
    df_p = (
        full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
        .sort_values(base.DATE_COL).reset_index(drop=True)
    )
    required = base.forecast_horizon + base.val_size + base.train_size
    if len(df_p) < required:
        return None

    test_start_idx  = len(df_p) - base.forecast_horizon
    val_start_idx   = test_start_idx - base.val_size
    train_start_idx = val_start_idx - base.train_size
    train_slice = slice(train_start_idx, val_start_idx)
    val_slice   = slice(val_start_idx, test_start_idx)

    train = df_p[base.TARGET_COL][train_slice].values
    val   = df_p[base.TARGET_COL][val_slice].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val.reshape(-1, 1)).flatten()

    exog_by_head = {}
    for head, cols in [('lstm', base.EXOG_COLS_LSTM), ('mlp', base.EXOG_COLS_MLP)]:
        if cols:
            sc = MinMaxScaler()
            exog_by_head[head] = dict(
                scaler=sc,
                train=sc.fit_transform(df_p[cols][train_slice].values),
                val=sc.transform(df_p[cols][val_slice].values),
            )
        else:
            exog_by_head[head] = dict(scaler=None, train=None, val=None)

    return dict(
        df_p=df_p, train=train, val=val, scaler=scaler,
        train_scaled=train_scaled, val_scaled=val_scaled,
        train_start_idx=train_start_idx, val_start_idx=val_start_idx,
        test_start_idx=test_start_idx, exog_by_head=exog_by_head,
    )


# ── Build the per-day aligned graphs/embeddings for one graph variant ────────
def build_graphs_for_variant(variant, shared, prod, product_id):
    """Returns (pyg_train, pyg_val, in_channels) for a graph-based variant."""
    metric_type = base.infer_metric_type(METRIC)
    current_df_wide = shared['df_wide_scaled'] if METRIC in _DISTANCE_METRICS else shared['df_wide_global']
    T_global = current_df_wide.shape[1]
    df_p = prod['df_p']
    product_offset = T_global - len(df_p)
    is_threshold = IS_THRESHOLD

    if variant.startswith('graph2vec'):
        graph_embeddings, _g2v_model, _csv, _bt, _et, _thr = base.load_or_generate_embeddings(
            product_id=product_id, metric=METRIC, metric_type=metric_type,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            threshold=PARAM_VAL if is_threshold else None,
            percentile=None if is_threshold else PARAM_VAL,
            dimensions=base.D_G, enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND, use_residuals=base.USE_RESIDUALS,
            model_type=base.MODEL_TYPE, seed=BENCH_SEED, df=current_df_wide,
            cat_labels=shared['cat_labels_dict'],
            train_end_idx=shared['global_val_start_idx'], save_embeddings=False,
        )
        pyg_windows = [
            base.embedding_to_single_node_data(graph_embeddings[i], target_label=product_id)
            for i in range(len(graph_embeddings))
        ]
    else:  # gcn_* / gat_*
        compute_func = (base.compute_distances_1vsAll if metric_type == 'distance'
                        else base.compute_similarities_1vsAll)
        nx_graphs, _thr = base.neighbourhood_graph(
            product_id=product_id, df=current_df_wide, metric=METRIC,
            metric_type=metric_type, window_size=WINDOW_SIZE, compute_func=compute_func,
            threshold=PARAM_VAL if is_threshold else None,
            percentile=None if is_threshold else PARAM_VAL,
            step_size=STEP_SIZE, cat_labels=shared['cat_labels_dict'], plot_dir=None,
            residuals=base.USE_RESIDUALS, enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND, train_end_idx=shared['global_val_start_idx'],
        )
        build_graphs_fn = (base.build_pyg_graphs_lstm if base._variant_head(variant) == 'lstm'
                           else base.build_pyg_graphs_mlp)
        pyg_windows = build_graphs_fn(
            nx_graphs, current_df_wide, product_id,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE, node_feature_mode=NODE_FEAT_MODE,
        )

    aligned = base._align_pyg_windows_to_timeline(
        pyg_windows, window_size=WINDOW_SIZE, step_size=STEP_SIZE, T=T_global)
    pyg_train = aligned[product_offset + prod['train_start_idx']:
                        product_offset + prod['val_start_idx']]
    pyg_val   = aligned[product_offset + prod['val_start_idx']:
                        product_offset + prod['test_start_idx']]
    in_channels = pyg_train[0].x.shape[1]
    return pyg_train, pyg_val, in_channels


# ── Train one (variant, product) and return (build_s, train_s) ───────────────
def run_one(variant, shared, product_id, store_id):
    prod = prepare_product(shared, product_id, store_id)
    if prod is None:
        return None

    device = base.device

    # ----- Baselines (graph-free) -----
    if variant == 'lstm_baseline':
        _seed_everything(BENCH_SEED)
        cols = base.EXOG_COLS_LSTM
        eb = prod['exog_by_head']['lstm']
        model = base.LSTMBaseline(
            input_size=1 + (len(cols) if cols else 0), hidden_size=base.HIDDEN_SIZE,
            num_layers=base.NUM_LAYERS, dropout=base.DROPOUT_LSTM,
        ).to(device)
        tr_ds = base.LSTMBaselineDataset(prod['train_scaled'], eb['train'] if cols else None, base.lookback_window)
        va_ds = base.LSTMBaselineDataset(prod['val_scaled'],   eb['val']   if cols else None, base.lookback_window)
        tr_ld = DataLoader(tr_ds, batch_size=base.BATCH_SIZE, shuffle=False)
        va_ld = DataLoader(va_ds, batch_size=base.BATCH_SIZE, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE_LSTM)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=base.PATIENCE // 3)
        t0 = time.time()
        base.train_lstm_baseline(
            epochs=base.EPOCHS, model=model, train_loader=tr_ld, val_loader=va_ld,
            exog_cols=cols, criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
            optimizer=opt, device=device, best_model_path=None, scheduler=sch, patience=base.PATIENCE,
        )
        return 0.0, time.time() - t0

    if variant == 'mlp_baseline':
        _seed_everything(BENCH_SEED)
        mlp_exog_scaler = base.ExogenousScaler(continuous_strategy='minmax')
        mlp_exog_scaler.fit(prod['df_p'][base.EXOG_COLS_MLP].iloc[
            slice(prod['train_start_idx'], prod['val_start_idx'])], base.EXOG_COLS_MLP)
        cfg = base.TrainConfig(
            lookback=base.lookback_window, horizon=1, batch_size=base.BATCH_SIZE,
            train_size=base.train_size, val_size=base.val_size, lr=base.LEARNING_RATE_MLP,
            dropout=base.DROPOUT_MLP, epochs=base.EPOCHS, patience=base.PATIENCE,
            hidden_sizes=base.MLP_HIDDEN_SIZES, device=str(device),
        )
        t0 = time.time()
        base.train_mlp_forecaster(
            df=prod['df_p'], cfg=cfg, seed=BENCH_SEED, loss_type='mse',
            product_id=f"{product_id}_{store_id}", scaler=prod['scaler'], target_channel=0,
            target_col=base.TARGET_COL, exog_cols=base.EXOG_COLS_MLP,
            test_size=base.forecast_horizon, exog_scaler=mlp_exog_scaler,
        )
        return 0.0, time.time() - t0

    # ----- Graph variants (gcn/gat/graph2vec × lstm/mlp) -----
    head = base._variant_head(variant)
    exog_cols = base.EXOG_COLS_LSTM if head == 'lstm' else base.EXOG_COLS_MLP
    eb = prod['exog_by_head'][head]

    t_build0 = time.time()
    pyg_train, pyg_val, in_channels = build_graphs_for_variant(variant, shared, prod, product_id)
    build_s = time.time() - t_build0

    if head == 'lstm':
        DatasetClass, collate_fn, train_fn = (
            base.GCN_LSTMTimeSeriesDataset, base.collate_pyg_ts_lstm, base.train_gcn_lstm_model)
    else:
        DatasetClass, collate_fn, train_fn = (
            base.GCNMLPTimeSeriesDataset, base.collate_pyg_ts_mlp, base.train_gcn_mlpmodel)

    pin = torch.cuda.is_available()
    tr_ds = DatasetClass(prod['train_scaled'], eb['train'] if exog_cols else None,
                         base.lookback_window, pyg_train, graph_window_size=WINDOW_SIZE)
    va_ds = DatasetClass(prod['val_scaled'], eb['val'] if exog_cols else None,
                         base.lookback_window, pyg_val, graph_window_size=WINDOW_SIZE)
    tr_ld = DataLoader(tr_ds, batch_size=base.BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=base.BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)

    ts_input_size = 1 + (len(exog_cols) if exog_cols else 0)
    _seed_everything(BENCH_SEED)
    model = base.build_forecaster(variant, False, in_channels, ts_input_size).to(device)
    model.ablate_z = False
    lr = base.LEARNING_RATE_LSTM if head == 'lstm' else base.LEARNING_RATE_MLP
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=base.PATIENCE // 3)

    t_train0 = time.time()
    train_fn(
        seed=BENCH_SEED, epochs=base.EPOCHS, model=model,
        train_loader=tr_ld, val_loader=va_ld,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=opt, device=device, best_model_path=None,
        scheduler=sch, patience=base.PATIENCE,
        diag_csv_path=None, diag_meta={},
    )
    return build_s, time.time() - t_train0


# ── Benchmark driver ─────────────────────────────────────────────────────────
def main():
    max_n = max(PRODUCT_COUNTS)
    shared = prepare_shared()

    # Keep only products with enough history, then take the first max_n so every
    # variant is timed on the SAME product set at each cutoff.
    eligible = []
    for (pid, sid) in shared['products']:
        if len(eligible) >= max_n:
            break
        if prepare_product(shared, pid, sid) is not None:
            eligible.append((pid, sid))
    print(f"Benchmarking on {len(eligible)} eligible products (requested up to {max_n}).")

    cutoffs = sorted(n for n in PRODUCT_COUNTS if n <= len(eligible))
    rows = []  # (variant, n_products, build_s, train_s, total_s, per_product_train_s)

    for variant in BENCH_VARIANTS:
        print(f"\n{'#'*70}\n# VARIANT: {variant}\n{'#'*70}")
        cum_build = cum_train = 0.0
        n_done = 0
        for i, (pid, sid) in enumerate(eligible, start=1):
            try:
                res = run_one(variant, shared, pid, sid)
            except Exception as e:  # one bad product must not kill the sweep
                print(f"  [{variant}] product {pid}/{sid} FAILED: {e}")
                traceback.print_exc()
                res = None
            if res is not None:
                b, t = res
                cum_build += b
                cum_train += t
                n_done += 1
                print(f"  [{variant}] {i}/{len(eligible)}  build={b:6.2f}s  train={t:7.2f}s"
                      f"  (cum train={cum_train:8.2f}s)")
            if i in cutoffs:
                per_prod = cum_train / n_done if n_done else float('nan')
                rows.append((variant, i, round(cum_build, 3), round(cum_train, 3),
                             round(cum_build + cum_train, 3), round(per_prod, 4)))
                print(f"  >>> CUTOFF N={i}: cum_build={cum_build:.2f}s  cum_train={cum_train:.2f}s  "
                      f"total={cum_build + cum_train:.2f}s")

    # ── Persist + print ──────────────────────────────────────────────────────
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_products", "build_seconds", "train_seconds",
                    "total_seconds", "mean_train_seconds_per_product"])
        w.writerows(rows)
    print(f"\nSaved scalability results to {RESULTS_CSV}")

    print(f"\n{'variant':<16}{'N':>6}{'build_s':>12}{'train_s':>12}{'total_s':>12}{'train/prod':>12}")
    for variant, n, b, t, tot, per in rows:
        print(f"{variant:<16}{n:>6}{b:>12.2f}{t:>12.2f}{tot:>12.2f}{per:>12.3f}")


if __name__ == '__main__':
    main()
