"""
graph2vec_embedding_main.py
---------------------------
End-to-end Graph2Vec + LSTM / MLP experiment with a TRAINABLE embedding layer.

Key idea
--------
Pre-trained Graph2Vec embeddings are loaded into an ``nn.Embedding`` table
(one row per timestep), which is then fine-tuned end-to-end during training
alongside the LSTM or MLP head.  At inference time the model looks up the
(already fine-tuned) embedding for each window position by absolute timestep
index — no dynamic graph building at test time is needed.

Both model types are evaluated in sequence for every (metric, threshold/pct,
window) combination, producing a CSV row for every experiment.
"""

import os
import random
import sys
import itertools
import pickle
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------- #
# Path setup (mirrors graph2vec_lstm copy.py)                             #
# ---------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                            # local files
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))   # Dissertation25-26/
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))         # LSTM with Embedding Layer/

# Local (this folder)
from models_with_embedding import LSTMWithEmbedding, MLPWithEmbedding
from dataset_with_embedding import TimeSeriesEmbIdxDataset
from train_with_embedding import train_model_with_embedding
from inference_with_embedding import inference_with_embedding

# Original codebase (reuse unchanged helpers)
from GNN.Graph2vec_FixedThreshold.LSTM.plots import plot_results
from GNN.Graph2vec_FixedThreshold.LSTM.generate_graph2vecwithadaptativethreshold import (
    load_or_generate_embeddings,
    infer_metric_type,
)
from model_utils.utils import generate_exogenous_features, compute_metrics

# ---------------------------------------------------------------------- #
# Constants                                                               #
# ---------------------------------------------------------------------- #
DATA_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather')
)
DATE_COL   = 'date'
TARGET_COL = 'value'

SEEDS = [42]

PRODUCTS_TO_TEST = [
    (26008,  6269),
    (907969, 6269),
    (907967, 6269),
    (213626, 6269),
    (911753, 6269),
]

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3",
    "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3",
    "is_post_holiday_7",
    "is_bridge_day",
]

grid_configs = [
    {'metric': 'spearman',
     'thresholds': [round(t, 3) for t in np.arange(0.87, 0.96, 0.01)]},
]

window_sizes               = [15]
step_sizes                 = [1]
enable_edges_opts          = [True]
enable_second_degree_opts  = [False]

# Model families to evaluate
MODEL_TYPES = ['lstm', 'mlp']   # 'lstm', 'mlp', or both

# Hyper-parameters
EPOCHS        = 1000
PATIENCE      = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE   = 32          # LSTM hidden size
NUM_LAYERS    = 1
DROPOUT       = 0.0
MLP_HIDDEN    = (128, 64)   # MLP hidden layer widths
BATCH_SIZE    = 32
EMB_DIM       = 20          # Graph2Vec embedding dimension (must match saved embeddings)

# Misc
USE_RESIDUALS  = False
MODEL_TYPE_STR = 'ridge'
SAVE_MODELS    = True
SAVE_PLOTS     = True
SAVE_EMBEDDINGS = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================================== #
#  Helpers                                                                #
# ====================================================================== #

def _build_full_aligned_embeddings(
    graph_embeddings: np.ndarray,
    window_size: int,
    target_length: int,
) -> np.ndarray:
    """
    Build aligned_embeddings of length exactly ``target_length``.

    The first ``window_size - 1`` rows are zero-padded (no graph exists yet).
    If graph_embeddings is shorter than needed, the last known embedding is
    repeated to fill the remaining positions.
    """
    emb_dim = graph_embeddings.shape[1]
    padding = np.zeros((window_size - 1, emb_dim), dtype=np.float32)
    aligned = np.vstack([padding, graph_embeddings.astype(np.float32)])

    if len(aligned) < target_length:
        n_missing = target_length - len(aligned)
        extension = np.tile(aligned[-1:], (n_missing, 1))
        aligned = np.vstack([aligned, extension])

    return aligned[:target_length]   # trim if somehow longer


def _model_forward_name(model_type: str) -> str:
    return 'LSTM' if model_type == 'lstm' else 'MLP'


def _build_model(
    model_type: str,
    ts_input_size: int,
    emb_dim: int,
    seq_length: int,
    num_graphs: int,
    pretrained_weights: torch.Tensor,
):
    if model_type == 'lstm':
        return LSTMWithEmbedding(
            ts_input_size=ts_input_size,
            emb_dim=emb_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_graphs=num_graphs,
            pretrained_embeddings=pretrained_weights,
        )
    else:
        return MLPWithEmbedding(
            seq_length=seq_length,
            ts_input_size=ts_input_size,
            emb_dim=emb_dim,
            hidden_sizes=MLP_HIDDEN,
            dropout=DROPOUT,
            num_graphs=num_graphs,
            pretrained_embeddings=pretrained_weights,
        )


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #

def main():
    print(f"Loading data from {DATA_PATH}...")
    df_global = pd.read_feather(DATA_PATH)

    if df_global.index.name == DATE_COL:
        df_global = df_global.reset_index(drop=True)
    df_global = df_global.reset_index(drop=True)
    df_global[DATE_COL] = pd.to_datetime(df_global[DATE_COL])
    df_global = df_global.sort_values(
        [DATE_COL, 'item_id', 'store_id']
    ).reset_index(drop=True)
    df_global = generate_exogenous_features(
        df_global, date_col=DATE_COL, exog_cols=EXOG_COLS
    )
    full_df = df_global.copy()

    # Wide format + per-product row-wise scaling (train+val only, no leakage)
    cat_labels_dict = (
        full_df.drop_duplicates('item_id')
               .set_index('item_id')['cat_label']
               .to_dict()
        if 'cat_label' in full_df.columns else {}
    )
    df_wide_global = full_df.pivot_table(
        index='item_id', columns=DATE_COL,
        values=TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = (
        pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')
    )

    global_horizon      = 152
    train_val_end_idx   = len(df_wide_global.columns) - global_horizon
    if train_val_end_idx > 0:
        subset = df_wide_global.iloc[:, :train_val_end_idx].values
        row_min   = subset.min(axis=1, keepdims=True)
        row_max   = subset.max(axis=1, keepdims=True)
        row_range = np.where(row_max - row_min == 0, 1, row_max - row_min)
        df_wide_global.iloc[:, :] = (df_wide_global.values - row_min) / row_range

    # CSV for aggregated results
    results_csv = os.path.join(SCRIPT_DIR, 'embedding_results.csv')

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PRODUCT {product_id}  STORE {store_id}")
        print(f"{'='*80}")

        df = (
            full_df[
                (full_df['item_id'] == product_id)
                & (full_df['store_id'] == store_id)
            ]
            .sort_values(DATE_COL)
            .reset_index(drop=True)
        )

        forecast_horizon = 152
        seq_length       = 30
        train_size       = 455
        val_size         = 154

        required = forecast_horizon + val_size + train_size
        if len(df) < required:
            print(f"Skipping: {len(df)} rows < {required} required.")
            continue

        test_start_idx  = len(df) - forecast_horizon
        val_start_idx   = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size
        T_total         = len(df)

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice   = slice(val_start_idx,   test_start_idx)
        test_slice  = slice(test_start_idx,  None)

        # Target splits
        train_raw = df[TARGET_COL][train_slice].values
        val_raw   = df[TARGET_COL][val_slice].values
        test_raw  = df[TARGET_COL][test_slice].values

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
        val_scaled   = scaler.transform(val_raw.reshape(-1, 1)).flatten()
        test_scaled  = scaler.transform(test_raw.reshape(-1, 1)).flatten()

        # Exog splits
        if EXOG_COLS:
            exog_train_raw    = df[EXOG_COLS][train_slice].values
            exog_val_raw      = df[EXOG_COLS][val_slice].values
            exog_test_raw_arr = df[EXOG_COLS][test_slice].values
            exog_scaler       = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train_raw)
            exog_val_scaled   = exog_scaler.transform(exog_val_raw)
            exog_test_scaled  = exog_scaler.transform(exog_test_raw_arr)
        else:
            exog_train_scaled = exog_val_scaled = exog_test_scaled = None
            exog_scaler = None

        ts_input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0)

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- SEED {seed} ---")

            plots_dir      = os.path.join(SCRIPT_DIR, 'grid_search_plots', f'seed_{seed}')
            best_models_dir = os.path.join(SCRIPT_DIR, 'best_models', f'seed_{seed}')
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(best_models_dir, exist_ok=True)

            # ---------------------------------------------------------- #
            # Iterate over (metric, threshold, window, edges, …)          #
            # ---------------------------------------------------------- #
            for config in grid_configs:
                metric     = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])
                is_threshold_mode = (
                    thresholds is not None and thresholds != [None]
                )
                params   = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(
                    params, window_sizes, step_sizes,
                    enable_edges_opts, enable_second_degree_opts,
                )

                for param_val, window_sz, step_sz, enable_edges, enable_2nd in iterator:
                    current_threshold  = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    print(f"\n{'='*60}")
                    p_str = (f"threshold={param_val}" if is_threshold_mode
                             else f"percentile={param_val}")
                    print(
                        f"metric={metric}  {p_str}  window={window_sz}"
                        f"  edges={enable_edges}  2nd={enable_2nd}"
                    )
                    print(f"{'='*60}")

                    # -------------------------------------------------- #
                    # Load / generate Graph2Vec embeddings                 #
                    # -------------------------------------------------- #
                    metric_type = infer_metric_type(metric)
                    (
                        graph_embeddings,
                        graph2vec_model,
                        csv_path,
                        _build_t,
                        _emb_t,
                        fixed_threshold,
                    ) = load_or_generate_embeddings(
                        product_id=product_id,
                        metric=metric,
                        metric_type=metric_type,
                        window_size=window_sz,
                        step_size=step_sz,
                        threshold=current_threshold,
                        percentile=current_percentile,
                        dimensions=EMB_DIM,
                        use_residuals=USE_RESIDUALS,
                        model_type=MODEL_TYPE_STR,
                        seed=seed,
                        train_end_idx=val_start_idx,
                        df=df_wide_global,
                        cat_labels=cat_labels_dict,
                        save_embeddings=SAVE_EMBEDDINGS,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_2nd,
                    )

                    emb_dim_actual = (
                        graph_embeddings.shape[1]
                        if graph_embeddings.ndim > 1 else 1
                    )
                    print(f"Embedding dim={emb_dim_actual}  "
                          f"resolved_threshold={fixed_threshold}")

                    # Build embedding array for the FULL time series
                    aligned_emb = _build_full_aligned_embeddings(
                        graph_embeddings, window_sz, T_total
                    )
                    # shape: (T_total, emb_dim_actual)
                    emb_table_size = len(aligned_emb)

                    pretrained_tensor = torch.from_numpy(
                        aligned_emb.astype(np.float32)
                    )   # (T_total, emb_dim_actual)

                    # -------------------------------------------------- #
                    # Evaluate each model type                             #
                    # -------------------------------------------------- #
                    for model_type_key in MODEL_TYPES:
                        model_name = _model_forward_name(model_type_key)
                        print(f"\n  >> Model: {model_name}")

                        # Re-set seeds for reproducibility across model types
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        # Datasets
                        train_dataset = TimeSeriesEmbIdxDataset(
                            target_data    = train_scaled,
                            exog_data      = exog_train_scaled if EXOG_COLS else None,
                            seq_length     = seq_length,
                            idx_offset     = train_start_idx,
                            emb_table_size = emb_table_size,
                        )
                        val_dataset = TimeSeriesEmbIdxDataset(
                            target_data    = val_scaled,
                            exog_data      = exog_val_scaled if EXOG_COLS else None,
                            seq_length     = seq_length,
                            idx_offset     = val_start_idx,
                            emb_table_size = emb_table_size,
                        )
                        use_pin = torch.cuda.is_available()
                        train_loader = DataLoader(
                            train_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=use_pin,
                        )
                        val_loader = DataLoader(
                            val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=use_pin,
                        )

                        # Model
                        model = _build_model(
                            model_type_key,
                            ts_input_size  = ts_input_size,
                            emb_dim        = emb_dim_actual,
                            seq_length     = seq_length,
                            num_graphs     = emb_table_size,
                            pretrained_weights = pretrained_tensor,
                        ).to(device)

                        criterion  = nn.MSELoss()
                        criterion2 = nn.MSELoss()
                        optimizer  = torch.optim.Adam(
                            model.parameters(), lr=LEARNING_RATE
                        )
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.5,
                            patience=PATIENCE // 3,
                        )

                        # Checkpoint paths
                        if is_threshold_mode:
                            dir_label = f"th{current_threshold}"
                        else:
                            dir_label = f"pct{current_percentile}"

                        ckpt_dir = os.path.join(
                            best_models_dir, str(window_sz), str(step_sz),
                            metric, dir_label, model_type_key,
                        )
                        os.makedirs(ckpt_dir, exist_ok=True)

                        if csv_path:
                            base = (
                                os.path.basename(csv_path)
                                .replace('embeddings_',
                                         f'best_{model_name.lower()}_{product_id}_')
                                .replace('.csv', f'_seed{seed}.pth')
                            )
                        else:
                            base = (
                                f"best_{model_name.lower()}_{product_id}"
                                f"_{metric}_{dir_label}_seed{seed}.pth"
                            )

                        best_model_path = os.path.join(ckpt_dir, base)
                        history_path    = best_model_path.replace('.pth', '_history.pkl')

                        # Train or load
                        if (
                            SAVE_MODELS
                            and os.path.exists(best_model_path)
                            and os.path.exists(history_path)
                        ):
                            print(f"  Loading checkpoint: {best_model_path}")
                            model.load_state_dict(
                                torch.load(best_model_path, map_location=device)
                            )
                            with open(history_path, 'rb') as fh:
                                hist = pickle.load(fh)
                            train_losses = hist['train_losses']
                            val_losses   = hist['val_losses']
                        else:
                            print("  Training new model...")
                            (
                                model,
                                train_losses,
                                val_losses,
                                best_epoch,
                                train_time,
                            ) = train_model_with_embedding(
                                seed           = seed,
                                epochs         = EPOCHS,
                                model          = model,
                                train_loader   = train_loader,
                                val_loader     = val_loader,
                                criterion      = criterion,
                                criterion2     = criterion2,
                                optimizer      = optimizer,
                                device         = device,
                                best_model_path = best_model_path if SAVE_MODELS else None,
                                scheduler      = scheduler,
                                patience       = PATIENCE,
                            )
                            if SAVE_MODELS:
                                with open(history_path, 'wb') as fh:
                                    pickle.dump(
                                        {
                                            'train_losses': train_losses,
                                            'val_losses':   val_losses,
                                            'best_epoch':   best_epoch,
                                            'train_time':   train_time,
                                        },
                                        fh,
                                    )

                        if SAVE_MODELS and os.path.exists(best_model_path):
                            model.load_state_dict(
                                torch.load(best_model_path, map_location=device)
                            )

                        # Inference
                        forecast, inference_time = inference_with_embedding(
                            model           = model,
                            device          = device,
                            seq_length      = seq_length,
                            forecast_window = forecast_horizon,
                            test_start_idx  = test_start_idx,
                            val_scaled      = val_scaled,
                            exog_val_scaled = exog_val_scaled,
                            exog_test_scaled = exog_test_scaled,
                            test_scaled     = test_scaled,
                            scaler          = scaler,
                            exog_cols       = EXOG_COLS,
                            emb_table_size  = emb_table_size,
                            warmup_steps    = window_sz,
                        )

                        # Metrics
                        valid_mask   = ~np.isnan(forecast)
                        valid_test   = test_raw[valid_mask]
                        valid_fcast  = forecast[valid_mask]

                        rmse = mae = bias = r2 = pocid = None
                        try:
                            rmse, mae, bias, r2, pocid = compute_metrics(
                                valid_test, valid_fcast
                            )
                        except Exception:
                            if len(valid_test) > 0:
                                rmse  = float(np.sqrt(mean_squared_error(valid_test, valid_fcast)))
                                mae   = float(mean_absolute_error(valid_test, valid_fcast))
                                bias  = float(np.mean(valid_fcast - valid_test))
                                r2    = float(r2_score(valid_test, valid_fcast))

                        print(
                            f"  {model_name} → RMSE={rmse:.4f}  MAE={mae:.4f}"
                            f"  Bias={bias:.4f}  R²={r2:.4f}"
                        )

                        # Persist results to CSV
                        file_exists = os.path.exists(results_csv)
                        with open(results_csv, 'a', newline='') as csvf:
                            writer = csv.writer(csvf)
                            if not file_exists:
                                writer.writerow([
                                    'product_id', 'store_id', 'seed',
                                    'model_type', 'metric', 'window_size',
                                    'step_size', 'threshold', 'percentile',
                                    'enable_edges', 'enable_second_degree',
                                    'rmse', 'mae', 'bias', 'r2_score', 'pocid',
                                ])
                            writer.writerow([
                                product_id, store_id, seed,
                                model_type_key, metric, window_sz, step_sz,
                                current_threshold  if is_threshold_mode else '',
                                current_percentile if not is_threshold_mode else '',
                                enable_edges, enable_2nd,
                                rmse, mae, bias, r2, pocid,
                            ])

                        # Optional plot
                        if SAVE_PLOTS:
                            train_idx = df[DATE_COL][train_slice].values
                            val_idx   = df[DATE_COL][val_slice].values
                            test_idx  = df[DATE_COL][test_slice].values

                            p_str_label = (
                                f"th:{current_threshold}"
                                if is_threshold_mode
                                else f"pct:{current_percentile}"
                            )
                            label = (
                                f"{p_str_label}|w:{window_sz}"
                                f"|{model_name}"
                            )
                            sub_dir = os.path.join(
                                plots_dir, 'embedding', metric,
                                f'window_{window_sz}', f'step_{step_sz}',
                                model_type_key,
                                f'item_{product_id}',
                            )
                            os.makedirs(sub_dir, exist_ok=True)
                            plot_path = os.path.join(
                                sub_dir,
                                f"item_{product_id}_store_{store_id}"
                                f"_{metric}_{dir_label}_{model_type_key}"
                                f"_seed{seed}.html",
                            )
                            try:
                                plot_results(
                                    train_raw, val_raw, test_raw,
                                    {label: forecast},
                                    train_idx, val_idx, test_idx,
                                    {label: train_losses},
                                    {label: val_losses},
                                    metric=metric,
                                    embedding_strategy=f'graph2vec+{model_name}Embedding',
                                    window_size=window_sz, step_size=step_sz,
                                    threshold=(fixed_threshold
                                               if is_threshold_mode else None),
                                    percentile=(current_percentile
                                                if not is_threshold_mode else None),
                                    seed=seed,
                                    target_col=TARGET_COL,
                                    title=(
                                        f'Graph2Vec+{model_name} (Trainable Emb) '
                                        f'Item={product_id}'
                                    ),
                                    save_path=plot_path,
                                    rmse={label: rmse},
                                    mae={label: mae},
                                    bias={label: bias},
                                    score={label: r2},
                                    pocid={label: pocid},
                                )
                            except Exception as exc:
                                print(f"  Plot failed: {exc}")

    print("\nDone.  Results saved to:", results_csv)


if __name__ == '__main__':
    main()
