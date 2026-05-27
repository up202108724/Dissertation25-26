import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add parent DynamicSimilarities dir so lstm.py / mlp.py can be imported
_parent_dir = os.path.normpath(os.path.join(script_dir, '..', '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
# Ensure THIS folder takes precedence over the shared parent (which also
# contains an older mlp.py / utils.py). Insert last so it ends up at index 0.
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
from lstm import LSTM, TimeSeriesDataset, train_lstm, recursive_inference_lstm
from mlp import train_mlp_forecaster, recursive_inference_mlp, recursive_inference_mlp_dynamic_exog

from plots import plot_results
from utils import generate_exogenous_features
from train import TrainConfig, train_gcn_mlp
from gnninference import recursive_inference_gcn_mlp
from utils import compute_distances_1vsAll, compute_similarities_1vsAll, neighbourhood_graph
import itertools
import csv
import hashlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from lstm import LSTM, TimeSeriesDataset, train_lstm, recursive_inference_lstm
from mlp import (
    MLPForecaster, WindowDataset, train_mlp_forecaster,
    recursive_inference_mlp, recursive_inference_mlp_dynamic_exog,
    make_windows,
)
#from GNN.DynamicSimilarities.utils import infer_metric_type

def infer_metric_type(metric):
    distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
    if metric in distance_metrics:
        return 'distance'
    else:
        return 'similarity'
    
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(script_dir, '..', '..', '..', '..', 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'

train_size = 455
val_size = 153
forecast_horizon = 153
lookback_window = 30

NODE_FEATURES = ['ts']
EXOG_COLS = [
    "dom_sin","dom_cos", "wom_sin", "wom_cos",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
    "lag_1", "lag_7", "lag_14","lag_28",
    "rolling_mean_excl_3", 
    #"rolling_mean_excl_5",
    "rolling_mean_excl_7", "rolling_mean_excl_14", "rolling_mean_excl_28",
    "month_sin", "month_cos", "quarter",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]

batch_size = 32
hidden_sizes = (256,64)
dropout = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
seeds = [42]
#seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed
grid_configs = [
    # Distâncias Robustas e Lock-step
    #{'metric': 'cid', 'percentiles': [0.5, 1, 2]},
    #{'metric': 'amplitude_offset', 'percentiles': [0.5, 1, 2]},
    {'metric': 'spearman', 'thresholds': [0.75, 0.82, 0.85, 0.88, 0.91]},
    #{'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(1.3, 2.2, 0.2)]},
    # Distâncias Robustas e Lock-step
    #{'metric': 'amplitude_offset', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.5, 0.01)]},
]


SEEDS = [42]
window_sizes = [lookback_window]     
step_sizes = [1]
enable_edges_opts = [True]
enable_second_degree_opts = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
MODEL_TYPE = 'ridge'
loss_type = 'huber'
PATIENCE = 150
SAVE_MODELS = False
SAVE_PLOTS = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
INCLUDE_CAL_LOOKBACK = True   # include full lookback calendar as target-node features
SELECTED_NODE_FEATURES = NODE_FEATURES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRODUCTS_TO_TEST = [
  (26008, 6269),
  #(907969, 6269),
  #(907967, 6269),
  (213626, 6269),
  (911753,6269)
]


# Per-product local neighbor MLP settings ------------------------------------
# When USE_NEIGHBOR_LOCAL_MLPS is True, ONE small univariate MLP is trained
# per product in df_wide_global. At inference, instead of reading the
# neighbor's REAL future value from df_wide (which is leakage), each neighbor
# is fed its OWN local-MLP prediction for the current step. Past dates in the
# graph window still come from the real history.
USE_NEIGHBOR_LOCAL_MLPS    = True
NEIGHBOR_MLP_EPOCHS        = 100
NEIGHBOR_MLP_HIDDEN_SIZES  = (64, 32)
NEIGHBOR_MLP_BATCH_SIZE    = 32
NEIGHBOR_MLP_LR            = 1e-3
NEIGHBOR_MLP_WEIGHT_DECAY  = 1e-4


def train_neighbor_local_mlps(
    df_wide_global,
    global_val_start_idx,
    product_minmax_scalers,
    lookback,
    val_size,
    epochs,
    seed,
    device,
    save_dir,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_sizes=(64, 32),
    batch_size=32,
    skip_pids=None,
):
    """Train ONE univariate MLP per product in df_wide_global.

    Each model forecasts only its product's target value (no exogenous
    features) so it can be invoked cheaply per-step during graph construction.
    Models are cached on disk; subsequent calls reload instead of retraining.

    Returns
    -------
    models           : dict[pid] -> torch.nn.Module (on `device`, in eval mode)
    histories_unsc   : dict[pid] -> np.ndarray (full unscaled target series up
                                                 to `global_val_start_idx`,
                                                 used as the initial roll buffer)
    """
    os.makedirs(save_dir, exist_ok=True)
    skip_pids = set(skip_pids or [])

    models, histories_unsc = {}, {}
    pids = [p for p in df_wide_global.index if p not in skip_pids]
    print(f"[neighbor-local] Preparing {len(pids)} per-product local MLPs "
          f"(lookback={lookback}, epochs={epochs}, dir={save_dir})...")

    n_loaded, n_trained, n_skipped = 0, 0, 0
    for j, pid in enumerate(pids, 1):
        series_full = df_wide_global.loc[pid].values.astype(np.float32)
        train_unsc  = series_full[:global_val_start_idx]
        if np.sum(np.abs(train_unsc)) == 0:
            n_skipped += 1
            continue  # silent product

        if pid not in product_minmax_scalers:
            n_skipped += 1
            continue
        scaler_pid = product_minmax_scalers[pid]

        train_scaled = scaler_pid.transform(train_unsc.reshape(-1, 1)).astype(np.float32)
        # Validation slice: take the last `lookback` train points + the val window
        val_lo = max(0, global_val_start_idx - lookback)
        val_hi = global_val_start_idx + val_size
        val_unsc    = series_full[val_lo:val_hi].reshape(-1, 1)
        val_scaled  = scaler_pid.transform(val_unsc).astype(np.float32)

        try:
            X_tr, y_tr = make_windows(train_scaled, lookback, 1, target_channel=0)
        except ValueError:
            n_skipped += 1
            continue
        try:
            X_va, y_va = make_windows(val_scaled,   lookback, 1, target_channel=0)
        except ValueError:
            X_va = np.zeros((0, lookback, 1), dtype=np.float32)
            y_va = np.zeros((0, 1, 1),        dtype=np.float32)

        if X_tr.shape[0] == 0:
            n_skipped += 1
            continue

        model_path = os.path.join(save_dir, f'mlp_pid_{pid}.pth')
        model = MLPForecaster(
            lookback=lookback, in_channels=1, horizon=1, out_dim=1,
            hidden_sizes=hidden_sizes, dropout=0.2, activation='relu',
        ).to(device)

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                n_loaded += 1
            except Exception:
                os.remove(model_path)

        if not os.path.exists(model_path):
            # Re-seed for reproducibility per neighbor
            torch.manual_seed(seed)
            tr_loader = DataLoader(WindowDataset(X_tr, y_tr),
                                   batch_size=batch_size, shuffle=False)
            va_loader = (DataLoader(WindowDataset(X_va, y_va),
                                    batch_size=batch_size, shuffle=False)
                         if X_va.shape[0] > 0 else None)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.MSELoss()
            best_val, best_state = float('inf'), None
            for _ in range(epochs):
                model.train()
                for xb, yb in tr_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if va_loader is not None and len(va_loader.dataset) > 0:
                    model.eval(); vsum, vn = 0.0, 0
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            xb = xb.to(device); yb = yb.to(device)
                            vsum += loss_fn(model(xb), yb).item() * xb.size(0)
                            vn   += xb.size(0)
                    vavg = vsum / max(vn, 1)
                    if vavg < best_val:
                        best_val = vavg
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
                else:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
                torch.save(model.state_dict(), model_path)
                n_trained += 1

        models[pid]         = model.eval()
        histories_unsc[pid] = train_unsc.astype(np.float64)

        if j % 50 == 0:
            print(f"  [neighbor-local] {j}/{len(pids)} processed "
                  f"(loaded={n_loaded}, trained={n_trained}, skipped={n_skipped})", end='\r')

    print(f"[neighbor-local] Done. ready={len(models)}  "
          f"loaded={n_loaded}  trained={n_trained}  skipped={n_skipped}")
    return models, histories_unsc


def evaluate_local_mlps_all_products(
    df_wide_global,
    global_val_start_idx,
    product_minmax_scalers,
    neighbor_models,
    forecast_horizon,
    lookback,
    device,
    out_csv_path,
    seed,
):
    """Run each per-product local MLP recursively over its OWN test horizon
    and write per-product metrics (RMSE, MAE, BIAS, R2, POCID) to CSV.

    For every pid in `neighbor_models`:
      - seed buffer = unscaled series up to `global_val_start_idx + val_size`
        (i.e. the same point used as test_start by the per-target pipeline),
      - autoregress `forecast_horizon` steps with the pid's own MLP,
      - compare to the real test slice `df_wide_global.loc[pid][test_start:]`.

    One row per product is appended to `out_csv_path`.
    """
    L = df_wide_global.shape[1]
    test_start_idx = L - forecast_horizon  # same convention as per-target loop

    header = ["seed", "product_id", "n_test", "rmse", "mae", "bias", "r2", "pocid"]
    file_exists = os.path.exists(out_csv_path)
    rows_written = 0
    with open(out_csv_path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(header)

        for pid, model in neighbor_models.items():
            series_full = df_wide_global.loc[pid].values.astype(np.float64)
            if pid not in product_minmax_scalers:
                continue
            sc = product_minmax_scalers[pid]

            history_unsc = series_full[:test_start_idx]
            test_unsc    = series_full[test_start_idx:]
            if len(history_unsc) < lookback or len(test_unsc) == 0:
                continue

            buf = list(history_unsc)
            preds = []
            model = model.to(device).eval()
            with torch.no_grad():
                for _ in range(len(test_unsc)):
                    win_unsc   = np.asarray(buf[-lookback:], dtype=np.float32).reshape(-1, 1)
                    win_scaled = sc.transform(win_unsc).astype(np.float32)
                    x_t = torch.from_numpy(win_scaled).unsqueeze(0).to(device)
                    y_p = model(x_t).view(-1)[0].item()
                    nxt = float(sc.inverse_transform([[y_p]])[0, 0])
                    preds.append(nxt)
                    buf.append(nxt)

            preds = np.asarray(preds, dtype=np.float64)
            actual = test_unsc.astype(np.float64)
            mask = ~np.isnan(actual)
            actual_v, preds_v = actual[mask], preds[mask]
            if len(actual_v) == 0:
                continue

            rmse = float(np.sqrt(mean_squared_error(actual_v, preds_v)))
            mae  = float(mean_absolute_error(actual_v, preds_v))
            bias = float(np.mean(preds_v - actual_v))
            try:
                r2 = float(r2_score(actual_v, preds_v))
            except Exception:
                r2 = float('nan')
            if len(actual_v) > 1:
                d_act, d_prd = np.diff(actual_v), np.diff(preds_v)
                pocid = float(((d_act * d_prd) > 0).sum() / len(d_act))
            else:
                pocid = float('nan')

            writer.writerow([seed, pid, len(actual_v), rmse, mae, bias, r2, pocid])
            rows_written += 1

    print(f"[local-eval] seed={seed}: wrote {rows_written} per-product rows to "
          f"{os.path.basename(out_csv_path)}")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    if df.index.name == DATE_COL:
         df = df.reset_index(drop=True)
         
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    # Pre-generate df_wide and category labels for inference
    cat_labels_dict = full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict() if 'cat_label' in full_df.columns else {}
    df_wide_global = full_df.pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum').fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    # Globally define train split based on identical logic for all items
    L = len(df_wide_global.columns)
    global_train_start_idx = L - forecast_horizon - val_size - train_size
    global_val_start_idx = L - forecast_horizon - val_size

    # Build dictionary of local StandardScalers and transform df_wide_scaled for distances
    if global_train_start_idx < 0: global_train_start_idx = 0
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    
    df_wide_scaled = df_wide_global.copy()
    product_scalers = {}
    product_minmax_scalers = {}
    for item_id_iter in df_wide_global.index:
        z_scaler = StandardScaler()
        train_ts = train_df_wide.loc[item_id_iter].values.reshape(-1, 1)
        z_scaler.fit(train_ts)
        product_scalers[item_id_iter] = z_scaler
        full_ts = df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(full_ts).flatten()

        # MinMax scaler per product fit on the same train slice. This is used to
        # scale neighbor (and target) node windows when building graphs so that
        # all nodes the GCN aggregates over live on a consistent train-fit scale
        # (same family as the per-product target scaler used elsewhere).
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(train_ts)
        product_minmax_scalers[item_id_iter] = mm_scaler

    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)

    # Cache of (seed -> (neighbor_models, neighbor_histories_unsc)) so we train
    # the per-product local MLPs only once per seed and reuse across all
    # PRODUCTS_TO_TEST.
    neighbor_local_cache = {}
    
    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")
        
        df_product = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].copy()
        df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
        df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)
        
        test_start_idx = len(df_product) - forecast_horizon
        val_start_idx = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice = slice(val_start_idx, test_start_idx)
        test_slice = slice(test_start_idx, None)
        
        train = df_product[TARGET_COL][train_slice].values
        val = df_product[TARGET_COL][val_slice].values
        test = df_product[TARGET_COL][test_slice].values
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

        if len(EXOG_COLS) > 0:
            exog_train = df_product[EXOG_COLS][train_slice].values
            exog_val = df_product[EXOG_COLS][val_slice].values
            exog_test = df_product[EXOG_COLS][test_slice].values

            # Keep UNSCALED copies of val/test exog for leak-safe recursive
            # inference (the lag_*/rolling_mean_excl_* columns will be
            # recomputed from predicted targets and re-scaled per step).
            exog_val_raw = exog_val.copy()
            exog_test_raw = exog_test.copy()

            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
            
            exog_indices = df_product.columns.get_indexer(EXOG_COLS)
            df_product.iloc[train_slice, exog_indices] = exog_train_scaled
            df_product.iloc[val_slice, exog_indices] = exog_val_scaled
            df_product.iloc[test_slice, exog_indices] = exog_test_scaled
        else:
            exog_scaler = None
            exog_train_scaled = None
            exog_val_scaled = None
            exog_test_scaled = None
            exog_val_raw = None
            exog_test_raw = None

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")
            
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}')
            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            # ---------------------------------------------------------------
            # Per-product local MLPs for leak-free graph neighbor values.
            # Trained ONCE per seed and reused across all target products.
            # ---------------------------------------------------------------
            if USE_NEIGHBOR_LOCAL_MLPS:
                if seed not in neighbor_local_cache:
                    nbr_save_dir = os.path.join(best_models_seed_dir, 'local_neighbors')
                    neighbor_models, neighbor_histories_unsc = train_neighbor_local_mlps(
                        df_wide_global=df_wide_global,
                        global_val_start_idx=global_val_start_idx,
                        product_minmax_scalers=product_minmax_scalers,
                        lookback=lookback_window,
                        val_size=val_size,
                        epochs=NEIGHBOR_MLP_EPOCHS,
                        seed=seed,
                        device=device,
                        save_dir=nbr_save_dir,
                        lr=NEIGHBOR_MLP_LR,
                        weight_decay=NEIGHBOR_MLP_WEIGHT_DECAY,
                        hidden_sizes=NEIGHBOR_MLP_HIDDEN_SIZES,
                        batch_size=NEIGHBOR_MLP_BATCH_SIZE,
                    )
                    neighbor_local_cache[seed] = (neighbor_models, neighbor_histories_unsc)
                neighbor_models, neighbor_histories_unsc = neighbor_local_cache[seed]

                # Per-product evaluation of the local MLPs over the full test
                # horizon. One row per product (all ~1500 products) is appended
                # to local_mlp_all_products.csv.
                local_eval_csv = os.path.join(
                    script_dir, f'local_mlp_all_products_seed_{seed}.csv'
                )
                if not os.path.exists(local_eval_csv):
                    evaluate_local_mlps_all_products(
                        df_wide_global=df_wide_global,
                        global_val_start_idx=global_val_start_idx,
                        product_minmax_scalers=product_minmax_scalers,
                        neighbor_models=neighbor_models,
                        forecast_horizon=forecast_horizon,
                        lookback=lookback_window,
                        device=device,
                        out_csv_path=local_eval_csv,
                        seed=seed,
                    )
            else:
                neighbor_models, neighbor_histories_unsc = None, None

            all_configs = [{'metric': 'no_emb'}] + grid_configs if USE_EMBEDDINGS else [{'metric': 'no_emb'}]

            base_lstm_forecast, base_lstm_train_losses, base_lstm_val_losses = None, None, None
            base_lstm_rmse, base_lstm_mae, base_lstm_bias, base_lstm_score, base_lstm_pocid = None, None, None, None, None
            base_mlp_forecast, base_mlp_t_losses, base_mlp_v_losses = None, None, None
            base_mlp_rmse, base_mlp_mae, base_mlp_bias, base_mlp_score, base_mlp_pocid = None, None, None, None, None

            for config in all_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])
                
                results_by_w_s = {}

                if metric == 'no_emb':
                    is_threshold_mode = False
                    iterator = [(None, 15, 1, False, False)]
                else:
                    is_threshold_mode = thresholds is not None and thresholds != [None]
                    params = thresholds if is_threshold_mode else percentiles
                    iterator = itertools.product(params, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts)
            
                for param_val, window_sz, step_sz, enable_edges, enable_second_degree in iterator:
                    use_embeddings = (metric != 'no_emb')
                    
                    current_threshold = param_val if use_embeddings and is_threshold_mode else None
                    current_percentile = param_val if use_embeddings and not is_threshold_mode else None

                    key = (param_val, window_sz, step_sz)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None
                        }
                        if base_lstm_forecast is not None:
                            results_by_w_s[key]['forecasts']["LSTM Baseline"] = base_lstm_forecast
                            results_by_w_s[key]['train_losses']["LSTM Baseline"] = base_lstm_train_losses
                            results_by_w_s[key]['val_losses']["LSTM Baseline"] = base_lstm_val_losses
                            results_by_w_s[key]['rmse']["LSTM Baseline"] = base_lstm_rmse
                            results_by_w_s[key]['mae']["LSTM Baseline"] = base_lstm_mae
                            results_by_w_s[key]['bias']["LSTM Baseline"] = base_lstm_bias
                            results_by_w_s[key]['score']["LSTM Baseline"] = base_lstm_score
                            results_by_w_s[key]['pocid']["LSTM Baseline"] = base_lstm_pocid
                        if base_mlp_forecast is not None:
                            results_by_w_s[key]['forecasts']["MLP"] = base_mlp_forecast
                            results_by_w_s[key]['train_losses']["MLP"] = base_mlp_t_losses
                            results_by_w_s[key]['val_losses']["MLP"] = base_mlp_v_losses
                            results_by_w_s[key]['rmse']["MLP"] = base_mlp_rmse
                            results_by_w_s[key]['mae']["MLP"] = base_mlp_mae
                            results_by_w_s[key]['bias']["MLP"] = base_mlp_bias
                            results_by_w_s[key]['score']["MLP"] = base_mlp_score
                            results_by_w_s[key]['pocid']["MLP"] = base_mlp_pocid

                    if not use_embeddings:
                        # ── MLP Baseline ──────────────────────────────────────
                        print(f"\n{'='*60}")
                        print(f"Running MLP Baseline (seed={seed})")
                        print(f"{'='*60}")
                        _mlp_scaler = MinMaxScaler()
                        _mlp_cfg = TrainConfig(
                            lookback=lookback_window, horizon=1, batch_size=batch_size,
                            train_size=train_size, val_size=val_size, lr=LEARNING_RATE,
                            epochs=EPOCHS, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                            device=str(device),
                        )
                        mlp_model_bl, _mlp_scaler, mlp_t_losses_bl, mlp_v_losses_bl, _ = train_mlp_forecaster(
                            df=df_product, cfg=_mlp_cfg, seed=seed, loss_type='mse',
                            product_id=f"{product_id}_{store_id}_mlp_bl",
                            scaler=_mlp_scaler, target_channel=0, val_ratio=None,
                            hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS,
                            test_size=forecast_horizon,
                        )
                        # Leak-safe recursive inference: recompute lag_*/rolling_mean_excl_*
                        # from predicted targets each step (matches LSTM baseline behaviour).
                        if EXOG_COLS:
                            mlp_forecast_bl = recursive_inference_mlp_dynamic_exog(
                                model=mlp_model_bl,
                                target_scaler=_mlp_scaler,
                                exog_scaler=exog_scaler,
                                exog_cols=EXOG_COLS,
                                history_target_unscaled=val[-lookback_window:],
                                history_exog_unscaled=exog_val_raw[-lookback_window:],
                                future_exog_unscaled=exog_test_raw,
                                target_channel=0, device=str(device),
                            )
                        else:
                            _mlp_recent = val[-lookback_window:].reshape(-1, 1)
                            mlp_forecast_bl = recursive_inference_mlp(
                                model=mlp_model_bl, scaler=_mlp_scaler, recent_history=_mlp_recent,
                                future_exog=np.zeros((forecast_horizon, 0)),
                                target_channel=0, device=str(device),
                            )
                        _vm = ~np.isnan(mlp_forecast_bl)
                        _vt, _vf     = test[_vm], np.array(mlp_forecast_bl)[_vm]
                        mlp_rmse_bl  = np.sqrt(mean_squared_error(_vt, _vf)) if len(_vt) > 0 else None
                        mlp_mae_bl   = mean_absolute_error(_vt, _vf)          if len(_vt) > 0 else None
                        mlp_bias_bl  = np.mean(_vf - _vt)                     if len(_vt) > 0 else None
                        mlp_score_bl = r2_score(_vt, _vf)                     if len(_vt) > 0 else None
                        _d = (_vt[1:] - _vt[:-1]) * (_vf[1:] - _vf[:-1]) > 0
                        mlp_pocid_bl = _d.sum() / len(_d) if len(_d) > 0 else 0.0
                        base_mlp_forecast, base_mlp_t_losses, base_mlp_v_losses = mlp_forecast_bl, mlp_t_losses_bl, mlp_v_losses_bl
                        base_mlp_rmse, base_mlp_mae, base_mlp_bias = mlp_rmse_bl, mlp_mae_bl, mlp_bias_bl
                        base_mlp_score, base_mlp_pocid = mlp_score_bl, mlp_pocid_bl
                        results_by_w_s[key]['forecasts']['MLP'] = mlp_forecast_bl
                        results_by_w_s[key]['train_losses']['MLP'] = mlp_t_losses_bl
                        results_by_w_s[key]['val_losses']['MLP'] = mlp_v_losses_bl
                        results_by_w_s[key]['rmse']['MLP'] = mlp_rmse_bl
                        results_by_w_s[key]['mae']['MLP'] = mlp_mae_bl
                        results_by_w_s[key]['bias']['MLP'] = mlp_bias_bl
                        results_by_w_s[key]['score']['MLP'] = mlp_score_bl
                        results_by_w_s[key]['pocid']['MLP'] = mlp_pocid_bl
                        _mlp_cs = 0.5 * mlp_rmse_bl + 0.25 * mlp_mae_bl + 0.25 * abs(mlp_bias_bl) if None not in (mlp_rmse_bl, mlp_mae_bl, mlp_bias_bl) else None
                        print(f"MLP Baseline -> RMSE: {mlp_rmse_bl:.4f} | MAE: {mlp_mae_bl:.4f} | BIAS: {mlp_bias_bl:.4f} | POCID: {mlp_pocid_bl:.4f} | Score: {_mlp_cs:.4f}\n")

                        # ── LSTM Baseline ──────────────────────────────────────
                        print(f"\n{'='*60}")
                        print(f"Running LSTM Baseline (seed={seed})")
                        print(f"{'='*60}")
                        _lstm_input      = 1 + len(EXOG_COLS) if EXOG_COLS else 1
                        _ts_train_ds     = TimeSeriesDataset(train_scaled, exog_train_scaled if EXOG_COLS else None, lookback_window)
                        _ts_val_ds       = TimeSeriesDataset(val_scaled,   exog_val_scaled   if EXOG_COLS else None, lookback_window)
                        _ts_train_loader = DataLoader(_ts_train_ds, batch_size=batch_size, shuffle=False)
                        _ts_val_loader   = DataLoader(_ts_val_ds,   batch_size=batch_size, shuffle=False)
                        lstm_model_bl    = LSTM(input_size=_lstm_input, hidden_size=64, num_layers=2, dropout=dropout).to(device)
                        lstm_model_path  = os.path.join(best_models_seed_dir, f'lstm_product_{product_id}.pth')
                        lstm_optimizer   = torch.optim.Adam(lstm_model_bl.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
                        _lstm_crit       = nn.MSELoss()
                        lstm_model_bl, lstm_t_losses_bl, lstm_v_losses_bl, _ = train_lstm(
                            seed=seed, epochs=EPOCHS, model=lstm_model_bl,
                            train_loader=_ts_train_loader, val_loader=_ts_val_loader,
                            exog_cols=EXOG_COLS, criterion=_lstm_crit, criterion2=_lstm_crit,
                            optimizer=lstm_optimizer, device=device,
                            best_model_path=lstm_model_path, patience=PATIENCE,
                        )
                        lstm_model_bl.load_state_dict(torch.load(lstm_model_path, map_location=device, weights_only=True))
                        lstm_forecast_bl = recursive_inference_lstm(
                            model=lstm_model_bl, test_start_idx=test_start_idx, seq_length=lookback_window,
                            val_scaled=val_scaled, exog_val_scaled=exog_val_scaled,
                            exog_test_scaled=exog_test_scaled,
                            exog_test=exog_test if EXOG_COLS else np.zeros((forecast_horizon, 0)),
                            scaler=scaler, exog_scaler=exog_scaler, df_product=df_product,
                            device=device, exog_cols=EXOG_COLS if EXOG_COLS else [],
                            forecast_window=forecast_horizon, seed=seed, strategy='recursive',
                            item_id=product_id, store_id=store_id, loss_type=loss_type, script_dir=script_dir,
                        )
                        valid_mask_l = ~np.isnan(lstm_forecast_bl)
                        vt_l = test[valid_mask_l]
                        vf_l = np.array(lstm_forecast_bl)[valid_mask_l]
                        lstm_rmse_bl, lstm_mae_bl, lstm_bias_bl, lstm_score_bl, lstm_pocid_bl = None, None, None, None, None
                        if len(vt_l) > 0:
                            lstm_rmse_bl  = np.sqrt(mean_squared_error(vt_l, vf_l))
                            lstm_mae_bl   = mean_absolute_error(vt_l, vf_l)
                            lstm_bias_bl  = np.mean(vf_l - vt_l)
                            lstm_score_bl = r2_score(vt_l, vf_l)
                            d_orig = vt_l[1:] - vt_l[:-1]
                            d_pred = vf_l[1:] - vf_l[:-1]
                            lstm_pocid_bl = ((d_orig * d_pred) > 0).sum() / max(len(d_orig), 1)
                        base_lstm_forecast, base_lstm_train_losses, base_lstm_val_losses = lstm_forecast_bl, lstm_t_losses_bl, lstm_v_losses_bl
                        base_lstm_rmse, base_lstm_mae, base_lstm_bias = lstm_rmse_bl, lstm_mae_bl, lstm_bias_bl
                        base_lstm_score, base_lstm_pocid = lstm_score_bl, lstm_pocid_bl
                        results_by_w_s[key]['forecasts']['LSTM Baseline'] = lstm_forecast_bl
                        results_by_w_s[key]['train_losses']['LSTM Baseline'] = lstm_t_losses_bl
                        results_by_w_s[key]['val_losses']['LSTM Baseline'] = lstm_v_losses_bl
                        results_by_w_s[key]['rmse']['LSTM Baseline'] = lstm_rmse_bl
                        results_by_w_s[key]['mae']['LSTM Baseline'] = lstm_mae_bl
                        results_by_w_s[key]['bias']['LSTM Baseline'] = lstm_bias_bl
                        results_by_w_s[key]['score']['LSTM Baseline'] = lstm_score_bl
                        results_by_w_s[key]['pocid']['LSTM Baseline'] = lstm_pocid_bl
                        _lstm_cs = 0.5 * lstm_rmse_bl + 0.25 * lstm_mae_bl + 0.25 * abs(lstm_bias_bl) if None not in (lstm_rmse_bl, lstm_mae_bl, lstm_bias_bl) else None
                        print(f"LSTM Baseline -> RMSE: {lstm_rmse_bl:.4f} | MAE: {lstm_mae_bl:.4f} | BIAS: {lstm_bias_bl:.4f} | POCID: {lstm_pocid_bl:.4f} | Score: {_lstm_cs:.4f}\n")
                        csv_lstm_path = os.path.join(script_dir, "lstm_baseline.csv")
                        file_exists_lstm = os.path.exists(csv_lstm_path)
                        with open(csv_lstm_path, 'a', newline='') as csvfile_lstm:
                            writer_lstm = csv.writer(csvfile_lstm)
                            if not file_exists_lstm:
                                writer_lstm.writerow(["product_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid"])
                            writer_lstm.writerow([product_id, store_id, seed, "lstm_baseline", 15, 1, "", "", "", "", lstm_rmse_bl, lstm_mae_bl, lstm_bias_bl, lstm_score_bl, lstm_pocid_bl])
                        continue

                    # ── GraphSAGE+MLP with embeddings ──────────────────────────────
                    print(f"\n{'='*60}")
                    param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                    print(f"Running GraphSAGE+MLP: metric={metric}, {param_str}, window_size={window_sz}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

                    fixed_threshold = None
                    metric_type = infer_metric_type(metric)
                    distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                    current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                    compute_func = compute_distances_1vsAll if metric_type == 'distance' else compute_similarities_1vsAll

                    graphs_list, fixed_threshold = neighbourhood_graph(
                        product_id=product_id,
                        df=current_df_wide,
                        metric=metric,
                        metric_type=metric_type,
                        window_size=window_sz,
                        compute_func=compute_func,
                        threshold=current_threshold if is_threshold_mode else None,
                        percentile=current_percentile if not is_threshold_mode else None,
                        step_size=step_sz,
                        cat_labels=cat_labels_dict,
                        residuals=USE_RESIDUALS,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        train_end_idx=val_start_idx,
                        node_features=SELECTED_NODE_FEATURES,
                        node_scalers=product_minmax_scalers,
                    )
                    print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                    results_by_w_s[key]['threshold'] = fixed_threshold

                    cfg = TrainConfig(
                        lookback=lookback_window,
                        horizon=1, 
                        batch_size=batch_size,
                        train_size=train_size,
                        val_size=val_size,
                        lr=LEARNING_RATE,
                        epochs=EPOCHS,
                        weight_decay=WEIGHT_DECAY,
                        patience=PATIENCE,
                        device=str(device)
                    )

                    model, _, t_losses, v_losses, best_epoch = train_gcn_mlp(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse',
                        product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0,
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS,
                        graphs=graphs_list, test_size=forecast_horizon, graph_window_size=window_sz,
                        include_cal_lookback=INCLUDE_CAL_LOOKBACK,
                        node_features=SELECTED_NODE_FEATURES,
                        cal_columns=EXOG_COLS,
                    )

                    recent_target = val[-lookback_window:].reshape(-1, 1)
                    if exog_val_scaled is not None:
                        recent_exog = exog_val_scaled[-lookback_window:]
                        recent_history = np.column_stack([recent_target, recent_exog])
                    else:
                        recent_history = recent_target

                    start_infer = time.time()
                    forecast = recursive_inference_gcn_mlp(
                        model=model,
                        scaler=scaler,
                        recent_history=recent_history,
                        future_exog=exog_test_scaled,
                        target_channel=0,
                        device=str(device),
                        df_wide=current_df_wide,
                        cat_labels=cat_labels_dict,
                        target_id=product_id,
                        metric=metric,
                        fixed_threshold=fixed_threshold,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        past_dates=pd.to_datetime(df_product[DATE_COL][:test_start_idx]).dt.strftime('%Y-%m-%d').values,
                        future_dates=pd.to_datetime(df_product[DATE_COL][test_start_idx:]).dt.strftime('%Y-%m-%d').values,
                        graph_window_size=window_sz,
                        include_cal_lookback=INCLUDE_CAL_LOOKBACK,
                        node_features=SELECTED_NODE_FEATURES,
                        cal_columns=EXOG_COLS,
                        node_scalers=product_minmax_scalers,
                        # Leak-safe dynamic exog: recompute lag/rolling per step.
                        exog_scaler=exog_scaler if EXOG_COLS else None,
                        exog_cols=EXOG_COLS if EXOG_COLS else None,
                        history_exog_unscaled=exog_val_raw[-lookback_window:] if EXOG_COLS else None,
                        future_exog_unscaled=exog_test_raw if EXOG_COLS else None,
                        # Leak-safe neighbor values: each neighbor's future
                        # cells in the graph window are filled with that
                        # neighbor's OWN local-MLP prediction (past cells stay
                        # real). Disabled when neighbor_models is None.
                        neighbor_models=neighbor_models,
                        neighbor_histories_unscaled=neighbor_histories_unsc,
                    )
                    infer_time = time.time() - start_infer
                        
                    valid_mask = ~np.isnan(forecast)
                    valid_test = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse, mae, bias, score, pocid = None, None, None, None, None
                    if len(valid_test) > 0:
                        rmse = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                        mae = mean_absolute_error(valid_test, valid_forecast)
                        bias = np.mean(valid_forecast - valid_test)
                        score = r2_score(valid_test, valid_forecast)
                        diff_original = valid_test[1:] - valid_test[:-1]
                        diff_pred = valid_forecast[1:] - valid_forecast[:-1]
                        is_positive = (diff_original * diff_pred) > 0
                        pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0
                        
                    th_str = f"{fixed_threshold:.4f}" if fixed_threshold is not None else "N/A"
                    param_str_label = f"th:{current_threshold}" if is_threshold_mode else f"pct:{current_percentile} (val:{th_str})"
                    label_name = f"{param_str_label}|w:{window_sz}|st:{step_sz}|e:{enable_edges}|2nd:{enable_second_degree}"

                    results_by_w_s[key]['forecasts'][label_name] = forecast
                    results_by_w_s[key]['train_losses'][label_name] = t_losses
                    results_by_w_s[key]['val_losses'][label_name] = v_losses
                    results_by_w_s[key]['rmse'][label_name] = rmse
                    results_by_w_s[key]['mae'][label_name] = mae
                    results_by_w_s[key]['bias'][label_name] = bias
                    results_by_w_s[key]['score'][label_name] = score
                    results_by_w_s[key]['pocid'][label_name] = pocid
                
                    _gnn_cs = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias) if None not in (rmse, mae, bias) else None
                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | BIAS: {bias:.4f} | POCID: {pocid:.4f} | Score: {_gnn_cs:.4f}\n")
                    
                    csv_results_path = os.path.join(script_dir, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["product_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid"])
                        
                        writer.writerow([product_id, store_id, seed, metric, window_sz, step_sz, current_threshold, current_percentile, enable_edges, enable_second_degree, rmse, mae, bias, score, pocid])

                train_index = df_product[DATE_COL][train_slice].values
                val_index = df_product[DATE_COL][val_slice].values
                test_index = df_product[DATE_COL][test_slice].values

                if metric == 'no_emb':
                    values_str = "no_thresholds"
                    sub_dir = os.path.join(grid_search_plots_dir, 'no_emb', f'window_{15}', f'step_{1}', f'item_{product_id}', values_str)
                    os.makedirs(sub_dir, exist_ok=True)
                    save_plot_path = os.path.join(sub_dir, f"item_{product_id}_store_{store_id}_no_emb_seed_{seed}.html")
                    emb_title = f'Baseline Forecasts (MLP & LSTM | Seed={seed})'
                    
                    if SAVE_PLOTS:
                        plot_results(train, val, test, results_by_w_s[(None, 15, 1)]['forecasts'], train_index, val_index, test_index,
                                     results_by_w_s[(None, 15, 1)]['train_losses'], results_by_w_s[(None, 15, 1)]['val_losses'], metric=metric, embedding_strategy='graph2vec_mlp',
                                     window_size=15, step_size=1, threshold=None, percentile=None,
                                     target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                     save_path=save_plot_path, rmse=results_by_w_s[(None, 15, 1)]['rmse'], mae=results_by_w_s[(None, 15, 1)]['mae'], 
                                     bias=results_by_w_s[(None, 15, 1)]['bias'], score=results_by_w_s[(None, 15, 1)]['score'], pocid=results_by_w_s[(None, 15, 1)]['pocid'])
                else:
                    metric_type = infer_metric_type(metric)
                    grouped_results = {}
                    for (p, w, s), res_dicts in results_by_w_s.items():
                        group_key = (w, s)
                        if group_key not in grouped_results:
                            grouped_results[group_key] = {
                                'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                                'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {}
                            }
                        grouped_results[group_key]['forecasts'].update(res_dicts['forecasts'])
                        grouped_results[group_key]['train_losses'].update(res_dicts['train_losses'])
                        grouped_results[group_key]['val_losses'].update(res_dicts['val_losses'])
                        grouped_results[group_key]['rmse'].update(res_dicts['rmse'])
                        grouped_results[group_key]['mae'].update(res_dicts['mae'])
                        grouped_results[group_key]['bias'].update(res_dicts['bias'])
                        grouped_results[group_key]['score'].update(res_dicts['score'])
                        grouped_results[group_key]['pocid'].update(res_dicts['pocid'])

                    for (w, s), res_dicts in grouped_results.items():
                        if thresholds is not None and len(thresholds) > 0 and percentiles is None:
                            raw_str = "_".join(map(str, thresholds))
                        else:
                            raw_str = "_".join(map(str, percentiles))
                        
                        values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                        sub_dir = os.path.join(grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}', f'item_{product_id}', values_str)
                        os.makedirs(sub_dir, exist_ok=True)
                        
                        save_plot_path = os.path.join(sub_dir, f"item_{product_id}_{metric}_seed_{seed}_all_configs.html")
                        emb_title = f'Graph2Vec MLP Forecasts ({metric} | Seed={seed} | W={w} | S={s})'

                        if SAVE_PLOTS:
                            plot_results(train, val, test, res_dicts['forecasts'], train_index, val_index, test_index,
                                         res_dicts['train_losses'], res_dicts['val_losses'], metric=metric, embedding_strategy='graph2vec_mlp',
                                         window_size=w, step_size=s, threshold=None, percentile=None,
                                         target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                         save_path=save_plot_path, rmse=res_dicts['rmse'], mae=res_dicts['mae'], 
                                         bias=res_dicts['bias'], score=res_dicts['score'], pocid=res_dicts['pocid'])
                                     
    if SAVE_PLOTS:
        print("\nGenerating Correlation Plots across all collected CSVs...")
        for csv_file in os.listdir(script_dir):
            if csv_file.endswith('.csv') and csv_file != 'no_emb.csv' and csv_file != 'mlp_results.csv':
                metric_name = csv_file.replace('.csv', '')
                csv_path = os.path.join(script_dir, csv_file)
                try:
                    res_df = pd.read_csv(csv_path)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

                    x_col = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
                    plot_data = res_df.dropna(subset=[x_col, 'rmse', 'mae']).sort_values(by=x_col)
                    
                    if plot_data.empty:
                        continue

                    ax1.plot(plot_data[x_col], plot_data['rmse'], marker='o', linestyle='-', color='b')
                    ax1.set_title(f'{x_col.capitalize()} vs RMSE')
                    ax1.set_xlabel(x_col.capitalize())
                    ax1.set_ylabel('RMSE')
                    ax1.grid(True)

                    ax2.plot(plot_data[x_col], plot_data['mae'], marker='s', linestyle='-', color='r')
                    ax2.set_title(f'{x_col.capitalize()} vs MAE')
                    ax2.set_xlabel(x_col.capitalize())
                    ax2.set_ylabel('MAE')
                    ax2.grid(True)

                    plot_save_path = os.path.join(script_dir, f"{metric_name}_correlation_plot.png")
                    plt.tight_layout()
                    plt.savefig(plot_save_path)
                    plt.close()
                    print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
                except Exception as e:
                    print(f"Failed to generate plot for {csv_file}: {e}")

if __name__ == "__main__":
    main()