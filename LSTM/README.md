# LSTM baseline

A local, per-product forecasting baseline. For each (item, store) one LSTM is trained to predict the next day's sales from the previous 30 days plus calendar/holiday features. The full test window (153 days) is then forecast **recursively**. Four training strategies are compared on the same data.

## Pipeline

```
data (.feather) -> exogenous features -> per-product split (train / val / test)
   -> MinMax scaling (fit on train only) -> sliding-window dataset
   -> train (one of 4 strategies) -> recursive inference -> metrics, CSV, plots
   -> evaluate_strategies.py (rank strategies across products)
```

**Split per product:** test = last 153 days, validation = the 61 days before, train = the remaining 547 days.

## Files

| File | Role |
|---|---|
| `lstm.py` | Model definition |
| `dataset.py` | Sliding-window dataset |
| `utils.py` | Feature generation and metrics |
| `lstm_traincopy.py` | Training loops (one per strategy) |
| `lstm_inference.py` | Recursive multi-step inference |
| `lstm_strategies_top_products.py` | Main experiment (debug configuration: one product) |
| `lstm_strategies_top_products_parallelized.py` | Same experiment over all products in `top_12500.feather` |
| `grid_execution.py` | Hyperparameter grid search |
| `evaluate_strategies.py` | Cross-product strategy comparison and significance tests |
| `plots.py` | Interactive Plotly figures |
| `drawarchiteture.py` | Architecture diagram (`lstm_baseline_architecture.png/pdf`) |

## Building blocks

### `lstm.py`: model
`nn.LSTM` (hidden 32, 1 layer, `batch_first`) -> last time step's hidden state -> dropout -> `Linear(hidden, 1)`. It is many-to-one and outputs the next day's value. Dropout inside the LSTM is only active when `num_layers > 1`.

### `dataset.py`: `TimeSeriesDataset`
Sample `idx`:
- **Input:** `target[idx : idx+L]` stacked column-wise with the exogenous features, shape `(L, 1 + n_exog)`.
- **Label:** `target[idx+L]`.
- **Exog alignment:** the exogenous slice is `[idx+1 : idx+L+1]`, shifted one step ahead, so the features of the day being predicted are visible. This is valid because calendar and holiday features are known in advance.

### `utils.py`
- `generate_exogenous_features`: builds only the requested columns. It covers calendar parts, weekend/month-edge flags, US holidays, Black Friday, `is_pre_holiday_N` / `is_post_holiday_N`, bridge days, cyclical sin/cos terms, promos, lags and rolling means.
- `compute_metrics`: RMSE, MAE, bias, composite `score = 0.5·RMSE + 0.25·MAE + 0.25·|bias|`, and POCID (share of steps where the predicted direction of change matches the real one).
- `scale_exogenous_features`, `build_dynamic_exog_row`, `transform_exog_row`: scaling and per-step feature helpers. The main scripts do not use them (they use a plain `MinMaxScaler`).

## Training strategies (`lstm_traincopy.py`)

All use MSE loss, Adam, `ReduceLROnPlateau`, early stopping on validation loss and `shuffle=False`.

| Strategy | Function | Logic |
|---|---|---|
| `best_val` | `train_model` | Train on train, validate each epoch, keep the checkpoint with the lowest validation loss. |
| `combined` | `train_model_combined` | Find the optimal epoch count `E` with `best_val`, reset to the initial weights, then retrain on train+val for exactly `E` epochs. Uses the validation data for training but has no checkpoint selection. |
| `expanding_window` | `train_model_expanding_window` | Walk-forward: window k trains on `[0 : end_k]` and validates on the next block, then `end_k` grows by `val_step_size`. Weights carry over between windows. Folds: initial train 365 days, step 81 days (3 folds). |
| `sliding_window` | `train_model_sliding_window` | Same as expanding, but the training window has fixed length and its start also moves forward (recent data only). |
| *(unused)* | `train_model_best_train_loss` | Saves the checkpoint with the best training loss instead of validation loss. Not wired into the experiment. |

For both window strategies the best checkpoint is chosen by the lowest validation loss **across windows**, so losses from blocks of different difficulty are compared directly.

## Inference (`lstm_inference.py`)

`recursive_inference` forecasts the horizon one step at a time:
1. Start from the last `L` scaled validation values.
2. Predict `y_hat`, append it to the window and drop the oldest value.
3. Take the next day's exogenous row from the known future calendar. If `lag_*` or `rolling_mean_*` columns are used, recompute them from the model's own predictions (inverse-scaled), so there is no test-data leakage.
4. Inverse-transform at the end and log each step's input and output to `inference_logs/seed_*/<loss>/<strategy>/`.

Errors compound over 153 steps, so forecasts tend to flatten toward the mean.

## Experiment drivers

### `lstm_strategies_top_products.py` (main experiment)
For each seed, product and strategy: train from scratch, run recursive inference, compute metrics, append a row to the results CSV, and save a Plotly strategy comparison.

Built-in parallelism: the coordinator spawns `N_PRODUCT_WORKERS` subprocesses (`PARALLEL_MODE="product"`, each takes `products[worker_id::N]`) or one per seed (`"seed"`). Each worker writes `lstm_strategies_results_w{id}.csv` and logs to `log_worker{id}_of{N}.txt`, with 1 thread per process. The coordinator merges the CSVs at the end.

Current configuration: `hidden_size=32`, `num_layers=1`, `dropout=0.0`, `lr=1e-3`, `lookback=30`, `batch=32`, `EPOCHS=1000`, `patience=150`, `SEEDS=[42, 1000, 26008]`.

Known caveats:
- `target_products = [(911753, 6269)]` and `.values[:5]` restrict it to one product (debug setting).
- The CSV row only stores `rmse`, `mae`, `train_time`, `inference_time` and `best_epoch`. Bias, score and POCID are computed but not saved, although `evaluate_strategies.py` looks for them.

### `lstm_strategies_top_products_parallelized.py`
Identical, except `target_products = None` and the `[:5]` cap is removed. It runs all products from `top_12500.feather`.

### `grid_execution.py`
Grid search over learning rate, hidden size, layers, dropout, sequence length and batch size for a few hard-coded products, using `train_model`. It plots the configurations side by side.

Known caveats:
- It imports `LSTM.lstm_train` and `model_utils.utils`, which do not exist in this folder.
- It uses a different split (455/154/152).
- It uses the pre-built future exog without dynamic lag handling.

## Analysis and plots

### `evaluate_strategies.py`
Picks one strategy across products on a scale-free basis, since product scales differ widely:
1. Average each metric over seeds (one value per strategy and product).
2. Per-product ranks: mean rank and win rate.
3. Geometric-mean ratio to the best strategy.
4. Seed stability (std across seeds).
5. Friedman test, then pairwise Wilcoxon tests with Holm correction.
6. Mean training/inference time as a tiebreaker.

```
python evaluate_strategies.py                    # auto-discovers lstm_strategies_results_w*.csv
python evaluate_strategies.py a.csv b.csv        # explicit files
```
Output: `strategy_evaluation_summary.csv`.

### `plots.py`
`plot_results` produces an interactive Plotly figure with train/val/test, the forecasts, loss curves and a metrics table.

### `drawarchiteture.py`
Draws the architecture diagram with matplotlib. It does not use any data.

## Outputs

| Path | Content |
|---|---|
| `lstm_strategies_results*.csv` | Per-run metrics and timings |
| `strategy_evaluation_summary.csv` | Mean rank per strategy and metric |
| `grid_search_results.csv` | Grid search results |
| `inference_logs/` | Step-by-step recursive inference logs |
| `best_models/`, `grid_search_plots/` | Checkpoints and HTML plots (created at run time) |
