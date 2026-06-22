"""
MTGNN end-to-end pipeline for the 61-product, single-store benchmark.

Trains ONE global MTGNN over all 61 products jointly and evaluates it
per-product, mirroring the dissertation's protocol so results drop into the
same comparison tables / plots as the LSTM/MLP/GCN/GAT/Graph2Vec models:

  * splits         : 547 train / 61 val / 153 test (chronological, no shuffle)
  * target scaling : per-node MinMax fitted on the training segment only
  * exogenous      : the same calendar/holiday features as the baselines
                     (generate_exogenous_features), shared across nodes
  * objective      : one-step-ahead MSE, AdamW, ReduceLROnPlateau, early stopping
  * inference      : recursive multi-step roll-out over the 153-day horizon;
                     calendar features are known a-priori, demand is fed back
  * metrics        : RMSE / MAE / Bias / Score / POCID per product (compute_metrics)
  * plots          : per-product Forecast-vs-Actual via plots.plot_results

The learned data-adaptive adjacency is saved for the graph-quality analysis.

Run:  python agcrn_pipeline.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ── Paths & sys.path setup ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))     # DynamicSimilarities/

from plots import plot_results                               # noqa: E402
from utils import generate_exogenous_features, compute_metrics  # noqa: E402
from mtgnn_model import MTGNN                                # noqa: E402

# ── config (matches the baseline pipeline) ──────────────────────────────────
# The 61-product single-store benchmark lives in top_12500.feather; that file
# defines the node set AGCRN forecasts jointly.
DATA_PATH        = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../dataset/top_12500.feather"))
DATE_COL         = "date"
TARGET_COL       = "value"

val_size         = 61
forecast_horizon = 153
train_size       = 761 - val_size - forecast_horizon        # 547
lookback_window  = 30
BATCH_SIZE       = 32

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]

EPOCHS        = 1000
PATIENCE      = 100
LR            = 1e-3
# MTGNN hyper-parameters (Wu et al. 2020)
SUBGRAPH_SIZE = 20          # top-k neighbours kept in the learned graph
NODE_DIM      = 40          # graph-constructor node-embedding dim
NUM_LAYERS    = 3           # TC→GC blocks
GCN_DEPTH     = 2           # mix-hop propagation depth
SEED          = 42

PLOTS_DIR     = os.path.join(SCRIPT_DIR, "plots_mtgnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(s):
    torch.manual_seed(s); np.random.seed(s % 2**32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── data prep ───────────────────────────────────────────────────────────────
def prepare_data():
    """Load the 61-node demand panel + shared calendar exogenous features.

    Returns a dict with raw and scaled demand (T, N), scaled exog (T, F),
    per-node MinMax scalers, the date index, and the chronological split points.
    """
    df = pd.read_feather(DATA_PATH)
    if DATE_COL not in df.columns:
        df = df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id"]).reset_index(drop=True)

    # demand matrix (N items × T dates) → (T, N)
    wide = df.pivot_table(index="item_id", columns=DATE_COL,
                          values=TARGET_COL, aggfunc="sum").fillna(0.0)
    items = list(wide.index)
    dates = pd.to_datetime(wide.columns)
    demand = wide.values.T.astype(np.float32)                # (T, N)
    T, N = demand.shape

    # calendar exog (shared across nodes): one row per date
    cal = pd.DataFrame({DATE_COL: dates})
    cal = generate_exogenous_features(cal, date_col=DATE_COL, exog_cols=EXOG_COLS)
    exog = cal[EXOG_COLS].values.astype(np.float32)          # (T, F)

    # chronological split indices
    test_start  = T - forecast_horizon
    val_start   = test_start - val_size
    train_start = val_start - train_size
    assert train_start >= 0, f"not enough history: T={T}"

    # per-node MinMax target scaling, fitted on train only
    scalers = []
    demand_s = np.zeros_like(demand)
    for n in range(N):
        sc = MinMaxScaler()
        sc.fit(demand[train_start:val_start, n].reshape(-1, 1))
        demand_s[:, n] = sc.transform(demand[:, n].reshape(-1, 1)).flatten()
        scalers.append(sc)

    # exog MinMax fitted on train only
    ex_sc = MinMaxScaler().fit(exog[train_start:val_start])
    exog_s = ex_sc.transform(exog).astype(np.float32)        # (T, F)

    return dict(
        items=items, dates=dates, N=N, T=T, F=exog_s.shape[1],
        demand=demand, demand_s=demand_s, exog_s=exog_s, scalers=scalers,
        train_start=train_start, val_start=val_start, test_start=test_start,
    )


def _channels(demand_s, exog_s):
    """(T, N, C) = per-node demand ‖ broadcast calendar exog."""
    T, N = demand_s.shape
    F = exog_s.shape[1]
    x = np.empty((T, N, 1 + F), dtype=np.float32)
    x[:, :, 0] = demand_s
    x[:, :, 1:] = exog_s[:, None, :].repeat(N, axis=1)
    return x


def make_windows(X, demand_s, start, end):
    """One-step-ahead windows for targets in [start, end).
    X:(T,N,C) demand_s:(T,N). Returns (xs (B,L,N,C), ys (B,N))."""
    xs, ys = [], []
    for t in range(start, end):
        if t - lookback_window < 0:
            continue
        xs.append(X[t - lookback_window:t])
        ys.append(demand_s[t])
    return np.stack(xs), np.stack(ys)


# ── train ───────────────────────────────────────────────────────────────────
def train(model, Xtr, ytr, Xva, yva):
    """Train AGCRN one-step-ahead. Returns (model, train_losses, val_losses)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).to(device); yva_t = torch.tensor(yva).to(device)
    n = Xtr_t.shape[0]

    train_losses, val_losses = [], []
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
            opt.zero_grad()
            pred = model(xb)[:, 0, :, 0]                      # horizon=1 → (B, N)
            loss = lossf(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vpred = model(Xva_t)[:, 0, :, 0]
            vloss = lossf(vpred, yva_t).item()
        val_losses.append(vloss)
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 25 == 0 or bad == 0:
            print(f"  epoch {epoch:4d}  val_mse={vloss:.5f}  best={best_val:.5f}  bad={bad}")
        if bad >= PATIENCE:
            print(f"  early stop @ epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ── recursive multi-step inference ──────────────────────────────────────────
@torch.no_grad()
def recursive_forecast(model, data):
    model.eval()
    N, F = data["N"], data["F"]
    demand_s = data["demand_s"]; exog_s = data["exog_s"]
    test_start = data["test_start"]

    buf = list(demand_s[:test_start, :])                     # rolling scaled-demand buffer
    preds_s = np.zeros((forecast_horizon, N), dtype=np.float32)

    for h in range(forecast_horizon):
        t = test_start + h
        window_d = np.stack(buf[-lookback_window:])          # (L, N)
        x = np.empty((lookback_window, N, 1 + F), dtype=np.float32)
        x[:, :, 0] = window_d
        x[:, :, 1:] = exog_s[t - lookback_window:t, None, :].repeat(N, axis=1)
        xb = torch.tensor(x).unsqueeze(0).to(device)         # (1, L, N, C)
        yhat = model(xb)[0, 0, :, 0].cpu().numpy()           # (N,) scaled
        preds_s[h] = yhat
        buf.append(yhat)

    # inverse-scale per node → original demand
    preds = np.zeros_like(preds_s)
    for n in range(N):
        preds[:, n] = data["scalers"][n].inverse_transform(
            preds_s[:, n].reshape(-1, 1)
        ).flatten()
    return preds                                             # (H, N)


# ── evaluation + plotting ───────────────────────────────────────────────────
def evaluate_and_plot(preds, data, train_losses, val_losses):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ts, vs, tes = data["train_start"], data["val_start"], data["test_start"]
    dates = data["dates"]
    demand = data["demand"]

    train_idx = dates[ts:vs]
    val_idx   = dates[vs:tes]
    test_idx  = dates[tes:]

    rows = []
    for n, iid in enumerate(data["items"]):
        y_test = demand[tes:, n]
        y_pred = preds[:, n]
        rmse, mae, bias, score, pocid = compute_metrics(y_test, y_pred)
        rows.append(dict(item_id=iid, rmse=rmse, mae=mae,
                         bias=bias, score=score, pocid=pocid * 100.0))

        plot_results(
            train=demand[ts:vs, n], val=demand[vs:tes, n], test=y_test,
            forecast=y_pred,
            train_index=train_idx, val_index=val_idx, test_index=test_idx,
            train_losses=train_losses, val_losses=val_losses,
            target_col=TARGET_COL,
            title=f"MTGNN — item {iid}",
            save_path=os.path.join(PLOTS_DIR, f"mtgnn_item_{iid}.png"),
            rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid * 100.0,
        )

    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _seed(SEED)
    print(f"Loading {DATA_PATH}")
    data = prepare_data()
    print(f"  N={data['N']} products, T={data['T']} days, exog F={data['F']}")

    X = _channels(data["demand_s"], data["exog_s"])          # (T, N, C)
    Xtr, ytr = make_windows(X, data["demand_s"], data["train_start"], data["val_start"])
    Xva, yva = make_windows(X, data["demand_s"], data["val_start"], data["test_start"])
    print(f"  train windows {Xtr.shape}  val windows {Xva.shape}")

    model = MTGNN(num_nodes=data["N"], in_dim=X.shape[-1], out_dim=1,
                  seq_length=lookback_window, subgraph_size=SUBGRAPH_SIZE,
                  node_dim=NODE_DIM, layers=NUM_LAYERS, gcn_depth=GCN_DEPTH).to(device)
    print(f"  MTGNN params: {sum(p.numel() for p in model.parameters()):,}")

    model, train_losses, val_losses = train(model, Xtr, ytr, Xva, yva)

    preds = recursive_forecast(model, data)
    res = evaluate_and_plot(preds, data, train_losses, val_losses)

    print("\nPer-product metrics (head):")
    print(res.head(10).to_string(index=False))
    print(f"\nMean RMSE={res.rmse.mean():.4f}  MAE={res.mae.mean():.4f}  "
          f"Bias={res.bias.mean():.4f}  POCID={res.pocid.mean():.2f}")

    out_csv = os.path.join(SCRIPT_DIR, "mtgnn_results.csv")
    res.to_csv(out_csv, index=False)
    np.save(os.path.join(SCRIPT_DIR, "mtgnn_learned_adjacency.npy"),
            model.learned_adjacency().cpu().numpy())
    print(f"\nSaved {out_csv}, per-product plots in {PLOTS_DIR}/, and learned adjacency.")


if __name__ == "__main__":
    main()
