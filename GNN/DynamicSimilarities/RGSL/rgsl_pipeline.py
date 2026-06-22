"""
RGSL end-to-end pipeline for the 61-product, single-store benchmark.

Trains ONE global RGSL over all 61 products jointly and evaluates it
per-product, mirroring the dissertation's protocol (so results drop into the
same tables/plots as the LSTM/MLP/GCN/GAT/Graph2Vec/AGCRN/MTGNN models):

  * splits         : 547 train / 61 val / 153 test (chronological, no shuffle)
  * target scaling : per-node MinMax fitted on the training segment only
  * exogenous      : the same calendar/holiday features as the baselines
  * objective      : one-step-ahead MSE (+ sparsity reg on the learned graph),
                     AdamW, ReduceLROnPlateau, early stopping
  * inference      : recursive multi-step roll-out over the 153-day horizon
  * metrics        : RMSE / MAE / Bias / Score / POCID per product
  * plots          : per-product Forecast-vs-Actual via plots.plot_results

The DISTINCTIVE bit: RGSL fuses a learned (Gumbel-sampled) graph with an
EXPLICIT PRIOR built here from the dissertation's loss-independent Spearman/CID
similarity over the training period. So Path-A (hand-crafted) and Path-B
(learned) graphs live in the SAME model, and the learned prior-weight β tells
you how much the network leans on your hand-crafted graph vs. its own.

Run:  python rgsl_pipeline.py
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

from plots import plot_results                                          # noqa: E402
from utils import (generate_exogenous_features, compute_metrics,        # noqa: E402
                   compute_similarities_1vsAll, compute_distances_1vsAll)
from rgsl_model import RGSL                                             # noqa: E402

# ── config ──────────────────────────────────────────────────────────────────
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

# ── prior graph (Path-A) — fed into RGSL as the explicit prior ──────────────
PRIOR_METRIC    = "spearman"     # 'spearman' (similarity) or 'cid' (distance)
PRIOR_THRESHOLD = 0.634          # dissertation's Spearman tau (CID would use 4.158)

EPOCHS        = 1000
PATIENCE      = 100
LR            = 1e-3
RNN_UNITS     = 64
EMBED_DIM     = 10
NODE_DIM      = 10
NUM_LAYERS    = 2
CHEB_K        = 2
TAU           = 1.0
REG_WEIGHT    = 1e-4             # sparsity penalty on the learned graph
SEED          = 42

PLOTS_DIR     = os.path.join(SCRIPT_DIR, "plots_rgsl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(s):
    torch.manual_seed(s); np.random.seed(s % 2**32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── data prep ───────────────────────────────────────────────────────────────
def prepare_data():
    df = pd.read_feather(DATA_PATH)
    if DATE_COL not in df.columns:
        df = df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id"]).reset_index(drop=True)

    wide = df.pivot_table(index="item_id", columns=DATE_COL,
                          values=TARGET_COL, aggfunc="sum").fillna(0.0)
    items = list(wide.index)
    dates = pd.to_datetime(wide.columns)
    demand = wide.values.T.astype(np.float32)                # (T, N)
    T, N = demand.shape

    cal = pd.DataFrame({DATE_COL: dates})
    cal = generate_exogenous_features(cal, exog_cols=EXOG_COLS, date_col=DATE_COL)
    exog = cal[EXOG_COLS].values.astype(np.float32)          # (T, F)

    test_start  = T - forecast_horizon
    val_start   = test_start - val_size
    train_start = val_start - train_size
    assert train_start >= 0, f"not enough history: T={T}"

    scalers = []
    demand_s = np.zeros_like(demand)
    for n in range(N):
        sc = MinMaxScaler()
        sc.fit(demand[train_start:val_start, n].reshape(-1, 1))
        demand_s[:, n] = sc.transform(demand[:, n].reshape(-1, 1)).flatten()
        scalers.append(sc)

    ex_sc = MinMaxScaler().fit(exog[train_start:val_start])
    exog_s = ex_sc.transform(exog).astype(np.float32)

    return dict(
        items=items, dates=dates, N=N, T=T, F=exog_s.shape[1],
        demand=demand, demand_s=demand_s, exog_s=exog_s, scalers=scalers,
        train_start=train_start, val_start=val_start, test_start=test_start,
    )


def build_prior_adjacency(demand, train_start, val_start, metric, threshold):
    """Loss-independent Path-A prior over the N products from the TRAIN window.

    Spearman → similarity matrix, keep weights >= threshold.
    CID       → distance matrix, keep edges <= threshold, weight = 1/(1+dist).
    Returns a symmetric (N, N) tensor with zero diagonal.
    """
    series = demand[train_start:val_start, :].T.astype(np.float32)   # (N, T_train)
    N = series.shape[0]
    A = np.zeros((N, N), dtype=np.float32)

    if metric in ("spearman", "pearson", "kendall"):
        for i in range(N):
            sims = compute_similarities_1vsAll(series[i], series, metric=metric)
            keep = sims >= threshold
            A[i] = np.where(keep, np.clip(sims, 0.0, None), 0.0)
    elif metric == "cid":
        # CID expects z-scored series (train-fitted)
        mu = series.mean(1, keepdims=True); sd = series.std(1, keepdims=True) + 1e-8
        z = (series - mu) / sd
        for i in range(N):
            dist = compute_distances_1vsAll(z[i], z, metric="cid")
            keep = dist <= threshold
            A[i] = np.where(keep, 1.0 / (1.0 + dist), 0.0)
    else:
        raise ValueError(f"unsupported PRIOR_METRIC {metric!r}")

    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)                                      # symmetrise
    return torch.from_numpy(A)


def _channels(demand_s, exog_s):
    T, N = demand_s.shape
    F = exog_s.shape[1]
    x = np.empty((T, N, 1 + F), dtype=np.float32)
    x[:, :, 0] = demand_s
    x[:, :, 1:] = exog_s[:, None, :].repeat(N, axis=1)
    return x


def make_windows(X, demand_s, start, end):
    xs, ys = [], []
    for t in range(start, end):
        if t - lookback_window < 0:
            continue
        xs.append(X[t - lookback_window:t])
        ys.append(demand_s[t])
    return np.stack(xs), np.stack(ys)


# ── train ───────────────────────────────────────────────────────────────────
def train(model, Xtr, ytr, Xva, yva):
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
            loss = lossf(pred, yb) + REG_WEIGHT * model.graph_reg_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva_t)[:, 0, :, 0], yva_t).item()
        val_losses.append(vloss)
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 25 == 0 or bad == 0:
            print(f"  epoch {epoch:4d}  val_mse={vloss:.5f}  best={best_val:.5f}  "
                  f"prior_gate={model.prior_weight():.3f}  bad={bad}")
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

    buf = list(demand_s[:test_start, :])
    preds_s = np.zeros((forecast_horizon, N), dtype=np.float32)

    for h in range(forecast_horizon):
        t = test_start + h
        window_d = np.stack(buf[-lookback_window:])
        x = np.empty((lookback_window, N, 1 + F), dtype=np.float32)
        x[:, :, 0] = window_d
        x[:, :, 1:] = exog_s[t - lookback_window:t, None, :].repeat(N, axis=1)
        xb = torch.tensor(x).unsqueeze(0).to(device)
        yhat = model(xb)[0, 0, :, 0].cpu().numpy()
        preds_s[h] = yhat
        buf.append(yhat)

    preds = np.zeros_like(preds_s)
    for n in range(N):
        preds[:, n] = data["scalers"][n].inverse_transform(
            preds_s[:, n].reshape(-1, 1)).flatten()
    return preds


# ── evaluation + plotting ───────────────────────────────────────────────────
def evaluate_and_plot(preds, data, train_losses, val_losses):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ts, vs, tes = data["train_start"], data["val_start"], data["test_start"]
    dates = data["dates"]; demand = data["demand"]
    train_idx, val_idx, test_idx = dates[ts:vs], dates[vs:tes], dates[tes:]

    rows = []
    for n, iid in enumerate(data["items"]):
        y_test = demand[tes:, n]; y_pred = preds[:, n]
        rmse, mae, bias, score, pocid = compute_metrics(y_test, y_pred)
        rows.append(dict(item_id=iid, rmse=rmse, mae=mae,
                         bias=bias, score=score, pocid=pocid * 100.0))
        plot_results(
            train=demand[ts:vs, n], val=demand[vs:tes, n], test=y_test, forecast=y_pred,
            train_index=train_idx, val_index=val_idx, test_index=test_idx,
            train_losses=train_losses, val_losses=val_losses, target_col=TARGET_COL,
            title=f"RGSL — item {iid}",
            save_path=os.path.join(PLOTS_DIR, f"rgsl_item_{iid}.png"),
            rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid * 100.0,
        )
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _seed(SEED)
    print(f"Loading {DATA_PATH}")
    data = prepare_data()
    print(f"  N={data['N']} products, T={data['T']} days, exog F={data['F']}")

    prior = build_prior_adjacency(data["demand"], data["train_start"],
                                  data["val_start"], PRIOR_METRIC, PRIOR_THRESHOLD)
    dens = float((prior > 0).float().mean())
    print(f"  prior graph ({PRIOR_METRIC}, tau={PRIOR_THRESHOLD}): "
          f"{int((prior > 0).sum())} edges, density {dens*100:.1f}%")

    X = _channels(data["demand_s"], data["exog_s"])
    Xtr, ytr = make_windows(X, data["demand_s"], data["train_start"], data["val_start"])
    Xva, yva = make_windows(X, data["demand_s"], data["val_start"], data["test_start"])
    print(f"  train windows {Xtr.shape}  val windows {Xva.shape}")

    model = RGSL(num_nodes=data["N"], prior_adj=prior, in_dim=X.shape[-1], out_dim=1,
                 horizon=1, rnn_units=RNN_UNITS, embed_dim=EMBED_DIM, node_dim=NODE_DIM,
                 num_layers=NUM_LAYERS, cheb_k=CHEB_K, tau=TAU).to(device)
    print(f"  RGSL params: {sum(p.numel() for p in model.parameters()):,}")

    model, train_losses, val_losses = train(model, Xtr, ytr, Xva, yva)

    preds = recursive_forecast(model, data)
    res = evaluate_and_plot(preds, data, train_losses, val_losses)

    print("\nPer-product metrics (head):")
    print(res.head(10).to_string(index=False))
    print(f"\nMean RMSE={res.rmse.mean():.4f}  MAE={res.mae.mean():.4f}  "
          f"Bias={res.bias.mean():.4f}  POCID={res.pocid.mean():.2f}")
    print(f"Final LM3 prior gate = {model.prior_weight():.3f} "
          f"(~1 = leans on prior / ~0 = leans on learned)")

    out_csv = os.path.join(SCRIPT_DIR, "rgsl_results.csv")
    res.to_csv(out_csv, index=False)
    np.save(os.path.join(SCRIPT_DIR, "rgsl_learned_adjacency.npy"),
            model.learned_adjacency().cpu().numpy())
    np.save(os.path.join(SCRIPT_DIR, "rgsl_prior_adjacency.npy"), prior.cpu().numpy())
    print(f"\nSaved {out_csv}, plots in {PLOTS_DIR}/, and learned/fused/prior adjacencies.")


if __name__ == "__main__":
    main()
