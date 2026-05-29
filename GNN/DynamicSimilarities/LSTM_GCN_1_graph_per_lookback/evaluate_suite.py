"""
Thesis-grade evaluation suite for the per-step GCN+LSTM forecaster.

Reads the persistent results CSV written by ``main_smooth.py`` (e.g.
``spearman.csv``) and produces aggregate statistics suitable for the
results-discussion chapter:

  1. Per (product, seed), pick the BEST threshold (lowest RMSE) on the
     full (non-ablation) model.
  2. Per seed, average the resulting best-RMSE across products
     (mean ± std).  Also report the most-frequently-chosen threshold and
     the mean of the chosen thresholds.
  3. Pick the BEST SEED — the one with the lowest mean RMSE across
     products.
  4. Aggregate results for the best seed: mean ± std of RMSE, MAE, bias,
     R^2, POCID across products.
  5. Ablation delta: per (product, seed), best-RMSE(full) - best-RMSE(abl).
     Positive ⇒ GCN helps.
  6. Read ``timings.csv`` (if present) and report wall-clock time per seed
     and total.

All tables are printed to stdout and saved as CSV files inside
``evaluation/<metric>/``.

Usage:
    python evaluate_suite.py                 # uses METRIC default below
    python evaluate_suite.py spearman        # explicit metric
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Defaults ──────────────────────────────────────────────────────────────
METRIC_DEFAULT = "spearman"
METRICS_TO_REPORT = ["rmse", "mae", "bias", "r2_score", "pocid"]
SELECTION_METRIC = "rmse"           # used to pick the best threshold/seed
LOWER_IS_BETTER = {"rmse", "mae"}   # bias is signed; r2/pocid higher is better


# ──────────────────────────────────────────────────────────────────────────
def _load_results(metric: str) -> pd.DataFrame:
    path = os.path.join(SCRIPT_DIR, f"{metric}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path, on_bad_lines="skip")
    # normalise ablate_z to bool
    if "ablate_z" in df.columns:
        df["ablate_z"] = (
            df["ablate_z"].astype(str).str.lower()
              .map({"true": True, "1": True, "1.0": True,
                    "false": False, "0": False, "0.0": False})
        )
    else:
        df["ablate_z"] = False
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    for c in METRICS_TO_REPORT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _best_per_product_seed(df: pd.DataFrame, sel: str = SELECTION_METRIC) -> pd.DataFrame:
    """For each (product, seed), keep the row with the best `sel` value
    (lower is better if `sel` ∈ LOWER_IS_BETTER, else higher)."""
    ascending = sel in LOWER_IS_BETTER
    df = df.dropna(subset=[sel]).copy()
    df["__rank"] = (
        df.groupby(["product_id", "seed"])[sel]
          .rank(method="first", ascending=ascending)
    )
    best = df[df["__rank"] == 1].drop(columns="__rank").reset_index(drop=True)
    return best


def _per_seed_summary(best_full: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the best-per-product rows across products, per seed."""
    rows = []
    for seed, g in best_full.groupby("seed"):
        row = {
            "seed": int(seed),
            "n_products": int(g["product_id"].nunique()),
            "mean_best_threshold": float(g["threshold"].mean()),
            "median_best_threshold": float(g["threshold"].median()),
            "mode_best_threshold": (
                float(g["threshold"].mode().iloc[0])
                if not g["threshold"].mode().empty else float("nan")
            ),
        }
        for m in METRICS_TO_REPORT:
            if m not in g.columns:
                continue
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"]  = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _pick_best_seed(per_seed: pd.DataFrame, sel: str = SELECTION_METRIC) -> int:
    col = f"{sel}_mean"
    ascending = sel in LOWER_IS_BETTER
    return int(per_seed.sort_values(col, ascending=ascending).iloc[0]["seed"])


def _ablation_delta(df: pd.DataFrame, sel: str = SELECTION_METRIC) -> pd.DataFrame:
    """Per (product, seed) compute best-(full) and best-(ablation) and the
    delta `ablate - full` on the selection metric (positive ⇒ GCN helps)."""
    full = _best_per_product_seed(df[df["ablate_z"] == False], sel)[  # noqa: E712
        ["product_id", "seed", sel, "threshold"]
    ].rename(columns={sel: f"{sel}_full", "threshold": "threshold_full"})

    ab = df[df["ablate_z"] == True]                                   # noqa: E712
    if ab.empty:
        return pd.DataFrame()
    abl = _best_per_product_seed(ab, sel)[
        ["product_id", "seed", sel]
    ].rename(columns={sel: f"{sel}_abl"})

    merged = full.merge(abl, on=["product_id", "seed"], how="inner")
    sign = +1 if sel in LOWER_IS_BETTER else -1
    merged[f"delta_{sel}_abl_minus_full"] = sign * (merged[f"{sel}_abl"] - merged[f"{sel}_full"])
    return merged


def _ablation_summary(ab_delta: pd.DataFrame, sel: str = SELECTION_METRIC) -> pd.DataFrame:
    if ab_delta.empty:
        return pd.DataFrame()
    col = f"delta_{sel}_abl_minus_full"
    rows = []
    for seed, g in ab_delta.groupby("seed"):
        rows.append({
            "seed": int(seed),
            "n_products": int(len(g)),
            f"{col}_mean": float(g[col].mean()),
            f"{col}_std":  float(g[col].std(ddof=1)) if len(g) > 1 else 0.0,
            "wins_for_gcn": int((g[col] > 0).sum()),
            "wins_for_ablation": int((g[col] < 0).sum()),
            "ties": int((g[col] == 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _load_timings() -> pd.DataFrame | None:
    path = os.path.join(SCRIPT_DIR, "timings.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def evaluate(metric: str = METRIC_DEFAULT) -> None:
    out_dir = os.path.join(SCRIPT_DIR, "evaluation", metric)
    os.makedirs(out_dir, exist_ok=True)

    df = _load_results(metric)
    print(f"Loaded {len(df)} rows from {metric}.csv "
          f"({df['product_id'].nunique()} products, "
          f"{df['seed'].nunique()} seeds, "
          f"{int(df['ablate_z'].sum())} ablation rows)")

    full = df[df["ablate_z"] == False]                                # noqa: E712
    if full.empty:
        print("No full-model rows found (all rows are ablation). Aborting.")
        return

    # ── 1. Best threshold per (product, seed) ────────────────────────────
    best_full = _best_per_product_seed(full, SELECTION_METRIC)
    best_full.to_csv(os.path.join(out_dir, "best_per_product_seed.csv"), index=False)

    _print_section("1. Best threshold per (product, seed) — head")
    print(best_full[["product_id", "seed", "threshold",
                     *[m for m in METRICS_TO_REPORT if m in best_full.columns]]]
          .head(15).to_string(index=False))

    # ── 2. Per-seed summary across products ──────────────────────────────
    per_seed = _per_seed_summary(best_full)
    per_seed.to_csv(os.path.join(out_dir, "per_seed_summary.csv"), index=False)

    _print_section("2. Per-seed summary (across products)")
    print(per_seed.round(4).to_string(index=False))

    # ── 3. Best seed ─────────────────────────────────────────────────────
    best_seed = _pick_best_seed(per_seed, SELECTION_METRIC)
    _print_section(f"3. Best seed by mean {SELECTION_METRIC.upper()}: {best_seed}")
    print(per_seed[per_seed["seed"] == best_seed].round(4).to_string(index=False))

    # ── 4. Best-seed mean ± std table ────────────────────────────────────
    best_seed_rows = best_full[best_full["seed"] == best_seed]
    summary_rows = []
    for m in METRICS_TO_REPORT:
        if m not in best_seed_rows.columns:
            continue
        vals = best_seed_rows[m].dropna()
        summary_rows.append({
            "metric": m,
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min":  float(vals.min()) if len(vals) else float("nan"),
            "max":  float(vals.max()) if len(vals) else float("nan"),
            "n":    int(len(vals)),
        })
    best_seed_summary = pd.DataFrame(summary_rows)
    best_seed_summary.to_csv(
        os.path.join(out_dir, f"best_seed_{best_seed}_summary.csv"), index=False,
    )
    _print_section(f"4. Best-seed ({best_seed}) — mean ± std across products")
    print(best_seed_summary.round(4).to_string(index=False))

    # ── 5. Ablation delta ────────────────────────────────────────────────
    ab_delta = _ablation_delta(df, SELECTION_METRIC)
    if not ab_delta.empty:
        ab_delta.to_csv(os.path.join(out_dir, "ablation_per_product_seed.csv"), index=False)
        ab_summary = _ablation_summary(ab_delta, SELECTION_METRIC)
        ab_summary.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)

        _print_section("5. Ablation delta (positive ⇒ GCN helps)")
        print(ab_summary.round(4).to_string(index=False))
    else:
        _print_section("5. Ablation delta")
        print("No ablation rows present — skip.")

    # ── 6. Timings ───────────────────────────────────────────────────────
    timings = _load_timings()
    _print_section("6. Wall-clock timings")
    if timings is None:
        print("timings.csv not found (run main_smooth.py to generate).")
    else:
        per_prod = timings[timings["product_id"].astype(str).str.isnumeric()]
        per_seed_total = timings[timings["product_id"] == "TOTAL_SEED"]
        total_all = timings[timings["product_id"] == "TOTAL_ALL"]

        if not per_seed_total.empty:
            t = per_seed_total[["seed", "seconds"]].copy()
            t["seconds"] = pd.to_numeric(t["seconds"], errors="coerce")
            t["minutes"] = t["seconds"] / 60.0
            print("Per-seed totals:")
            print(t.round(2).to_string(index=False))
        if not total_all.empty:
            tot = float(total_all["seconds"].iloc[0])
            print(f"\nGrand total: {tot:.1f} s  ({tot/60:.2f} min)")
        if not per_prod.empty:
            per_prod = per_prod.copy()
            per_prod["seconds"] = pd.to_numeric(per_prod["seconds"], errors="coerce")
            print(f"\nPer-product per-seed: {len(per_prod)} rows "
                  f"(mean {per_prod['seconds'].mean():.1f} s, "
                  f"median {per_prod['seconds'].median():.1f} s)")

    _print_section(f"All evaluation artifacts written to: {out_dir}")


if __name__ == "__main__":
    metric = sys.argv[1] if len(sys.argv) > 1 else METRIC_DEFAULT
    evaluate(metric)
