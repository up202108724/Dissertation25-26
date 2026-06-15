"""
Evaluation suite for picking ONE training strategy across all products.

Why not just average RMSE across products?
    Products live on very different scales (RMSE ranges from ~11 to ~36 in this
    dataset). A raw mean is dominated by a handful of high-volume series, so it
    measures "who wins the big sellers", not "who is the best strategy overall".

This suite therefore evaluates strategies in a SCALE-FREE way:

  1. Average each metric over seeds  -> one value per (strategy, product).
  2. Per-product ranking             -> mean rank + win-rate per strategy
                                        (scale-free; the headline number).
  3. Relative-to-best ratio          -> geometric-mean of metric / best-per-product
                                        (how far from ideal, on average).
  4. Seed stability                  -> std of the metric across seeds.
  5. Significance                    -> Friedman test over products; if significant,
                                        pairwise Wilcoxon signed-rank vs the leader,
                                        Holm-corrected (no extra dependencies).
  6. Cost tiebreaker                 -> mean train / inference time.

Run:
    python evaluate_strategies.py                      # auto-discovers worker CSVs
    python evaluate_strategies.py path1.csv path2.csv  # explicit files
"""

import sys
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

KEY = ["item_id", "store_id"]

# metric -> True if LOWER is better
METRIC_DIRECTION = {
    "rmse": True, "mae": True, "score": True, "bias_abs": True, "pocid": False,
}
PRIMARY_METRIC = "rmse"   # change to "score" if you want the composite as headline


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_results(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "bias" in df.columns:
        df["bias_abs"] = df["bias"].abs()
    # keep only successful runs if a status column exists
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def available_metrics(df):
    return [m for m in METRIC_DIRECTION if m in df.columns]


def pivot_metric(df, metric):
    """Average over seeds, then return a products x strategies matrix for one metric.
    Only products evaluated under EVERY strategy are kept (fair paired comparison)."""
    per_prod = (
        df.groupby(KEY + ["strategy"])[metric].mean().reset_index()
    )
    mat = per_prod.pivot_table(index=KEY, columns="strategy", values=metric)
    return mat.dropna(axis=0, how="any")


# --------------------------------------------------------------------------- #
# Scale-free aggregates
# --------------------------------------------------------------------------- #
def rank_table(mat, lower_is_better):
    """Per-product ranks (1 = best). Returns ranks matrix."""
    ascending = lower_is_better
    return mat.rank(axis=1, ascending=ascending, method="average")


def summarize_metric(mat, lower_is_better):
    ranks = rank_table(mat, lower_is_better)
    best_per_prod = mat.min(axis=1) if lower_is_better else mat.max(axis=1)

    # relative-to-best ratio (>=1 for lower-is-better). Guard against zero.
    safe_best = best_per_prod.replace(0, np.nan)
    if lower_is_better:
        rel = mat.div(safe_best, axis=0)
    else:
        rel = safe_best.values[:, None] / mat  # ideal/value

    summary = pd.DataFrame({
        "mean_rank": ranks.mean(),
        "win_rate": (ranks == 1).mean(),
        "rel_to_best_geomean": np.exp(np.log(rel).mean()),
        "mean_value": mat.mean(),
        "median_value": mat.median(),
    })
    return summary.sort_values("mean_rank"), ranks


# --------------------------------------------------------------------------- #
# Significance (Friedman + Holm-corrected pairwise Wilcoxon)
# --------------------------------------------------------------------------- #
def significance(mat, lower_is_better):
    cols = list(mat.columns)
    cols_data = [mat[c].values for c in cols]
    chi2, p_fried = friedmanchisquare(*cols_data)

    leader = (mat.mean().idxmin() if lower_is_better else mat.mean().idxmax())

    pairs, pvals = [], []
    for a, b in itertools.combinations(cols, 2):
        try:
            _, p = wilcoxon(mat[a].values, mat[b].values)
        except ValueError:
            p = np.nan
        pairs.append((a, b))
        pvals.append(p)

    # Holm correction
    order = np.argsort([p if not np.isnan(p) else 1.0 for p in pvals])
    m = len(pvals)
    holm = [np.nan] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx]) if not np.isnan(pvals[idx]) else np.nan
        if not np.isnan(adj):
            running_max = max(running_max, adj)
            holm[idx] = running_max
    pair_df = pd.DataFrame(
        {"pair": [f"{a} vs {b}" for a, b in pairs], "p_raw": pvals, "p_holm": holm}
    )
    return chi2, p_fried, leader, pair_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    paths = sys.argv[1:] or sorted(glob.glob("lstm_strategies_results_w*.csv"))
    if not paths:
        paths = sorted(glob.glob("lstm_results_seed_*.csv")) or ["lstm_strategies_results.csv"]
    print(f"Loading: {paths}\n")

    df = load_results(paths)
    metrics = available_metrics(df)
    n_prod = df[KEY].drop_duplicates().shape[0]
    print(f"{len(df)} rows | {df.strategy.nunique()} strategies | "
          f"{df.seed.nunique()} seeds | {n_prod} products")
    print(f"Metrics found: {metrics}\n")

    overall_rank = pd.DataFrame()

    for metric in metrics:
        lower = METRIC_DIRECTION[metric]
        mat = pivot_metric(df, metric)
        summary, ranks = summarize_metric(mat, lower)
        print("=" * 72)
        print(f"METRIC: {metric}  ({'lower' if lower else 'higher'} is better) "
              f"| {mat.shape[0]} products compared")
        print("=" * 72)
        print(summary.round(4).to_string())

        chi2, p, leader, pairs = significance(mat, lower)
        print(f"\nFriedman chi2={chi2:.3f}  p={p:.2e}  "
              f"({'significant' if p < 0.05 else 'NOT significant'} differences)")
        print(f"Leader: {leader}")
        print(pairs.round(4).to_string(index=False))
        print()

        overall_rank[metric] = summary["mean_rank"]

    # ---- seed stability (std across seeds, averaged over products) ----------
    if df.seed.nunique() > 1:
        stab = (df.groupby(KEY + ["strategy"])[PRIMARY_METRIC].std()
                  .groupby("strategy").mean().sort_values())
        print("=" * 72)
        print(f"SEED STABILITY (mean across-seed std of {PRIMARY_METRIC}, lower=more stable)")
        print("=" * 72)
        print(stab.round(4).to_string())
        print()

    # ---- cost ----------------------------------------------------------------
    cost_cols = [c for c in ["train_time", "inference_time"] if c in df.columns]
    if cost_cols:
        cost = df.groupby("strategy")[cost_cols].mean()
        print("=" * 72)
        print("COST (mean seconds)")
        print("=" * 72)
        print(cost.round(3).to_string())
        print()

    # ---- final verdict -------------------------------------------------------
    overall_rank["avg_rank_over_metrics"] = overall_rank.mean(axis=1)
    overall_rank = overall_rank.sort_values("avg_rank_over_metrics")
    print("#" * 72)
    print("OVERALL MEAN-RANK PER STRATEGY (across all metrics; lower = better)")
    print("#" * 72)
    print(overall_rank.round(3).to_string())
    winner = overall_rank.index[0]
    print(f"\n>>> Recommended strategy: {winner}")
    print("    (confirm the Friedman/Wilcoxon results above show its lead is "
          "statistically meaningful; otherwise prefer the cheapest/most stable).")

    overall_rank.to_csv("strategy_evaluation_summary.csv")
    print("\nSaved -> strategy_evaluation_summary.csv")


if __name__ == "__main__":
    main()