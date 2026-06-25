"""
Scalability plots from scalability_results1.csv (the complete 8-model run).

Produces:
  1. cum_train_evolution.{png,pdf}  - cumulative training time vs n, one line per model
  2. cum_infer_evolution.{png,pdf}  - cumulative inference time vs n, one line per model
  3. mean_build_time.{png,pdf}      - average per-product graph build time per model (bar)
  4. build_time_summary.csv         - mean/std/total build_s per model
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "scalability_results1.csv")

df = pd.read_csv(CSV)

# Stable, readable ordering: baselines first, then graph2vec, gcn, gat.
ORDER = ["lstm_baseline", "mlp_baseline",
         "graph2vec_lstm", "graph2vec_mlp",
         "gcn_lstm", "gcn_mlp",
         "gat_lstm", "gat_mlp"]
variants = [v for v in ORDER if v in df["variant"].unique()]
# any unexpected variants appended at the end
variants += [v for v in df["variant"].unique() if v not in variants]

cmap   = plt.get_cmap("tab10")
colors = {v: cmap(i % 10) for i, v in enumerate(variants)}


def _line_plot(ycol, ylabel, title, outname):
    fig, ax = plt.subplots(figsize=(11, 6))
    for v in variants:
        sub = df[df["variant"] == v].sort_values("n")
        ax.plot(sub["n"], sub[ycol], label=v, color=colors[v], linewidth=1.6)
    ax.set_xlabel("Number of products processed (n)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(HERE, f"{outname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ── 1. Cumulative training time evolution ────────────────────────────────
_line_plot(
    "cum_train_s",
    "Cumulative training time (s)",
    "Scalability: cumulative training time vs workload size",
    "cum_train_evolution",
)

# ── 2. Cumulative inference time evolution ───────────────────────────────
_line_plot(
    "cum_infer_s",
    "Cumulative inference time (s)",
    "Scalability: cumulative inference time vs workload size",
    "cum_infer_evolution",
)

# ── 3. Average graph build time per model ────────────────────────────────
build = (df.groupby("variant")["build_s"]
           .agg(mean_build_s="mean", std_build_s="std",
                total_build_s="sum", n_products="count")
           .reindex(variants)
           .reset_index())
build_csv = os.path.join(HERE, "build_time_summary.csv")
build.round(4).to_csv(build_csv, index=False)
print(f"\nSaved {build_csv}")
print(build.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(build["variant"], build["mean_build_s"],
              yerr=build["std_build_s"].fillna(0), capsize=4,
              color=[colors[v] for v in build["variant"]], alpha=0.85)
ax.set_ylabel("Mean per-product graph build time (s)", fontsize=11)
ax.set_title("Average graph build time per model (± std)", fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
for b, v in zip(bars, build["mean_build_s"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(HERE, f"mean_build_time.{ext}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
plt.close(fig)
