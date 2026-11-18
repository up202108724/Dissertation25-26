"""
Robust statistical evaluation of the hybrid GNN forecasters (Reviewer 1 & 3).

Runs, for every metric x model family (LSTM / MLP):
  1. Classic Demsar  - Friedman + Nemenyi on product x seed blocks (600)  [original analysis]
  2. Product-level   - Friedman + Nemenyi on 60 product blocks (seeds averaged)
  3. Approach A      - pairwise Wilcoxon (Holm), rank-biserial r, BCa CIs over products
  4. Approach B      - hierarchical (product-cluster) paired bootstrap over all runs
  5. Per-seed check  - Friedman per seed + rank stability across seeds
  6. Significance comparison table across all four testing strategies
and repeats 1-4 with the GCN z-ablation treated as a separate model.

Outputs go to  Statistical Analysis/Results/<graph_type>/
    cd_diagrams/   CD diagrams (PDF + PNG) for 600-block and product-level rankings
    figures/       forest plots (A vs B CIs), rank CIs, per-seed rank stability
    tables/        CSVs + LaTeX tables ready for the paper
    run_log.txt    full console output

Usage:  python statistical_analysis_run.py
"""
import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import statistical_analysis as sa  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = os.path.join(HERE, "spearman_inference_fix.csv")
GRAPH_TYPE = "spearman"
METRICS = ["mae", "pocid"]
FAMILIES = ["lstm", "mlp"]
ALPHA = 0.05
N_BOOT = 10000
RESAMPLE_SEEDS = False        # True -> two-stage bootstrap (products, then seeds)
REFERENCE = None              # e.g. {'lstm': 'lstm_baseline', 'mlp': 'mlp_baseline'} -> only vs baseline
RUN_ABLATIONS = True          # also treat ablate_z=True runs as their own model
RANDOM_STATE = 0

OUT_DIR = os.path.join(HERE, "Results", os.path.splitext(os.path.basename(CSV_PATH))[0])
CD_DIR = os.path.join(OUT_DIR, "cd_diagrams")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

# Plot styling (validated categorical slots 1-2; text stays neutral ink)
C_A, C_B = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

PRETTY = {
    "lstm_baseline": "LSTM", "graph2vec_lstm": "Graph2Vec-LSTM", "gcn_lstm": "GCN-LSTM",
    "gat_lstm": "GAT-LSTM", "mlp_baseline": "MLP", "graph2vec_mlp": "Graph2Vec-MLP",
    "gcn_mlp": "GCN-MLP", "gat_mlp": "GAT-MLP",
}


def pretty(name):
    if name.endswith("_ablated"):
        return PRETTY.get(name[:-8], name[:-8]) + " (no z)"
    return PRETTY.get(name, name)


def tag(metric, family, ablations):
    return f"{metric}_{family}" + ("_ablations" if ablations else "")


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"))
    plt.close(fig)


# =============================================================================
# PLOTS
# =============================================================================
def plot_cd(avg_ranks, nemenyi, title, name):
    """CD diagram; the Nemenyi matrix is always drawn, the title carries the omnibus p."""
    ranks = avg_ranks.rename(index=pretty)
    sig = nemenyi.rename(index=pretty, columns=pretty)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    sp.critical_difference_diagram(ranks, sig, ax=ax, label_fmt_left="{label} ({rank:.2f})  ",
                                   label_fmt_right="  ({rank:.2f}) {label}")
    ax.set_title(title, pad=14)
    save(fig, CD_DIR, name)


def plot_forest(pa, pb, metric, family, name):
    """Mean improvement per pair with Approach A (BCa over products) and B (hierarchical) CIs."""
    pa = pa.reset_index(drop=True)
    pb = pb.set_index(["model_a", "model_b"]).loc[list(zip(pa.model_a, pa.model_b))].reset_index()
    n = len(pa)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.4))
    ax.axvline(0, color=INK_MUTED, lw=1, zorder=1)
    for df, off, col, lab, sig in ((pa, 0.14, C_A, "A: product-level (BCa CI, Wilcoxon-Holm)", pa.significant),
                                   (pb, -0.14, C_B, "B: hierarchical bootstrap (Holm)", pb.significant)):
        yy = y + off
        ax.hlines(yy, df.ci_low, df.ci_high, color=col, lw=2, zorder=2)
        for yi, x, s in zip(yy, df.mean_improvement, sig):
            ax.plot(x, yi, "o", ms=8, mec=col, mew=2, mfc=col if s else "white", zorder=3)
        ax.plot([], [], "o-", color=col, ms=8, label=lab)
    ax.plot([], [], "o", ms=8, mec=INK_MUTED, mfc=INK_MUTED, label="significant (p_Holm < %.2f)" % ALPHA)
    ax.plot([], [], "o", ms=8, mec=INK_MUTED, mfc="white", label="not significant")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty(a)} vs {pretty(b)}" for a, b in zip(pa.model_a, pa.model_b)])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Mean {metric.upper()} improvement of left model (> 0 = left model better)")
    ax.set_title(f"{metric.upper()} | {family.upper()} | pairwise differences, 95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 - 0.25 / max(n, 1)), ncol=2,
              frameon=False, fontsize=8.5)
    save(fig, FIG_DIR, name)


def plot_rank_ci(rb, metric, family, name):
    rb = rb.sort_values("avg_rank", ascending=False)
    y = np.arange(len(rb))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(rb) + 1.2))
    ax.hlines(y, rb.rank_ci_low, rb.rank_ci_high, color=C_A, lw=2)
    ax.plot(rb.avg_rank, y, "o", ms=8, color=C_A)
    for yi, r, pbst in zip(y, rb.rank_ci_high, rb.prob_best):
        ax.text(r + 0.02, yi, f"P(best)={pbst:.2f}", va="center", fontsize=8.5, color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(m) for m in rb.model])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average rank (1 = best), 95% product-level bootstrap CI")
    ax.set_title(f"{metric.upper()} | {family.upper()} | rank uncertainty")
    save(fig, FIG_DIR, name)


def plot_seed_stability(seed_ranks, pooled, metric, family, name):
    """Dots = average rank within each seed (60 products); diamond = pooled over seeds."""
    order = pooled.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(order) + 1.2))
    for yi, m in enumerate(order):
        xs = seed_ranks[m].values
        ax.plot(xs, np.full_like(xs, yi, dtype=float), "o", ms=6, color=C_A, alpha=0.45, mec="white", mew=0.8)
        ax.plot(pooled[m], yi, "D", ms=8, color=INK)
    ax.plot([], [], "o", color=C_A, alpha=0.6, label="single seed")
    ax.plot([], [], "D", color=INK, label="all seeds")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty(m) for m in order])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average rank over products (1 = best)")
    ax.set_title(f"{metric.upper()} | {family.upper()} | rank stability across seeds")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, FIG_DIR, name)


# =============================================================================
# ANALYSIS PIECES
# =============================================================================
def demsar_blocks(df, metric, family, ablations, mode):
    """Friedman + Nemenyi via the existing framework; Nemenyi is computed even if Friedman fails."""
    pivot, avg_ranks, nemenyi = sa.run_demsar_framework_by_family(
        df, metric=metric, alpha=ALPHA, mode=mode, compare_ablations=ablations, model_family=family)
    chi2, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    if nemenyi is None:
        nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
        nemenyi.index = nemenyi.columns = pivot.columns.tolist()
    return pivot, avg_ranks, nemenyi, chi2, p


def per_seed(df, metric, family):
    d = sa._load_filtered(df, False, family)
    hib = sa._higher_is_better(metric)
    rows, rank_rows = [], {}
    for seed, g in d.groupby("seed"):
        piv = g.pivot_table(index="product_id", columns="model_variant", values=metric, aggfunc="first").dropna()
        chi2, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
        r = piv.rank(axis=1, ascending=not hib).mean()
        rank_rows[seed] = r
        rows.append(dict(metric=metric, model_family=family, seed=seed, n_products=len(piv),
                         friedman_chi2=chi2, friedman_p=p, best_model=r.idxmin(),
                         **{f"rank_{m}": v for m, v in r.items()}))
    return pd.DataFrame(rows), pd.DataFrame(rank_rows).T


def latex_pairwise(pa, pb, metric, family):
    pb = pb.set_index(["model_a", "model_b"])
    lines = [
        r"\begin{table}[ht]", r"\centering\small",
        rf"\caption{{Product-level pairwise comparisons ({metric.upper()}, {family.upper()} family, $N={PRODUCTS}$ products). "
        r"$\Delta$: mean improvement of the first model (positive = better), 95\% BCa CI over products; "
        r"$r_{rb}$: matched-pairs rank-biserial correlation; W/L: products won/lost; "
        r"$p_{W}$: Wilcoxon signed-rank, Holm-adjusted; $p_{HB}$: hierarchical bootstrap, Holm-adjusted.}",
        rf"\label{{tab:pairwise_{metric}_{family}}}",
        r"\begin{tabular}{lrcrcrr}", r"\toprule",
        r"Comparison & $\Delta$ & 95\% CI & $r_{rb}$ & W/L & $p_{W}$ & $p_{HB}$ \\", r"\midrule",
    ]
    for _, r in pa.iterrows():
        hb = pb.loc[(r.model_a, r.model_b)]
        def fmt(p):
            s = "$<$0.001" if p < 0.001 else "%.3f" % p
            return r"\textbf{" + s + "}" if p < ALPHA else s
        lines.append(
            f"{pretty(r.model_a)} vs {pretty(r.model_b)} & {r.mean_improvement:.3f} & "
            f"[{r.ci_low:.3f}, {r.ci_high:.3f}] & {r.rank_biserial_r:.2f} & {r.wins_a}/{r.losses_a} & "
            f"{fmt(r.p_holm)} & {fmt(hb.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_omnibus(omni):
    lines = [
        r"\begin{table}[ht]", r"\centering\small",
        r"\caption{Omnibus tests with seeds pooled ($N=600$ product$\times$seed blocks) versus product-level "
        r"blocks ($N=60$, seeds averaged). $W$: Kendall's coefficient of concordance (product level).}",
        r"\label{tab:omnibus}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Metric & Family & $\chi^2_{600}$ & $p_{600}$ & $\chi^2_{60}$ & $p_{60}$ & $W$ \\", r"\midrule",
    ]
    for _, r in omni.iterrows():
        lines.append(f"{r.metric.upper()} & {r.model_family.upper()} & {r.friedman_chi2_600:.2f} & "
                     f"{r.friedman_p_600:.2e} & {r.friedman_chi2:.2f} & {r.friedman_p:.2e} & {r.kendall_W:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
PRODUCTS = None


def run(df, ablations):
    global PRODUCTS
    suffix = "_ablations" if ablations else ""
    label = " (z-ablation as separate model)" if ablations else ""
    omni_rows, pa_all, pb_all, rb_all, comp_rows, ranks_rows = [], [], [], [], [], []

    for metric in METRICS:
        for family in FAMILIES:
            t = tag(metric, family, ablations)
            ref = REFERENCE.get(family) if isinstance(REFERENCE, dict) else REFERENCE

            # 1. classic 600-block Demsar
            _, r600, n600, chi600, p600 = demsar_blocks(df, metric, family, ablations, "total")
            plot_cd(r600, n600, f"{metric.upper()} - {family.upper()}{label} - product x seed blocks "
                                f"(N=600), Friedman p={p600:.1e}", f"cd_{t}_600blocks")
            n600.to_csv(os.path.join(TAB_DIR, f"nemenyi_{t}_600blocks.csv"))

            # 2 + 3. product-level Demsar + Approach A
            omni, pa, r60, n60 = sa.product_level_analysis(
                df, metric, family, ablations, ref, ALPHA, n_boot=N_BOOT, random_state=RANDOM_STATE)
            PRODUCTS = omni["n_products"]
            plot_cd(r60, n60, f"{metric.upper()} - {family.upper()}{label} - product blocks "
                              f"(N={PRODUCTS}), Friedman p={omni['friedman_p']:.1e}", f"cd_{t}_perproduct")
            n60.to_csv(os.path.join(TAB_DIR, f"nemenyi_{t}_perproduct.csv"))
            omni.update(friedman_chi2_600=chi600, friedman_p_600=p600)
            omni_rows.append(omni)

            # 4. Approach B
            pb, rb = sa.hierarchical_bootstrap_analysis(
                df, metric, family, ablations, ref, ALPHA, N_BOOT, RESAMPLE_SEEDS, RANDOM_STATE)

            plot_forest(pa, pb, metric, family, f"forest_{t}")
            plot_rank_ci(rb, metric, family, f"rank_ci_{t}")

            for m in r600.index:
                ranks_rows.append(dict(metric=metric, model_family=family, model=m,
                                       rank_600blocks=r600[m], rank_perproduct=r60[m]))

            # 6. how each testing strategy judges every pair
            pbi = pb.set_index(["model_a", "model_b"])
            for _, r in pa.iterrows():
                a, b = r.model_a, r.model_b
                hb = pbi.loc[(a, b)]
                comp_rows.append(dict(
                    metric=metric, model_family=family, model_a=a, model_b=b,
                    mean_improvement=r.mean_improvement,
                    p_nemenyi_600=n600.loc[a, b], p_nemenyi_60=n60.loc[a, b],
                    p_wilcoxon_holm_60=r.p_holm, p_hierboot_holm=hb.p_holm,
                    sig_nemenyi_600=n600.loc[a, b] < ALPHA, sig_nemenyi_60=n60.loc[a, b] < ALPHA,
                    sig_wilcoxon_holm_60=r.p_holm < ALPHA, sig_hierboot_holm=hb.p_holm < ALPHA))

            pa_all.append(pa); pb_all.append(pb); rb_all.append(rb)

            if not ablations:
                with open(os.path.join(TAB_DIR, f"latex_pairwise_{metric}_{family}.tex"), "w", encoding="utf-8") as f:
                    f.write(latex_pairwise(pa, pb, metric, family))

    omni = pd.DataFrame(omni_rows)
    comp = pd.DataFrame(comp_rows)
    comp["conclusion_robust"] = comp[["sig_nemenyi_600", "sig_nemenyi_60",
                                      "sig_wilcoxon_holm_60", "sig_hierboot_holm"]].nunique(axis=1) == 1
    tables = {
        f"omnibus{suffix}": omni,
        f"average_ranks{suffix}": pd.DataFrame(ranks_rows),
        f"productlevel_pairwise{suffix}": pd.concat(pa_all, ignore_index=True),
        f"hierboot_pairwise{suffix}": pd.concat(pb_all, ignore_index=True),
        f"hierboot_ranks{suffix}": pd.concat(rb_all, ignore_index=True),
        f"significance_comparison{suffix}": comp,
    }
    for name, tbl in tables.items():
        tbl.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=False)
    if not ablations:
        with open(os.path.join(TAB_DIR, "latex_omnibus.tex"), "w", encoding="utf-8") as f:
            f.write(latex_omnibus(omni))
    return comp


def main():
    for d in (CD_DIR, FIG_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {CSV_PATH}: {df.shape[0]} rows, {df.product_id.nunique()} products, "
          f"{df.seed.nunique()} seeds, variants={sorted(df.model_variant.unique())}")

    comp = run(df, ablations=False)

    # 5. per-seed robustness
    seed_tables = []
    for metric in METRICS:
        for family in FAMILIES:
            st, seed_ranks = per_seed(df, metric, family)
            seed_tables.append(st)
            plot_seed_stability(seed_ranks, seed_ranks.mean(), metric, family, f"seed_stability_{metric}_{family}")
    pd.concat(seed_tables, ignore_index=True).to_csv(os.path.join(TAB_DIR, "per_seed_friedman.csv"), index=False)

    if RUN_ABLATIONS and (df["ablate_z"] == True).any():
        run(df, ablations=True)

    print("\n" + "=" * 70)
    print("SIGNIFICANCE COMPARISON (does each pairwise conclusion survive?)")
    print("=" * 70)
    cols = ["metric", "model_family", "model_a", "model_b", "sig_nemenyi_600", "sig_nemenyi_60",
            "sig_wilcoxon_holm_60", "sig_hierboot_holm", "conclusion_robust"]
    with pd.option_context("display.width", 200):
        print(comp[cols].to_string(index=False))
    print(f"\nAll outputs written to: {OUT_DIR}")


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    with open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        main()
