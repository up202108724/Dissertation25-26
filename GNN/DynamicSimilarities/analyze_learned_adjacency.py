"""
Inspect a learned adjacency produced by the AGCRN / MTGNN pipelines.

Both pipelines save a (N, N) ``*_learned_adjacency.npy`` and forecast the same
61-product node set, ordered by ascending ``item_id`` (the pivot index). This
tool maps row/col indices back to item_ids, prints summary statistics, and
writes plots — useful for the dissertation's graph-quality analysis (§3.6),
e.g. comparing the learned topology against the Spearman/CID graphs.

Usage
-----
    # auto-discovers AGCRN/agcrn_learned_adjacency.npy and
    # MTGNN/mtgnn_learned_adjacency.npy if no path is given:
    python analyze_learned_adjacency.py

    # explicit file + options:
    python analyze_learned_adjacency.py --adj AGCRN/agcrn_learned_adjacency.npy --topk 15

Convention: ``A[i, j]`` is the edge weight FROM node j INTO node i (message
aggregation direction used by both models), so row i = i's incoming neighbours.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plots import plot_networkx_plotly  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../../dataset/top_12500.feather"))


# ── helpers ──────────────────────────────────────────────────────────────────
def load_item_ids():
    """Node order = ascending item_id, matching the pivot index in both pipelines."""
    try:
        df = pd.read_feather(DATA_PATH)
        col = "item_id" if "item_id" in df.columns else df.columns[0]
        return sorted(df[col].unique().tolist())
    except Exception as e:  # noqa: BLE001
        print(f"  (could not load item_ids from {DATA_PATH}: {e}; using indices)")
        return None


def load_cat_labels():
    """item_id -> cat_label, for colouring the NetworkX graph by category."""
    try:
        df = pd.read_feather(DATA_PATH)
        if "cat_label" in df.columns and "item_id" in df.columns:
            return df.drop_duplicates("item_id").set_index("item_id")["cat_label"].to_dict()
    except Exception:  # noqa: BLE001
        pass
    return {}


def build_networkx_graph(A, items, cat_labels, top_per_node, min_weight):
    """Convert the learned adjacency into a directed NetworkX graph for drawing.

    Edge j -> i carries weight A[i,j]. To keep the picture legible we keep, for
    each target i, only its top-``top_per_node`` strongest incoming edges above
    ``min_weight`` (AGCRN's dense graph is otherwise a 3660-edge hairball).
    """
    N = A.shape[0]
    lbl = (lambda i: items[i]) if items is not None else (lambda i: i)
    G = nx.DiGraph()
    for i in range(N):
        node = lbl(i)
        G.add_node(node, cat_label=cat_labels.get(node, "Unknown Category"))

    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)
    for i in range(N):                                   # incoming edges of target i
        row = A_off[i]
        order = np.argsort(-row)
        kept = 0
        for j in order:
            if kept >= top_per_node or row[j] <= min_weight:
                break
            G.add_edge(lbl(j), lbl(i), weight=float(row[j]), similarity=float(row[j]))
            kept += 1
    return G


def default_adj_paths():
    cands = [
        os.path.join(SCRIPT_DIR, "AGCRN", "agcrn_learned_adjacency.npy"),
        os.path.join(SCRIPT_DIR, "MTGNN", "mtgnn_learned_adjacency.npy"),
    ]
    return [p for p in cands if os.path.exists(p)]


# ── statistics ───────────────────────────────────────────────────────────────
def print_stats(A, items, topk):
    N = A.shape[0]
    off = ~np.eye(N, dtype=bool)
    w = A[off]                                   # off-diagonal weights
    nnz = A[off] > 0
    out_deg = (A > 0).sum(axis=0) - np.diag(A > 0)   # j influences how many targets
    in_deg = (A > 0).sum(axis=1) - np.diag(A > 0)    # i listens to how many sources
    in_strength = A.sum(axis=1) - np.diag(A)
    out_strength = A.sum(axis=0) - np.diag(A)
    asym = np.abs(A - A.T).sum() / (np.abs(A).sum() + 1e-12)

    print("=" * 64)
    print(f"Adjacency: {N}x{N}  ({N} nodes)")
    print("-" * 64)
    print(f"  weight  min/mean/max     : {A.min():.4f} / {A.mean():.4f} / {A.max():.4f}")
    print(f"  off-diag nonzero (edges) : {int(nnz.sum())}  "
          f"(density {nnz.mean()*100:.1f}% of {N*(N-1)} possible)")
    print(f"  edge weight mean (nz)    : {w[w > 0].mean():.4f}" if nnz.any() else "  no edges")
    print(f"  self-loop (diag) mean    : {np.diag(A).mean():.4f}")
    print(f"  row-sum  mean/std        : {A.sum(1).mean():.4f} / {A.sum(1).std():.4f}"
          f"   (~1 => row-stochastic, e.g. AGCRN)")
    print(f"  asymmetry ||A-A^T||/||A||: {asym:.3f}   (0 = symmetric, larger = more directed)")
    print("-" * 64)
    print(f"  in-degree  min/mean/max  : {in_deg.min():.0f} / {in_deg.mean():.1f} / {in_deg.max():.0f}")
    print(f"  out-degree min/mean/max  : {out_deg.min():.0f} / {out_deg.mean():.1f} / {out_deg.max():.0f}")

    def lbl(i):
        return str(items[i]) if items is not None else f"#{i}"

    # most influential (highest out-strength) and most-dependent (in-strength) nodes
    print("-" * 64)
    print(f"  Top {topk} most INFLUENTIAL nodes (out-strength = sum weight sent):")
    for i in np.argsort(-out_strength)[:topk]:
        print(f"    item {lbl(i):>10}  out_strength={out_strength[i]:.3f}  out_deg={int(out_deg[i])}")
    print(f"  Top {topk} most DEPENDENT nodes (in-strength = sum weight received):")
    for i in np.argsort(-in_strength)[:topk]:
        print(f"    item {lbl(i):>10}  in_strength={in_strength[i]:.3f}  in_deg={int(in_deg[i])}")

    # strongest individual directed edges j → i
    print("-" * 64)
    print(f"  Top {topk} strongest directed edges (source -> target : weight):")
    A_off = A.copy(); np.fill_diagonal(A_off, 0.0)
    flat = np.argsort(-A_off, axis=None)[:topk]
    for f in flat:
        i, j = np.unravel_index(f, A_off.shape)     # A[i,j]: j -> i
        if A_off[i, j] <= 0:
            continue
        print(f"    {lbl(j):>10} -> {lbl(i):<10}  {A_off[i, j]:.4f}")
    print("=" * 64)

    return dict(out_strength=out_strength, in_strength=in_strength,
                out_deg=out_deg, in_deg=in_deg, asym=asym)


# ── plots ────────────────────────────────────────────────────────────────────
def make_plots(A, items, stats, out_prefix):
    N = A.shape[0]
    tick = items if items is not None else list(range(N))

    # 1) heatmap
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(A, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="edge weight  A[i,j]  (j -> i)")
    ax.set_title("Learned adjacency")
    ax.set_xlabel("source node j"); ax.set_ylabel("target node i")
    if N <= 80:
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(tick, rotation=90, fontsize=5)
        ax.set_yticklabels(tick, fontsize=5)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_heatmap.png", dpi=150)
    plt.close(fig)

    # 2) degree + strength distributions
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs[0, 0].hist(stats["in_deg"], bins=20, color="#3b76af")
    axs[0, 0].set_title("In-degree distribution"); axs[0, 0].set_xlabel("# incoming neighbours")
    axs[0, 1].hist(stats["out_deg"], bins=20, color="#ef8a43")
    axs[0, 1].set_title("Out-degree distribution"); axs[0, 1].set_xlabel("# outgoing neighbours")
    axs[1, 0].hist(stats["in_strength"], bins=20, color="#519e3e")
    axs[1, 0].set_title("In-strength (sum weight received)"); axs[1, 0].set_xlabel("strength")
    axs[1, 1].hist(A[~np.eye(N, dtype=bool)], bins=40, color="#9467bd")
    axs[1, 1].set_title("Off-diagonal edge-weight distribution"); axs[1, 1].set_xlabel("weight")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_distributions.png", dpi=150)
    plt.close(fig)

    print(f"  saved {out_prefix}_heatmap.png and {out_prefix}_distributions.png")


def export_edge_csv(A, items, out_prefix):
    N = A.shape[0]
    lbl = (lambda i: items[i]) if items is not None else (lambda i: i)
    rows = []
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > 0:
                rows.append(dict(source=lbl(j), target=lbl(i), weight=float(A[i, j])))
    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    df.to_csv(f"{out_prefix}_edges.csv", index=False)
    print(f"  saved {out_prefix}_edges.csv ({len(df)} directed edges)")


# ── main ─────────────────────────────────────────────────────────────────────
def analyze(adj_path, topk, draw_graph, graph_topk, graph_min_weight, target_item):
    A = np.load(adj_path).astype(np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square (N,N) adjacency, got {A.shape}")
    items = load_item_ids()
    if items is not None and len(items) != A.shape[0]:
        print(f"  (item_id count {len(items)} != N {A.shape[0]}; falling back to indices)")
        items = None

    print(f"\n### {os.path.basename(adj_path)}")
    stats = print_stats(A, items, topk)
    out_prefix = os.path.splitext(adj_path)[0]
    make_plots(A, items, stats, out_prefix)
    export_edge_csv(A, items, out_prefix)

    if draw_graph:
        cat_labels = load_cat_labels()
        G = build_networkx_graph(A, items, cat_labels, graph_topk, graph_min_weight)
        tgt = target_item if target_item is not None else None
        plot_networkx_plotly(
            G,
            title=f"{os.path.basename(out_prefix)} — learned graph "
                  f"(top-{graph_topk} incoming/node)",
            save_path=f"{out_prefix}_networkx.html",
            target_node=tgt,
        )
        print(f"  saved {out_prefix}_networkx.html "
              f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def main():
    ap = argparse.ArgumentParser(description="Analyze a learned adjacency matrix.")
    ap.add_argument("--adj", nargs="*", default=None,
                    help="path(s) to *_learned_adjacency.npy "
                         "(default: auto-discover AGCRN/ and MTGNN/)")
    ap.add_argument("--topk", type=int, default=10, help="how many top nodes/edges to list")
    ap.add_argument("--no-graph", dest="graph", action="store_false",
                    help="skip the interactive NetworkX graph (drawn by default)")
    ap.set_defaults(graph=True)
    ap.add_argument("--graph-topk", type=int, default=5,
                    help="incoming edges kept per node when drawing (default 5)")
    ap.add_argument("--graph-min-weight", type=float, default=0.0,
                    help="drop edges below this weight when drawing")
    ap.add_argument("--target-item", type=int, default=None,
                    help="highlight this item_id as the target node in the graph")
    args = ap.parse_args()

    paths = args.adj or default_adj_paths()
    if not paths:
        print("No adjacency files found. Run a pipeline first, or pass --adj <path>.")
        return
    for p in paths:
        analyze(p, args.topk, args.graph, args.graph_topk,
                args.graph_min_weight, args.target_item)


if __name__ == "__main__":
    main()
