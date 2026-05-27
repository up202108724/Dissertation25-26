"""
Draws the Graph2Vec + MLP architecture diagram used in the dissertation.

Two parallel pipelines feed a sliding-window construction block consumed by
a fully-connected MLP:
    1) Offline Graph2Vec pipeline (all-items -> sliding windows -> similarity
       graphs -> Graph2Vec -> per-window graph embeddings).
    2) Online inputs (target series + exogenous calendar features) that are
       concatenated with the embeddings, flattened and passed through dense
       layers to produce a direct multi-step forecast.

Run:
    python draw_architeture.py
Produces:
    graph2vec_mlp_architecture.png
    graph2vec_mlp_architecture.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ----------------------------- style ----------------------------------------
COLORS = {
    "input":   "#DCE9F7",   # light blue   - raw inputs
    "process": "#D6ECD2",   # light green  - preprocessing
    "model":   "#F4CCCC",   # light red    - learned model (Graph2Vec)
    "embed":   "#FFF2CC",   # light yellow - embeddings / projections
    "concat":  "#E0CFEE",   # light purple - feature fusion
    "mlp":     "#C8A2DA",   # purple       - dense layers
    "output":  "#B6D7A8",   # green        - forecast
}
EDGE = "#444444"


def box(ax, xy, w, h, text, color, fontsize=10, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # cx, ybottom, cx, ytop


def arrow(ax, p1, p2, style="-", curve=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color=EDGE,
            connectionstyle=f"arc3,rad={curve}",
            linestyle=style,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.5, 11.5)
    ax.axis("off")

    # ----------- column headers ---------------------------------------------
    ax.text(2.0, 11.05, "Graph2Vec\npipeline", ha="center", va="center",
            fontsize=10, style="italic", color="#555")
    ax.text(6.5, 11.05, "Temporal\ninput", ha="center", va="center",
            fontsize=10, style="italic", color="#555")
    ax.text(11.0, 11.05, "Exogenous\ninput", ha="center", va="center",
            fontsize=10, style="italic", color="#555")

    # Title
    ax.text(6.5, 11.4, "Graph2Vec + MLP Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ----------- LEFT column: Graph2Vec pipeline ----------------------------
    b1 = box(ax, (0.6, 9.6), 2.8, 0.8,
             "All-Items\nTime Series", COLORS["input"])
    b2 = box(ax, (0.6, 8.4), 2.8, 0.8,
             "Sliding Windows\n(width = 15, step = 1)", COLORS["process"])
    b3 = box(ax, (0.6, 7.2), 2.8, 0.8,
             "Similarity Graphs\n(Spearman / DTW / CID ...)", COLORS["process"])
    b4 = box(ax, (0.6, 6.0), 2.8, 0.8,
             "Graph2Vec Model\n(WL Kernel + Doc2Vec)",
             COLORS["model"], bold=True)
    b5 = box(ax, (0.6, 4.8), 2.8, 0.8,
             "Graph Embeddings\n(dim = 20 per window)", COLORS["embed"])

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
        arrow(ax, (a[0], a[1]), (b[2], b[3]))

    # ----------- MIDDLE column: Target series -------------------------------
    t1 = box(ax, (5.0, 9.6), 3.0, 0.8,
             "Target Time Series  (MinMax Scaled)", COLORS["input"])

    # ----------- RIGHT column: Exogenous ------------------------------------
    e1 = box(ax, (9.6, 9.6), 3.0, 0.8,
             "Exogenous Features\n(Calendar, Holidays — 28 cols)",
             COLORS["input"])

    # ----------- Sliding-window construction (fusion) ----------------------
    concat = box(
        ax, (3.2, 3.3), 6.6, 1.0,
        "Sliding-Window Construction\n"
        "[ Target  |  Exog. Features  |  Graph Embeddings ]\n"
        "Window shape:  (lookback = 15,  1 + 28 + 20 = 49 channels)",
        COLORS["concat"],
    )

    # arrows into the fusion block
    arrow(ax, (b5[0], b5[1]), (4.2, 4.3), curve=-0.1)
    arrow(ax, (t1[0], t1[1]), (6.5, 4.3))
    arrow(ax, (e1[0], e1[1]), (8.8, 4.3), curve=0.15)

    # ----------- MLP head ---------------------------------------------------
    flat = box(ax, (3.8, 2.3), 5.4, 0.65,
               "Flatten  →  (15 × 49 = 735 features)",
               COLORS["concat"])
    arrow(ax, (concat[0], concat[1]), (flat[2], flat[3]))

    h1 = box(ax, (3.8, 1.4), 5.4, 0.7,
             "Linear(735 → 64)  +  ReLU  +  Dropout",
             COLORS["mlp"], bold=False)
    arrow(ax, (flat[0], flat[1]), (h1[2], h1[3]))

    h2 = box(ax, (3.8, 0.5), 5.4, 0.7,
             "Linear(64 → 32)  +  ReLU  +  Dropout",
             COLORS["mlp"], bold=False)
    arrow(ax, (h1[0], h1[1]), (h2[2], h2[3]))

    head = box(ax, (3.5, -0.5), 6.0, 0.7,
               "Linear(32 → horizon × 1)  →  reshape (horizon, 1)",
               COLORS["embed"])
    arrow(ax, (h2[0], h2[1]), (head[2], head[3]))

    # Forecast caption
    ax.text(6.5, -1.05, "Forecast  (inverse-scaled)",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#2a4d9b")

    # ----------- Stage separator -------------------------------------------
    ax.plot([0.3, 12.7], [4.55, 4.55], linestyle=(0, (4, 4)),
            color="#888", linewidth=0.8)
    ax.text(0.35, 4.65, "Offline pre-processing  (Graph2Vec)",
            fontsize=8, color="#666")
    ax.text(0.35, 4.35, "Online training / inference  (MLP)",
            fontsize=8, color="#666")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "graph2vec_mlp_architecture.png")
    pdf_path = os.path.join(out_dir, "graph2vec_mlp_architecture.pdf")
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
