"""
Architecture diagram for the **GCN + LSTM** forecaster — a simplified,
end-to-end-trainable replacement for the Graph2Vec + LSTM pipeline.

Layout intentionally mirrors ``Graph2vec_FixedThreshold/LSTM/draw_architecture.py``
so the two diagrams are visually comparable in the dissertation.

Run:
    python draw_architecture.py
Produces:
    gcn_lstm_architecture.png
    gcn_lstm_architecture.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ----------------------------- style ----------------------------------------
COLORS = {
    "input":   "#DCE9F7",   # light blue   - raw inputs
    "process": "#D6ECD2",   # light green  - preprocessing
    "model":   "#F4CCCC",   # light red    - learned model (GCN)
    "embed":   "#FFF2CC",   # light yellow - embeddings / projections
    "concat":  "#E0CFEE",   # light purple - feature fusion
    "lstm":    "#C8A2DA",   # purple       - LSTM block
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


def arrow(ax, p1, p2, style="-", curve=0.0, color=EDGE):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color=color,
            connectionstyle=f"arc3,rad={curve}",
            linestyle=style,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 11)
    ax.axis("off")

    # ----------- column headers ---------------------------------------------
    ax.text(2.0, 10.6, "GCN graph\npipeline", ha="center", va="center",
            fontsize=10, style="italic", color="#555")
    ax.text(6.5, 10.6, "Temporal\ninput", ha="center", va="center",
            fontsize=10, style="italic", color="#555")
    ax.text(11.0, 10.6, "Exogenous\ninput", ha="center", va="center",
            fontsize=10, style="italic", color="#555")

    # Title
    ax.text(6.5, 10.95, "GCN + LSTM Architecture (jointly trained)",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ----------- LEFT column: GCN pipeline ----------------------------------
    b1 = box(ax, (0.6, 9.2), 2.8, 0.8,
             "All-Items\nTime Series", COLORS["input"])
    b2 = box(ax, (0.6, 8.0), 2.8, 0.8,
             "Sliding Windows\n(width = 15, step = 1)", COLORS["process"])
    b3 = box(ax, (0.6, 6.8), 2.8, 0.8,
             "Similarity Graph G_t\n(Spearman / DTW / CID ...)", COLORS["process"])
    b4 = box(ax, (0.6, 5.6), 2.8, 0.8,
             "Ego-graph + node feats\n(target + k neighbours)",
             COLORS["embed"])
    b5 = box(ax, (0.6, 4.4), 2.8, 0.8,
             "GCN Encoder (jointly trained)\nGCNConv → ReLU → GCNConv → z",
             COLORS["model"], bold=True)

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
        arrow(ax, (a[0], a[1]), (b[2], b[3]))

    # ----------- MIDDLE column: Target series -------------------------------
    t1 = box(ax, (5.0, 9.2), 3.0, 0.8,
             "Target Time Series  (MinMax Scaled)", COLORS["input"])

    # ----------- RIGHT column: Exogenous ------------------------------------
    e1 = box(ax, (9.6, 9.2), 3.0, 0.8,
             "Exogenous Features\n(Calendar, Holidays — 28 cols)",
             COLORS["input"])

    # ----------- Feature concatenation --------------------------------------
    concat = box(
        ax, (3.2, 3.0), 6.6, 1.0,
        "Feature Concatenation per step\n"
        "[ Target  |  Exog. Features  |  z  (broadcast over L) ]\n"
        "Input shape per step:  (L = 30,  1 + 28 + d_g = 1 + 28 + 16 = 45)",
        COLORS["concat"],
    )

    # arrows into concat
    arrow(ax, (b5[0], b5[1]), (4.2, 4.0), curve=-0.1)   # z (left)
    arrow(ax, (t1[0], t1[1]), (6.5, 4.0))               # target (middle)
    arrow(ax, (e1[0], e1[1]), (8.8, 4.0), curve=0.15)   # exog (right)

    # ----------- LSTM -------------------------------------------------------
    lstm = box(ax, (3.2, 1.7), 6.6, 0.9,
               "LSTM  (hidden = 32,  num_layers = 1,  batch_first = True)",
               COLORS["lstm"], bold=True)
    arrow(ax, (concat[0], concat[1]), (lstm[2], lstm[3]))

    # Dropout + Linear head
    head = box(ax, (3.8, 0.7), 5.4, 0.6,
               "Dropout  →  Linear (hidden → 1)   [ last hidden state ]",
               COLORS["embed"])
    arrow(ax, (lstm[0], lstm[1]), (head[2], head[3]))

    # Forecast
    out = box(ax, (5.0, -0.3), 3.0, 0.7,
              "Forecast  t+1\n(inverse-scaled, recursive at inference)",
              COLORS["output"], bold=True)
    arrow(ax, (head[0], head[1]), (out[2], out[3]))

    # ----------- Joint-training gradient feedback (dashed green) -----------
    grad_color = "#2E7D32"
    # loss -> LSTM
    arrow(ax, (out[0] + 1.2, out[3] - 0.1), (lstm[0] + 2.6, lstm[1] + 0.1),
          style=(0, (4, 3)), curve=0.3, color=grad_color)
    # LSTM -> concat -> GCN
    arrow(ax, (lstm[0] - 2.6, lstm[3] + 0.05), (b5[0] + 0.6, b5[1] + 0.05),
          style=(0, (4, 3)), curve=0.35, color=grad_color)
    ax.text(11.6, 1.4, "Forecasting-loss\ngradient flow\n(joint training)",
            ha="center", va="center", fontsize=8, color=grad_color, style="italic")

    # ----------- Stage separator -------------------------------------------
    ax.plot([0.3, 12.7], [4.25, 4.25], linestyle=(0, (4, 4)),
            color="#888", linewidth=0.8)
    ax.text(0.35, 4.35, "Per-batch graph construction  (online)",
            fontsize=8, color="#666")
    ax.text(0.35, 4.05, "End-to-end training / recursive inference  (GCN + LSTM)",
            fontsize=8, color="#666")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "gcn_lstm_architecture.png")
    pdf_path = os.path.join(out_dir, "gcn_lstm_architecture.pdf")
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
