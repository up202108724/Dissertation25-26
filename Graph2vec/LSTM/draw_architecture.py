"""
Draws the Graph2Vec + LSTM architecture diagram used in the dissertation.

Two pipelines feed a per-step feature-concatenation block consumed by an LSTM:
    1) OFFLINE Graph2Vec pipeline (all-items -> sliding windows -> similarity
       graphs -> Graph2Vec [WL kernel + Doc2Vec] -> per-window graph
       embeddings).  This stage is trained UNSUPERVISED and FROZEN: no gradient
       from the forecasting loss ever reaches it.
    2) ONLINE temporal pipeline (target series + exogenous calendar features),
       fused per step with the (frozen) embeddings and trained jointly with the
       LSTM head under the forecasting loss.

The offline/online separation is the key conceptual contrast with the
end-to-end GCN+LSTM model, and the diagram makes it explicit (solid arrows =
forward/trainable path, dashed grey arrow = frozen embedding injected as a
fixed input).

Run:
    python draw_architecture.py
Produces:
    graph2vec_lstm_architecture.png  (+ .pdf)
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
    "lstm":    "#C8A2DA",   # purple       - LSTM block
    "output":  "#B6D7A8",   # green        - forecast
    "loss":    "#F1948A",   # salmon       - loss
}
EDGE   = "#444444"
FROZEN = "#8a8a8a"


def box(ax, xy, w, h, text, color, fontsize=10, bold=False, ec=EDGE, ls="-"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3, edgecolor=ec, facecolor=color, linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", wrap=True,
    )
    return {
        "cx": x + w / 2, "cy": y + h / 2,
        "top": (x + w / 2, y + h), "bot": (x + w / 2, y),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
    }


def arrow(ax, p1, p2, style="-", curve=0.0, color=EDGE, lw=1.3, label=None,
          label_dxy=(0.0, 0.12), scale=14):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={curve}", linestyle=style,
    ))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_dxy[0], my + label_dxy[1], label, ha="center",
                va="center", fontsize=7.5, color=color, style="italic")


def main(
    window_size: int = 30,
    seq_len: int = 30,
    n_exog: int = 31,
    emb_dim: int = 20,
    wl_iters: int = 2,
    hidden: int = 64,
    num_layers: int = 1,
):
    in_width = 1 + n_exog + emb_dim

    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.0, 11.4)
    ax.axis("off")

    ax.text(6.5, 11.15, "Graph2Vec + LSTM Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # column headers
    ax.text(2.0, 10.55, "Graph2Vec pipeline\n(offline · unsupervised · frozen)",
            ha="center", va="center", fontsize=9, style="italic", color="#555")
    ax.text(6.5, 10.55, "Temporal input", ha="center", va="center",
            fontsize=9, style="italic", color="#555")
    ax.text(11.0, 10.55, "Exogenous input", ha="center", va="center",
            fontsize=9, style="italic", color="#555")

    # ----------- LEFT column: Graph2Vec pipeline ----------------------------
    b1 = box(ax, (0.6, 9.2), 2.8, 0.8, "All-Items\nTime Series", COLORS["input"], 9)
    b2 = box(ax, (0.6, 8.0), 2.8, 0.8,
             f"Sliding Windows\n(width = {window_size}, step = 1)", COLORS["process"], 9)
    b3 = box(ax, (0.6, 6.8), 2.8, 0.8,
             "Per-Window Similarity Graphs\n(Spearman, thr = 0.70)", COLORS["process"], 9)
    b4 = box(ax, (0.6, 5.6), 2.8, 0.8,
             f"Graph2Vec\n(WL kernel, {wl_iters} iters + Doc2Vec)",
             COLORS["model"], 9, bold=True)
    b5 = box(ax, (0.6, 4.4), 2.8, 0.8,
             f"Per-Window Graph Embeddings\n(dim = {emb_dim})", COLORS["embed"], 9)

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
        arrow(ax, a["bot"], b["top"])
    ax.text(3.6, 5.05, "frozen\n(no grad)", ha="left", va="center",
            fontsize=7.5, color=FROZEN, style="italic", fontweight="bold")

    # ----------- MIDDLE / RIGHT inputs --------------------------------------
    t1 = box(ax, (5.0, 9.2), 3.0, 0.8,
             "Target Time Series\n(MinMax scaled)", COLORS["input"], 9)
    e1 = box(ax, (9.4, 9.2), 3.2, 0.8,
             f"Exogenous Features\n(calendar / holiday / lags — {n_exog} cols)",
             COLORS["input"], 9)

    # ----------- Feature concatenation --------------------------------------
    concat = box(
        ax, (3.0, 2.9), 7.0, 1.15,
        "Per-step Feature Concatenation\n"
        "[ y$_t$  ‖  exog$_t$  ‖  graph-emb$_t$ ]\n"
        f"per-step width: 1 + {n_exog} + {emb_dim} = {in_width}   "
        f"→   (B, L={seq_len}, {in_width})",
        COLORS["concat"], 9.5,
    )

    # arrows into concat
    arrow(ax, b5["bot"], (4.0, 4.05), curve=-0.12, color=FROZEN, style=(0, (5, 3)),
          label="fixed input", label_dxy=(-0.7, 0.0))
    arrow(ax, t1["bot"], (6.3, 4.05))
    arrow(ax, e1["bot"], (8.8, 4.05), curve=0.12)

    # per-window -> per-step alignment note
    ax.text(6.5, 2.55,
            f"embeddings aligned per step (first {window_size} steps zero-padded)",
            ha="center", va="center", fontsize=7.5, color="#666", style="italic")

    # ----------- LSTM -------------------------------------------------------
    lstm = box(ax, (3.0, 1.45), 7.0, 0.85,
               f"LSTM  (hidden = {hidden},  layers = {num_layers},  batch_first)",
               COLORS["lstm"], 10, bold=True)
    arrow(ax, concat["bot"], lstm["top"])

    head = box(ax, (3.7, 0.45), 5.6, 0.6,
               f"Dropout → Linear({hidden} → 1)   [last hidden state]",
               COLORS["embed"], 9)
    arrow(ax, lstm["bot"], head["top"])

    out = box(ax, (4.9, -0.6), 3.2, 0.65,
              "Forecast t+1  (inverse-scaled)", COLORS["output"], 9.5, bold=True)
    arrow(ax, head["bot"], out["top"])

    # loss + backprop (only through the online branch)
    loss = box(ax, (9.6, -0.6), 2.6, 0.65, "MSE Loss", COLORS["loss"], 9.5, bold=True)
    arrow(ax, out["right"], loss["left"])
    arrow(ax, loss["top"], (10.9, 1.45), color="#C0392B", curve=-0.25, lw=1.8,
          label="$\\partial L/\\partial\\theta$ (LSTM only)", label_dxy=(1.0, 0.2))

    # ----------- Offline / online separator ---------------------------------
    ax.plot([0.3, 12.7], [4.22, 4.22], linestyle=(0, (4, 4)),
            color="#888", linewidth=0.9)
    ax.text(0.35, 4.36, "Offline pre-processing  (Graph2Vec — frozen)",
            fontsize=8, color="#666")
    ax.text(0.35, 4.04, "Online training / inference  (LSTM — trainable)",
            fontsize=8, color="#666")

    # ----------- legend -----------------------------------------------------
    ax.add_patch(FancyArrowPatch((0.6, -0.85), (1.3, -0.85), arrowstyle="-|>",
                 mutation_scale=12, color=EDGE, lw=1.3))
    ax.text(1.45, -0.85, "forward / trainable", fontsize=7.5, va="center")
    ax.add_patch(FancyArrowPatch((3.5, -0.85), (4.2, -0.85), arrowstyle="-|>",
                 mutation_scale=12, color=FROZEN, lw=1.3, linestyle=(0, (5, 3))))
    ax.text(4.35, -0.85, "frozen embedding (fixed input)", fontsize=7.5, va="center")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "graph2vec_lstm_architecture.png")
    pdf_path = os.path.join(out_dir, "graph2vec_lstm_architecture.pdf")
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
