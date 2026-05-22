import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─── colour palette ───────────────────────────────────────────────────────────
C_DATA      = "#D6EAF8"   # light blue  – raw data
C_GRAPH     = "#D5F5E3"   # light green – graph construction
C_G2V       = "#FADBD8"   # light red   – Graph2Vec
C_EMB       = "#FEF9E7"   # pale yellow – embeddings / features
C_EXOG      = "#EBF5FB"   # very light blue – exogenous
C_CONCAT    = "#E8DAEF"   # lavender    – concatenation
C_LSTM      = "#D7BDE2"   # purple      – LSTM
C_LINEAR    = "#FCF3CF"   # soft yellow – linear head
C_OUTPUT    = "#A9DFBF"   # mint green  – output
EDGE        = "#2C3E50"   # dark navy
# ──────────────────────────────────────────────────────────────────────────────

def fancy_box(ax, x, y, w, h, text, fc, fontsize=9.5, bold=False):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=EDGE, linewidth=1.6, zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        zorder=4, wrap=True,
        multialignment="center"
    )


def arrow(ax, x0, y0, x1, y1, label="", style="arc3,rad=0.0"):
    """Draw a directed arrow between two points."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=EDGE, lw=1.6,
            connectionstyle=style,
            mutation_scale=14
        ),
        zorder=2
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.05, my, label, fontsize=8, color="#555555", zorder=5)


def draw_architecture():
    fig, ax = plt.subplots(figsize=(17, 14))
    ax.set_xlim(0, 17)
    ax.set_ylim(-1.2, 15.5)
    ax.axis("off")

    fig.patch.set_facecolor("#FDFEFE")
    ax.set_facecolor("#FDFEFE")

    ax.text(
        8.5, 15.0,
        "Graph2Vec  +  LSTM  Architecture",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=EDGE
    )

    # ── box dimensions ────────────────────────────────────────────────────────
    BW = 3.3    # standard box width
    BH = 0.85   # standard box height
    CBW = 8.8   # wide box width for central row
    CBX = 4.1   # x start of central boxes

    # ── LEFT COLUMN  (Graph2Vec pipeline)  x-centre ≈ 1.8 ────────────────────
    LX = 0.15   # left edge of left column boxes
    LC = LX + BW / 2  # centre x

    left_ys = [13.3, 11.5, 9.7, 7.9, 6.1]   # bottom-y of each box

    left_boxes = [
        ("All-Items\nTime Series",        C_DATA,  False),
        ("Sliding Windows\n(width=15, step=1)", C_GRAPH, False),
        ("Similarity Graphs\n(Spearman / DTW / CID …)", C_GRAPH, False),
        ("Graph2Vec Model\n(WL Kernel  +  Doc2Vec)", C_G2V, True),
        ("Graph Embeddings\n(dim = 20 per window)", C_EMB,  False),
    ]

    for (txt, col, bold), y in zip(left_boxes, left_ys):
        fancy_box(ax, LX, y, BW, BH, txt, fc=col, bold=bold)

    for i in range(len(left_ys) - 1):
        arrow(ax, LC, left_ys[i], LC, left_ys[i + 1] + BH)   # bottom → top of next

    # ── CENTRE COLUMN  (Target time series)  x-centre ≈ 8.5 ──────────────────
    MCX = CBX + CBW / 2  # ≈ 8.5

    fancy_box(ax, CBX + 1.5, 13.3, BW + 0.2, BH,
              "Target Time Series  (MinMax Scaled)", C_DATA, bold=False)
    TC = CBX + 1.5 + (BW + 0.2) / 2   # ≈ 7.75

    # ── RIGHT COLUMN  (Exogenous features)  x-centre ≈ 14.6 ──────────────────
    RX = 12.9
    RC = RX + BW / 2

    fancy_box(ax, RX, 13.3, BW + 0.4, BH,
              "Exogenous Features\n(Calendar, Holidays — 28 cols)", C_EXOG, bold=False)

    # ── CONCATENATION ROW  y ≈ 4.3 ───────────────────────────────────────────
    CAT_Y = 4.3
    fancy_box(ax, CBX, CAT_Y, CBW, BH + 0.3,
              "Feature Concatenation\n"
              "[ Target  |  Exog. Features  |  Graph Embeddings ]\n"
              "Input shape per step:  (seq_len=30,  1 + 28 + 20 = 49)",
              C_CONCAT, fontsize=9, bold=True)

    # arrows from left / centre / right → concat
    arrow(ax, LC,  left_ys[-1],       CBX + 0.3, CAT_Y + (BH + 0.3) / 2,
          style="arc3,rad=-0.25")                           # from embedding
    arrow(ax, TC,  13.3,              CBX + CBW / 2, CAT_Y + BH + 0.3,
          style="arc3,rad=0.0")                             # from target (straight down)
    arrow(ax, RC,  13.3,              CBX + CBW - 0.3, CAT_Y + (BH + 0.3) / 2,
          style="arc3,rad=0.25")                            # from exog

    # vertical guide line for centre column: target → concat
    # (already covered by the centre arrow above)

    # ── LSTM ROW  y ≈ 2.7 ────────────────────────────────────────────────────
    LSTM_Y = 2.7
    fancy_box(ax, CBX, LSTM_Y, CBW, BH,
              "LSTM  (hidden = 32,  num_layers = 1,  batch_first = True)",
              C_LSTM, bold=True)
    arrow(ax, MCX, CAT_Y, MCX, LSTM_Y + BH)

    # ── DROPOUT + LINEAR ROW  y ≈ 1.3 ────────────────────────────────────────
    LIN_Y = 1.3
    fancy_box(ax, CBX, LIN_Y, CBW, BH,
              "Dropout  →  Linear (hidden → 1)   [ last hidden state ]",
              C_LINEAR, bold=False)
    arrow(ax, MCX, LSTM_Y, MCX, LIN_Y + BH)

    # ── OUTPUT  y ≈ -0.5 ─────────────────────────────────────────────────────
    OUT_W = 3.2
    OUT_X = MCX - OUT_W / 2
    OUT_Y = -0.55
    fancy_box(ax, OUT_X, OUT_Y, OUT_W, BH,
              "Forecast  t + 1\n(inverse-scaled)", C_OUTPUT, bold=True, fontsize=10)
    arrow(ax, MCX, LIN_Y, MCX, OUT_Y + BH)

    # ── LEGEND / SECTION LABELS ───────────────────────────────────────────────
    sections = [
        (LC,  14.5, "Graph2Vec\nPipeline"),
        (TC,  14.5, "Temporal\nInput"),
        (RC,  14.5, "Exogenous\nInput"),
    ]
    for sx, sy, stxt in sections:
        ax.text(sx, sy, stxt, ha="center", va="center",
                fontsize=8.5, color="#7F8C8D", style="italic")

    # dashed separators
    ax.axhline(y=5.6, xmin=0.01, xmax=0.99, color="#BDC3C7", lw=1, linestyle="--", zorder=1)
    ax.text(0.05, 5.75, "↑  Offline pre-processing  (Graph2Vec)",
            fontsize=7.5, color="#95A5A6")
    ax.text(0.05, 5.35, "↓  Online training / inference  (LSTM)",
            fontsize=7.5, color="#95A5A6")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "architecture_graph2vec_lstm.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    draw_architecture()