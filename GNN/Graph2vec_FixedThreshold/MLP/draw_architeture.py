import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─── colour palette ───────────────────────────────────────────────────────────
C_DATA   = "#D6EAF8"   # light blue   – raw data
C_GRAPH  = "#D5F5E3"   # light green  – graph construction
C_G2V    = "#FADBD8"   # light red    – Graph2Vec
C_EMB    = "#FEF9E7"   # pale yellow  – embeddings / features
C_EXOG   = "#EBF5FB"   # very light blue – exogenous
C_FLAT   = "#E8DAEF"   # lavender     – flatten
C_HIDDEN = "#D7BDE2"   # purple       – hidden layers
C_OUT    = "#FCF3CF"   # soft yellow  – output head
C_FORE   = "#A9DFBF"   # mint green   – forecast
EDGE     = "#2C3E50"
# ──────────────────────────────────────────────────────────────────────────────


def fbox(ax, x, y, w, h, text, fc, fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=EDGE, linewidth=1.6, zorder=3
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")


def arr(ax, x0, y0, x1, y1, style="arc3,rad=0.0"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=EDGE, lw=1.6,
                    connectionstyle=style, mutation_scale=14
                ), zorder=2)


def draw_architecture():
    fig, ax = plt.subplots(figsize=(17, 14))
    ax.set_xlim(0, 17)
    ax.set_ylim(-1.4, 15.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FDFEFE")
    ax.set_facecolor("#FDFEFE")

    ax.text(8.5, 15.0, "Graph2Vec  +  MLP  Architecture",
            ha="center", va="center", fontsize=16,
            fontweight="bold", color=EDGE)

    # ── box geometry ─────────────────────────────────────────────────────────
    BW  = 3.3    # standard box width
    BH  = 0.85   # standard box height
    CBW = 8.8    # central wide box width
    CBX = 4.1    # x-start of central blocks
    MCX = CBX + CBW / 2   # ≈ 8.5  – centre x of central column

    # ── LEFT COLUMN  (Graph2Vec pipeline) ────────────────────────────────────
    LX  = 0.15
    LC  = LX + BW / 2

    left_ys   = [13.3, 11.5, 9.7, 7.9, 6.1]
    left_meta = [
        ("All-Items\nTime Series",                   C_DATA,  False),
        ("Sliding Windows\n(width=15, step=1)",       C_GRAPH, False),
        ("Similarity Graphs\n(Spearman / DTW / CID …)", C_GRAPH, False),
        ("Graph2Vec Model\n(WL Kernel  +  Doc2Vec)",  C_G2V,   True),
        ("Graph Embeddings\n(dim = 20 per window)",   C_EMB,   False),
    ]

    for (txt, col, bold), y in zip(left_meta, left_ys):
        fbox(ax, LX, y, BW, BH, txt, fc=col, bold=bold)

    for i in range(len(left_ys) - 1):
        arr(ax, LC, left_ys[i], LC, left_ys[i + 1] + BH)

    # ── CENTRE COLUMN  (Target time series) ──────────────────────────────────
    TC = CBX + 1.5 + (BW + 0.2) / 2    # ≈ 7.75
    fbox(ax, CBX + 1.5, 13.3, BW + 0.2, BH,
         "Target Time Series  (MinMax Scaled)", C_DATA)

    # ── RIGHT COLUMN  (Exogenous features) ───────────────────────────────────
    RX = 12.9
    RC = RX + BW / 2
    fbox(ax, RX, 13.3, BW + 0.4, BH,
         "Exogenous Features\n(Calendar, Holidays — 28 cols)", C_EXOG)

    # ── WINDOW CONSTRUCTION (merge point) ────────────────────────────────────
    WIN_Y = 4.5
    fbox(ax, CBX, WIN_Y, CBW, BH + 0.3,
         "Sliding-Window Construction\n"
         "[ Target  |  Exog. Features  |  Graph Embeddings ]\n"
         "Window shape:  (lookback=15,  1 + 28 + 20 = 49  channels)",
         C_FLAT, fontsize=9, bold=True)

    arr(ax, LC,  left_ys[-1],  CBX + 0.3,        WIN_Y + (BH + 0.3) / 2,  style="arc3,rad=-0.25")
    arr(ax, TC,  13.3,         CBX + CBW / 2,    WIN_Y + BH + 0.3,         style="arc3,rad=0.0")
    arr(ax, RC,  13.3,         CBX + CBW - 0.3,  WIN_Y + (BH + 0.3) / 2,  style="arc3,rad=0.25")

    # ── FLATTEN ───────────────────────────────────────────────────────────────
    FLAT_Y = 3.0
    fbox(ax, CBX, FLAT_Y, CBW, BH,
         "Flatten   →   (15 × 49 = 735  features)",
         C_FLAT, bold=False)
    arr(ax, MCX, WIN_Y, MCX, FLAT_Y + BH)

    # ── HIDDEN LAYER 1 ────────────────────────────────────────────────────────
    H1_Y = 1.7
    fbox(ax, CBX, H1_Y, CBW, BH,
         "Linear(735 → 64)   +   ReLU   +   Dropout",
         C_HIDDEN, bold=False)
    arr(ax, MCX, FLAT_Y, MCX, H1_Y + BH)

    # ── HIDDEN LAYER 2 ────────────────────────────────────────────────────────
    H2_Y = 0.4
    fbox(ax, CBX, H2_Y, CBW, BH,
         "Linear(64 → 32)   +   ReLU   +   Dropout",
         C_HIDDEN, bold=False)
    arr(ax, MCX, H1_Y, MCX, H2_Y + BH)

    # ── OUTPUT HEAD ───────────────────────────────────────────────────────────
    OUT_Y = -0.9
    fbox(ax, CBX, OUT_Y, CBW, BH,
         "Linear(32 → horizon × 1)   →   reshape  (horizon, 1)",
         C_OUT, bold=False)
    arr(ax, MCX, H2_Y, MCX, OUT_Y + BH)

    # ── FORECAST ─────────────────────────────────────────────────────────────
    FORE_W = 3.6
    FORE_X = MCX - FORE_W / 2
    # no extra box below – label on the arrow
    ax.text(MCX, OUT_Y - 0.45, "Forecast  (inverse-scaled)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1A5276", zorder=4)

    # ── SECTION LABELS ────────────────────────────────────────────────────────
    for sx, stxt in [(LC, "Graph2Vec\nPipeline"), (TC, "Temporal\nInput"), (RC, "Exogenous\nInput")]:
        ax.text(sx, 14.5, stxt, ha="center", va="center",
                fontsize=8.5, color="#7F8C8D", style="italic")

    # dashed separator: offline vs online
    ax.axhline(y=5.85, xmin=0.01, xmax=0.99, color="#BDC3C7", lw=1, linestyle="--", zorder=1)
    ax.text(0.05, 6.0,  "↑  Offline pre-processing  (Graph2Vec)", fontsize=7.5, color="#95A5A6")
    ax.text(0.05, 5.55, "↓  Online training / inference  (MLP)",  fontsize=7.5, color="#95A5A6")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "architecture_graph2vec_mlp.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    draw_architecture()
