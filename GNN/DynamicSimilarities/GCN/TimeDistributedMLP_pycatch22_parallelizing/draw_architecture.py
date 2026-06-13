"""
Draw the architecture of the PER-STEP GCN + Time-Distributed MLP forecaster
(``SimpleGCNMLPForecaster``).

Pipeline (matches gcn_mlp_model.py exactly):

    one ego-graph per lookback day  (B*L graphs)
        -> 2-layer GCN -> LayerNorm -> z_t            (B, L, d_g)
    concat per step [ y_t || exog_t || z_t ]          (B, L, ts_in + d_g)
        -> time-distributed MLP (shared weights over the L steps)   (B, L, H)
        -> pool: cat[ last_step , mean_over_L ]        (B, 2H)
        -> Linear head                                  (B, horizon)

Run:
    python draw_architecture.py
Outputs ``gcn_tdmlp_architecture.png`` (and .pdf) next to this script.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette ────────────────────────────────────────────────────────────────
C_INPUT  = "#AED6F1"   # inputs
C_GCN    = "#F9E79F"   # GCN encoder
C_CONCAT = "#A9DFBF"   # concat
C_TD     = "#FAD7A0"   # time-distributed MLP
C_POOL   = "#D7BDE2"   # pooling
C_HEAD   = "#F5B7B1"   # head / output
C_LOSS   = "#F1948A"   # loss
C_EDGE   = "#555555"


def draw_gcn_tdmlp_architecture(
    seq_len: int = 30,
    n_exog: int = 28,
    in_channels: int = 24,
    gcn_hidden: int = 64,
    d_g: int = 16,
    hidden_sizes: tuple = (128, 64),
    save_path: str = None,
):
    ts_in        = 1 + n_exog
    per_step_in  = ts_in + d_g
    H            = hidden_sizes[-1]

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 17.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(cx, cy, w, h, label, color, fs=9, sub=None, sub_fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04",
            linewidth=1.3, edgecolor=C_EDGE, facecolor=color, zorder=3))
        yo = 0.16 if sub else 0.0
        ax.text(cx, cy + yo, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        if sub:
            ax.text(cx, cy - 0.20, sub, ha="center", va="center",
                    fontsize=sub_fs, color="#333", zorder=4)

    def varrow(x, y0, y1, color=C_EDGE, lw=1.5, text=None, dx=0.25):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=14, lw=lw, color=color, zorder=5))
        if text:
            ax.text(x + dx, (y0 + y1) / 2, text, ha="left", va="center",
                    fontsize=7, color="#333", style="italic")

    def diag(x0, y0, x1, y1, color=C_EDGE, lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, lw=lw, color=color, zorder=5))

    cx     = 5.5      # main column
    cx_g   = 2.7      # GCN column
    cx_t   = 8.3      # temporal column

    # ── (1) Inputs ─────────────────────────────────────────────────────────
    y_inp = 16.4
    box(cx_g, y_inp, 4.2, 0.9, "Ego-Graph  (per step)", C_INPUT,
        sub=f"L = {seq_len} graphs · {in_channels} node feats each")
    box(cx_t, y_inp, 4.0, 0.9, "Target + Exogenous", C_INPUT,
        sub=f"[ y_t ‖ exog_t ]  →  {ts_in} feats/step")

    # tiny ego-graph glyph
    gx, gy = cx_g, y_inp - 1.15
    for nb in [(-0.5, -0.35), (0.5, -0.35), (-0.35, 0.4), (0.4, 0.38)]:
        ax.add_line(Line2D([gx, gx + nb[0]], [gy, gy + nb[1]],
                    color="#7a7a7a", lw=1.0, zorder=2))
        ax.add_patch(Circle((gx + nb[0], gy + nb[1]), 0.08,
                    facecolor="white", edgecolor="#7a7a7a", lw=1.0, zorder=4))
    ax.add_patch(Circle((gx, gy), 0.12, facecolor="#E67E22",
                 edgecolor=C_EDGE, lw=1.3, zorder=5))

    # ── (2) GCN encoder ────────────────────────────────────────────────────
    y_gcn = 13.7
    ax.add_patch(FancyBboxPatch((cx_g - 2.35, y_gcn - 0.75), 4.7, 1.5,
                 boxstyle="round,pad=0.04", linewidth=1.3,
                 edgecolor=C_EDGE, facecolor=C_GCN, zorder=3))
    ax.text(cx_g, y_gcn + 0.45, "GCN Encoder  (per step)",
            ha="center", fontsize=9, fontweight="bold", zorder=4)
    ax.text(cx_g, y_gcn + 0.08, f"GCNConv({in_channels}→{gcn_hidden}) → ReLU → Dropout",
            ha="center", fontsize=7.5, color="#333", zorder=4)
    ax.text(cx_g, y_gcn - 0.25, f"GCNConv({gcn_hidden}→{d_g}) → LayerNorm",
            ha="center", fontsize=7.5, color="#333", zorder=4)
    ax.text(cx_g, y_gcn - 0.55, f"→  zₜ ∈ ℝ^{d_g}   (target node)",
            ha="center", fontsize=7.5, color="#333", zorder=4)
    varrow(cx_g, y_inp - 1.5, y_gcn + 0.78)

    # dashed per-step box
    ax.add_patch(plt.Rectangle((0.2, y_gcn - 0.95), 5.0, (y_inp + 0.55) - (y_gcn - 0.95),
                 linewidth=1.4, edgecolor="#888", facecolor="none",
                 linestyle="--", zorder=2))
    ax.text(0.28, y_inp + 0.62, f"per step  (t = 0 … L−1,  L = {seq_len})",
            ha="left", va="bottom", fontsize=7.5, color="#666")

    # ── (3) Per-step concatenation ─────────────────────────────────────────
    y_cat = 11.8
    diag(cx_g, y_gcn - 0.95, cx - 0.6, y_cat + 0.45)
    diag(cx_t, y_inp - 0.5, cx + 0.6, y_cat + 0.45)
    box(cx, y_cat, 8.2, 0.9, "Per-step Concatenation", C_CONCAT,
        sub=f"[ zₜ ‖ yₜ ‖ exogₜ ]  →  (B, {seq_len}, {d_g}+{ts_in}) = (B, {seq_len}, {per_step_in})")

    # ── (4) Time-distributed MLP ───────────────────────────────────────────
    y_td = 10.0
    varrow(cx, y_cat - 0.45, y_td + 0.55)
    sub_layers = " → ".join(f"Linear({a}→{b})·ReLU·Drop"
                            for a, b in zip([per_step_in, *hidden_sizes[:-1]], hidden_sizes))
    box(cx, y_td, 8.2, 1.0, "Time-Distributed MLP  (shared weights over L steps)", C_TD,
        fs=8.5, sub=sub_layers, sub_fs=7)
    ax.text(cx, y_td - 0.62, f"applied to each of the {seq_len} steps  →  (B, {seq_len}, {H})",
            ha="center", fontsize=7, color="#333", style="italic", zorder=4)

    # ── (5) Pooling: cat[last, mean] ───────────────────────────────────────
    y_pool = 8.1
    varrow(cx - 1.6, y_td - 0.65, y_pool + 0.45)
    varrow(cx + 1.6, y_td - 0.65, y_pool + 0.45)
    box(cx, y_pool, 8.2, 0.9, "Temporal Pooling", C_POOL,
        sub=f"concat[ last_step (B,{H})  ‖  mean_over_L (B,{H}) ]  →  (B, {2*H})")

    # ── (6) Linear head ────────────────────────────────────────────────────
    y_head = 6.5
    varrow(cx, y_pool - 0.45, y_head + 0.4)
    box(cx, y_head, 5.2, 0.8, f"Linear({2*H} → horizon)", C_HEAD)

    # ── (7) Forecast ───────────────────────────────────────────────────────
    y_out = 5.1
    varrow(cx, y_head - 0.4, y_out + 0.4)
    box(cx, y_out, 3.6, 0.8, "Forecast  ŷ(t+1)", C_CONCAT, fs=10,
        sub="(recursive roll-out)")

    # ── (8) Loss ───────────────────────────────────────────────────────────
    y_loss = 3.7
    varrow(cx, y_out - 0.4, y_loss + 0.4)
    box(cx, y_loss, 4.6, 0.75, "MSE Loss   ℒ = ‖ŷ − y‖²", C_LOSS, fs=8.5)

    # ── backprop annotation ────────────────────────────────────────────────
    bp_x = 10.3
    ax.add_patch(FancyArrowPatch((bp_x, y_loss), (bp_x, y_gcn + 0.5),
                 arrowstyle="-|>", mutation_scale=16, lw=2.0,
                 color="#C0392B", zorder=5))
    ax.text(bp_x + 0.18, (y_gcn + y_loss) / 2,
            "Back-\nprop\n(∂ℒ/∂θ)\njointly", ha="center", va="center",
            fontsize=8, color="#C0392B", fontweight="bold")

    ax.set_title("GCN + Time-Distributed MLP  (per-step ego-graphs, jointly trained)",
                 fontsize=13, fontweight="bold", pad=14)

    legend = [
        mpatches.Patch(color=C_INPUT,  label="Inputs"),
        mpatches.Patch(color=C_GCN,    label="GCN encoder"),
        mpatches.Patch(color=C_CONCAT, label="Concat / output"),
        mpatches.Patch(color=C_TD,     label="Time-distributed MLP"),
        mpatches.Patch(color=C_POOL,   label="Pooling (last‖mean)"),
        mpatches.Patch(color=C_HEAD,   label="Linear head"),
        mpatches.Patch(color=C_LOSS,   label="Loss"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=8, frameon=True, edgecolor="#ccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"Saved to {save_path} (+ .pdf)")
    else:
        plt.show()


if __name__ == "__main__":
    # Defaults mirror main_gcn_mlp.py: GCN_HIDDEN_SIZE=64, d_g=16,
    # HIDDEN_SIZES=(128, 64), 1 + len(EXOG_COLS) temporal feats,
    # catch24 node features (24-d).
    draw_gcn_tdmlp_architecture(
        seq_len=30, n_exog=28, in_channels=24,
        gcn_hidden=64, d_g=16, hidden_sizes=(128, 64),
        save_path=os.path.join(SCRIPT_DIR, "gcn_tdmlp_architecture.png"),
    )
