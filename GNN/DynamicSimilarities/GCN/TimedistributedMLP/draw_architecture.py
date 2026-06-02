import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle


def draw_gcn_mlp_architecture(
    seq_len: int = 30,
    n_exog: int = 28,
    in_channels: int = 8,
    gcn_hidden: int = 32,
    d_g: int = 16,
    mlp_hidden: tuple = (64, 32),
    save_path: str = None,
):
    """
    Draws the GCN + MLP per-step forecaster architecture.

    Parameters
    ----------
    seq_len     : lookback window length  (default 30)
    n_exog      : number of exogenous features  (default 28)
    in_channels : GCN node-feature dimension  (default 8)
    gcn_hidden  : GCNConv₁ output width  (default 32)
    d_g         : GCNConv₂ output width / embedding dim  (default 16)
    mlp_hidden  : tuple of MLP hidden layer widths  (default (64, 32))
    save_path   : path to save PNG; if None, shows interactively
    """

    fig, ax = plt.subplots(figsize=(9, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ------------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------------
    C_INPUT  = '#AED6F1'   # light blue  – inputs
    C_GCN    = '#F9E79F'   # yellow      – GCN encoder
    C_CONCAT = '#A9DFBF'   # green       – concat / output
    C_FLAT   = '#D5DBDB'   # grey        – flatten / reshape
    C_MLP    = '#FAD7A0'   # pale orange – MLP layers
    C_LOSS   = '#F1948A'   # salmon      – loss
    C_EDGE   = '#555555'

    ts_input_size = 1 + n_exog
    flat_per_step = ts_input_size + d_g
    flat_total    = seq_len * flat_per_step

    # ------------------------------------------------------------------
    # Helper: rounded box
    # ------------------------------------------------------------------
    def box(cx, cy, w, h, label, color, fontsize=9):
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor=C_EDGE,
            facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=4)

    def subtext(cx, cy, text, fontsize=7.5):
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, color='#333333', zorder=4)

    # Helper: straight vertical arrow
    def arrow(x, y_start, y_end, color=C_EDGE):
        ax.annotate('', xy=(x, y_end + 0.02), xytext=(x, y_start - 0.02),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4),
                    zorder=5)

    # Helper: diagonal arrow between two points
    def diag_arrow(x1, y1, x2, y2, color=C_EDGE):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                    zorder=5)

    cx = 5.0   # horizontal centre of main column

    # ------------------------------------------------------------------
    # (1) Top inputs
    # ------------------------------------------------------------------
    y_inp  = 15.5
    cx_gcn = 2.5   # left column centre (ego-graph / GCN)
    cx_ts  = 7.5   # right column centre (target + exog)

    box(cx_gcn, y_inp, 4.2, 0.9, 'Ego-Graph  (per step)', C_INPUT)
    subtext(cx_gcn, y_inp - 0.26,
            f'L = {seq_len} graphs  |  {in_channels} node features each')

    box(cx_ts, y_inp, 4.2, 0.9, 'Target  +  Exogenous', C_INPUT)
    subtext(cx_ts, y_inp - 0.26,
            f'[ y_t  ‖  exog_t ]  —  {ts_input_size} features/step')

    # ------------------------------------------------------------------
    # (2) GCN Encoder  (left column, inside dashed "per-step" box)
    # ------------------------------------------------------------------
    y_gcn  = 13.6
    gcn_h  = 1.35

    # Draw GCN box manually (3 lines of text)
    rect_gcn = FancyBboxPatch(
        (cx_gcn - 2.3, y_gcn - gcn_h / 2), 4.6, gcn_h,
        boxstyle="round,pad=0.05",
        linewidth=1.2, edgecolor=C_EDGE,
        facecolor=C_GCN, zorder=3,
    )
    ax.add_patch(rect_gcn)
    ax.text(cx_gcn, y_gcn + 0.33, 'GCN Encoder  (per step)',
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', zorder=4)
    ax.text(cx_gcn, y_gcn - 0.03,
            f'GCNConv({in_channels}\u2192{gcn_hidden}) \u2192 ReLU \u2192 Dropout',
            ha='center', va='center', fontsize=7.5, color='#333333', zorder=4)
    ax.text(cx_gcn, y_gcn - 0.38,
            f'GCNConv({gcn_hidden}\u2192{d_g}) \u2192 LayerNorm  \u2192  z\u209c  \u2208 \u211d\u207f  (n={d_g})',
            ha='center', va='center', fontsize=7.5, color='#333333', zorder=4)

    # Arrow: ego-graph input → GCN encoder
    arrow(cx_gcn, y_inp, y_gcn)

    # ------------------------------------------------------------------
    # Dashed "per-step" rectangle around ego-graph + GCN
    # ------------------------------------------------------------------
    y_ps_bot = y_gcn - gcn_h / 2 - 0.1
    y_ps_top = y_inp + 0.55
    rect_ps = Rectangle(
        (0.18, y_ps_bot), 5.0, y_ps_top - y_ps_bot,
        linewidth=1.5, edgecolor='#888888',
        facecolor='none', linestyle='--', zorder=2,
    )
    ax.add_patch(rect_ps)
    ax.text(0.23, y_ps_top + 0.05,
            f' per step  (t = 0 \u2026 L\u20131,  L = {seq_len})',
            ha='left', va='bottom', fontsize=7.5, color='#666666')

    # ------------------------------------------------------------------
    # (3) Feature Concatenation
    # ------------------------------------------------------------------
    y_concat = 12.0

    # GCN → concat
    diag_arrow(cx_gcn, y_gcn - gcn_h / 2 - 0.02, cx - 0.5, y_concat + 0.45)
    # ts input → concat (skips over GCN, comes from right column)
    diag_arrow(cx_ts, y_inp - 0.45, cx + 0.5, y_concat + 0.45)

    box(cx, y_concat, 7.8, 0.85, 'Feature Concatenation  (per step)', C_CONCAT)
    subtext(cx, y_concat - 0.25,
            f'[ z\u209c  \u2016  y\u209c  \u2016  exog\u209c ]'
            f'   \u2192   (B, {seq_len}, {d_g}+{ts_input_size}) = (B, {seq_len}, {flat_per_step})')

    # ------------------------------------------------------------------
    # (4) Flatten
    # ------------------------------------------------------------------
    y_flat = 10.75

    arrow(cx, y_concat, y_flat)
    box(cx, y_flat, 7.8, 0.85, 'Flatten', C_FLAT)
    subtext(cx, y_flat - 0.25,
            f'(B, {seq_len}, {flat_per_step})  \u2192  (B, {flat_total})')

    # ------------------------------------------------------------------
    # (5) MLP hidden layers
    # ------------------------------------------------------------------
    y_mlp = [9.5, 8.3]
    prev  = flat_total
    y_cur = y_flat

    for i, hs in enumerate(mlp_hidden):
        y_l = y_mlp[i]
        arrow(cx, y_cur, y_l)
        box(cx, y_l, 7.8, 0.85,
            f'Linear({prev} \u2192 {hs})  \u2192  ReLU  \u2192  Dropout',
            C_MLP)
        prev  = hs
        y_cur = y_l

    # ------------------------------------------------------------------
    # (6) Output linear
    # ------------------------------------------------------------------
    y_lin_out = 7.1

    arrow(cx, y_cur, y_lin_out)
    box(cx, y_lin_out, 5.5, 0.85,
        f'Linear({prev} \u2192 1)',
        C_MLP)

    # ------------------------------------------------------------------
    # (7) Forecast
    # ------------------------------------------------------------------
    y_out = 5.85

    arrow(cx, y_lin_out, y_out)
    box(cx, y_out, 3.5, 0.8, 'Forecast  t + 1', C_CONCAT, fontsize=10)
    subtext(cx, y_out - 0.25, '(inverse-scaled)')

    # ------------------------------------------------------------------
    # (8) MSE Loss
    # ------------------------------------------------------------------
    y_loss = 4.65

    arrow(cx, y_out, y_loss)
    box(cx, y_loss, 4.8, 0.75,
        'MSE Loss   \u2112 = \u2016\u0177 \u2212 y\u2016\u00b2',
        C_LOSS, fontsize=8.5)

    # ------------------------------------------------------------------
    # Backprop annotation (right margin, upward red arrow)
    # ------------------------------------------------------------------
    bp_x = 9.35
    ax.annotate(
        '',
        xy=(bp_x, y_gcn + 0.5),
        xytext=(bp_x, y_loss - 0.05),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.0),
        zorder=5,
    )
    ax.text(bp_x + 0.45, (y_gcn + y_loss) / 2,
            'Back-\nprop\n(\u2202\u2112/\u2202\u03b8)',
            ha='center', va='center', fontsize=8.5,
            color='#C0392B', fontweight='bold')

    # ------------------------------------------------------------------
    # Title & legend
    # ------------------------------------------------------------------
    ax.set_title('GCN + MLP  Architecture  (per-step ego-graphs)',
                 fontsize=13, fontweight='bold', pad=12)

    legend_items = [
        mpatches.Patch(color=C_INPUT,  label='Raw Inputs'),
        mpatches.Patch(color=C_GCN,    label='GCN Encoder'),
        mpatches.Patch(color=C_CONCAT, label='Concatenation / Output'),
        mpatches.Patch(color=C_FLAT,   label='Reshape / Flatten'),
        mpatches.Patch(color=C_MLP,    label='MLP Head'),
        mpatches.Patch(color=C_LOSS,   label='Loss'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8,
              frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    draw_gcn_mlp_architecture(save_path='gcn_mlp_architecture.png')
