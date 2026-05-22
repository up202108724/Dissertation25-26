"""
Architecture diagram for GCNMLPForecaster
  – GCN encoder (ONE ego-graph per sample, 7-stat node features)
  – z  concatenated with flattened ts_seq  →  MLP
Run:  python draw_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette ──────────────────────────────────────────────────────────────────
C_IN    = '#AED6F1'   # blue   – sequence inputs
C_FEAT  = '#FDEBD0'   # peach  – node features
C_GRAPH = '#A9DFBF'   # green  – graph / GCN
C_FUS   = '#FAD7A0'   # orange – fusion / concat
C_MLP   = '#F1948A'   # red    – MLP layers
ARROW   = '#1A252F'
EDGE    = '#2C3E50'

fig, ax = plt.subplots(figsize=(22, 12))
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, color, lines, fs=8.5, bold_first=False):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=EDGE, linewidth=1.6, zorder=3))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        dy = (i - (n - 1) / 2) * (fs + 1.5) * 0.014
        ax.text(x, y - dy, line,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if (bold_first and i == 0) else 'normal',
                zorder=4, color='#1A252F')


def arr(x1, y1, x2, y2, lbl='', sty='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.6,
                                connectionstyle=sty, mutation_scale=14),
                zorder=5)
    if lbl:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.1, lbl,
                fontsize=7, color='#555', zorder=6, style='italic')


# ── Column positions ──────────────────────────────────────────────────────────
XA, XB, XC, XD, XE, XF, XG = 1.5, 4.2, 7.2, 10.3, 13.5, 17.2, 21.0

# ── Section banners ───────────────────────────────────────────────────────────
for bx, bw, lbl, col in [
    (XA, 2.4, 'Data Inputs',       '#EBF5FB'),
    (XB, 2.4, 'Node Features',     '#FEF9E7'),
    (XC, 2.7, 'Ego-Graph  (×1)',   '#EAFAF1'),
    (XD, 2.7, 'GCN Encoder',       '#EAFAF1'),
    (XE, 2.7, 'Feature Fusion',    '#FEF9E7'),
    (XF, 3.2, 'MLP Forecaster',    '#FDEDEC'),
    (XG, 1.4, 'Output',            '#EAFAF1'),
]:
    ax.add_patch(FancyBboxPatch(
        (bx - bw / 2, 0.3), bw, 11.0,
        boxstyle="round,pad=0.1",
        facecolor=col, edgecolor='#BFC9CA',
        linewidth=1.0, zorder=1, alpha=0.5))
    ax.text(bx, 11.5, lbl,
            ha='center', va='center', fontsize=8.5,
            color='#5D6D7E', fontweight='bold', zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
#  A – DATA INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
box(XA, 9.8, 2.2, 0.9, '#D5DBDB',
    ['df_wide', '(items × dates)'], fs=8, bold_first=True)
box(XA, 8.1, 2.2, 0.9, C_IN,
    ['Historical Sales (scaled)', 'lookback = 30 days'], fs=8, bold_first=True)
box(XA, 6.4, 2.2, 0.9, C_IN,
    ['Calendar Features (next-step)', 'cal_dim = 21  (EXOG_COLS)'], fs=8, bold_first=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  B – NODE FEATURES
#      NODE_FEATURES = 7 stats only (no ts window, no calendar)
# ═══════════════════════════════════════════════════════════════════════════════
box(XB, 8.5, 2.3, 2.0, C_FEAT,
    ['ALL NODES',
     '──────────────',
     'mean7',
     'mean_all',
     'std_all',
     'zero_ratio',
     'slope',
     'min_v,  max_v',
     '──────────────',
     'feature_dim = 7'],
    fs=7.8, bold_first=True)

ax.text(XB, 10.7,
        'No ts window or calendar\nin node features',
        ha='center', fontsize=7.5, color='#7D6608',
        style='italic', fontweight='bold', zorder=6)

arr(XA + 1.1, 9.8, XB - 1.15, 8.8, lbl='window data')
arr(XA + 1.1, 8.1, XB - 1.15, 8.2)

# ═══════════════════════════════════════════════════════════════════════════════
#  C – EGO-GRAPH  (ONE per sample)
# ═══════════════════════════════════════════════════════════════════════════════
box(XC, 8.0, 2.6, 3.0, C_GRAPH,
    ['Ego-Graph   (ONE per sample)',
     '─────────────────────────────',
     'Pairwise similarity (Spearman)',
     'Threshold → edges',
     'Star topology',
     '  + optional within-star links',
     'target_id  →  node index 0',
     'edge_attr = similarity weight',
     'PyG  Data(x, edge_index, edge_attr)'],
    fs=7.4, bold_first=True)

ax.text(XC, 10.7, '1 graph / sample',
        ha='center', fontsize=8.2, color='#1A5276',
        fontweight='bold', zorder=6)

arr(XB + 1.15, 8.5, XC - 1.3, 8.2)

# ═══════════════════════════════════════════════════════════════════════════════
#  D – GCN ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
box(XD, 9.8, 2.5, 0.9, C_GRAPH,
    ['Batch.from_data_list', 'B graphs  →  PyG Batch'],
    fs=8, bold_first=True)
box(XD, 8.5, 2.5, 0.9, C_GRAPH,
    ['GCNConv 1  (add_self_loops=True)',
     'in(7) → hidden(32)',
     'edge_weight = similarity'],
    fs=7.4, bold_first=True)
box(XD, 7.2, 2.5, 0.7, C_GRAPH,
    ['ReLU  +  Dropout(p=0.2)'], fs=8)
box(XD, 6.1, 2.5, 0.9, C_GRAPH,
    ['GCNConv 2  (add_self_loops=True)',
     'hidden(32) → out(16)'],
    fs=7.4, bold_first=True)
box(XD, 4.7, 2.5, 1.0, C_GRAPH,
    ['Extract Target Node',
     'idx  =  ptr[:-1]   (node 0)',
     'z  :  (B, 16)'],
    fs=7.6, bold_first=True)

ax.annotate('', xy=(XD - 1.25, 9.8), xytext=(XC + 1.3, 8.0),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.4,
                            connectionstyle='arc3,rad=-0.3'), zorder=5)
arr(XD, 9.35, XD, 8.95)
arr(XD, 8.05, XD, 7.55)
arr(XD, 6.85, XD, 6.55)
arr(XD, 5.65, XD, 5.20)

# ═══════════════════════════════════════════════════════════════════════════════
#  E – FEATURE FUSION
#      ts_seq (B, 30, 22) flattened → (B, 660)
#      [z (16) || flat_ts (660)] → (B, 676)
# ═══════════════════════════════════════════════════════════════════════════════
box(XE, 7.6, 2.5, 0.9, C_IN,
    ['ts_seq  (B, 30, 1+21=22)',
     '[ value_t  |  cal_{t+1} ]'],
    fs=7.8, bold_first=True)

box(XE, 6.2, 2.5, 0.85, C_IN,
    ['Flatten ts_seq',
     '(B, 30 × 22  =  660)'],
    fs=7.8, bold_first=True)

box(XE, 4.7, 2.5, 1.1, C_FUS,
    ['Concatenate',
     '[ z  ||  flat_ts ]',
     '────────────────────',
     '(B,  16 + 660  =  676)'],
    fs=7.8, bold_first=True)

arr(XA + 1.1, 8.1, XE - 1.25, 7.6, lbl='ts_seq', sty='arc3,rad=-0.05')
arr(XA + 1.1, 6.4, XE - 1.25, 7.2, sty='arc3,rad=-0.03')
arr(XE, 7.15, XE, 6.62)
arr(XD + 1.25, 4.7, XE - 1.25, 4.7, lbl='z')
arr(XE, 5.77, XE, 5.25)

# ═══════════════════════════════════════════════════════════════════════════════
#  F – MLP FORECASTER
# ═══════════════════════════════════════════════════════════════════════════════
box(XF, 8.6, 3.0, 0.9, C_MLP,
    ['FC  +  ReLU  +  Dropout',
     'Linear(676 → 64)'],
    fs=7.8, bold_first=True)
box(XF, 7.2, 3.0, 0.9, C_MLP,
    ['FC  +  ReLU  +  Dropout',
     'Linear(64  → 32)'],
    fs=7.8, bold_first=True)
box(XF, 5.8, 3.0, 0.9, C_MLP,
    ['Output Layer',
     'Linear(32  →  horizon=152)'],
    fs=7.8, bold_first=True)
box(XF, 4.5, 3.0, 0.85, '#A9DFBF',
    ['Reshape',
     '(B, 152, 1)'],
    fs=8, bold_first=True)

ax.annotate('', xy=(XF - 1.5, 8.6), xytext=(XE + 1.25, 4.7),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.4,
                            connectionstyle='arc3,rad=-0.3'), zorder=5)
ax.text(XF - 1.75, 6.65, 'combined\n(B, 676)',
        fontsize=7, color='#555', rotation=90, va='center',
        style='italic', zorder=6)

arr(XF, 8.15, XF, 7.65)
arr(XF, 6.75, XF, 6.25)
arr(XF, 5.35, XF, 4.92)

# ═══════════════════════════════════════════════════════════════════════════════
#  G – OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
box(XG, 4.5, 1.3, 0.85, '#A9DFBF',
    ['Predictions', '(B, 152, 1)'],
    fs=8.5, bold_first=True)
arr(XF + 1.5, 4.5, XG - 0.65, 4.5)

# ── Recursive-inference note ──────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (XE - 1.3, 0.3), (XG + 0.7 - XE + 1.3), 1.3,
    boxstyle="round,pad=0.08",
    facecolor='#FDFEFE', edgecolor='#884EA0',
    linewidth=1.3, linestyle=':', zorder=2))
ax.text((XE + XG) / 2, 0.95,
        'Recursive Inference:  forecast(t)  →  appended to ts_lookback  →  ego-graph rebuilt  →  step t+1\n'
        '(152 autoregressive steps to produce the full forecast horizon)',
        ha='center', va='center', fontsize=7.2, color='#6C3483', zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(11, 11.5,
        'GCN + MLP Forecaster  (GCNMLPForecaster)  —  Architecture Block Diagram',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#1A252F', zorder=6)

plt.tight_layout(pad=0.4)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_gcn_mlp.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved  →  {out}')
plt.close()
