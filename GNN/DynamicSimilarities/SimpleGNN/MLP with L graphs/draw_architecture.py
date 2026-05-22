import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── Colors ──────────────────────────────────────────────────────────────────
C_IN   = '#AED6F1'   # inputs
C_GRAPH= '#A9DFBF'   # graph construction / sage
C_FUS  = '#FAD7A0'   # fusion
C_MLP  = '#F1948A'   # mlp
C_OUT  = '#D7BDE2'   # output
C_DATA = '#D5DBDB'   # data sources
ARROW  = '#1A252F'
EDGE   = '#2C3E50'

# ── Helper functions ─────────────────────────────────────────────────────────
def box(x, y, w, h, color, lines, fs=8.5, bold_first=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=EDGE, linewidth=1.6, zorder=3)
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        dy = (i - (n-1)/2) * (fs + 1.5) * 0.014
        weight = 'bold' if (bold_first and i == 0) else 'normal'
        ax.text(x, y - dy, line, ha='center', va='center', fontsize=fs,
                fontweight=weight, zorder=4, color='#1A252F')

def arrow(x1, y1, x2, y2, label='', color=ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=1.6, mutation_scale=14), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my+0.1, label, fontsize=7, color='#555', zorder=6, style='italic')

def section_label(x, y, text):
    ax.text(x, y, text, fontsize=9, color='#666', ha='center', style='italic', zorder=4)

# ═══════════════════════════════════════════════════════════════════════════
#  ROW POSITIONS
# ═══════════════════════════════════════════════════════════════════════════
# We lay the diagram left-to-right in "columns":
#   Col A  (x≈1.3) : External Data Inputs
#   Col B  (x≈4.1) : Graph Construction
#   Col C  (x≈7.3) : GCN Encoder
#   Col D  (x≈10.3): Central-node extraction + TS/Cal inputs
#   Col E  (x≈13.5): Spatial-Temporal Fusion
#   Col F  (x≈17)  : MLP Forecaster
#   Col G  (x≈20.5): Output

XA, XB, XC, XD, XE, XF, XG = 1.6, 4.4, 7.6, 11.0, 14.0, 17.5, 21.0

# ── Section banners ─────────────────────────────────────────────────────────
for bx, bw, label, col in [
    (XA,   2.6, "Data Inputs",          '#EBF5FB'),
    (XB,   2.6, "Graph Construction",   '#EAFAF1'),
    (XC,   2.8, "GCN Encoder",          '#EAFAF1'),
    (XD,   2.8, "Feature Assembly",     '#FEF9E7'),
    (XE,   2.4, "Fusion",               '#FEF9E7'),
    (XF,   2.8, "MLP Forecaster",       '#FDEDEC'),
    (XG,   1.6, "Output",               '#F5EEF8'),
]:
    banner = FancyBboxPatch((bx - bw/2, 0.3), bw, 12.2,
                            boxstyle="round,pad=0.1",
                            facecolor=col, edgecolor='#BFC9CA',
                            linewidth=1.0, zorder=1, alpha=0.5)
    ax.add_patch(banner)
    ax.text(bx, 12.7, label, ha='center', va='center', fontsize=8.5,
            color='#5D6D7E', fontweight='bold', zorder=4)

# ═══════════════════════════════════════════════════════════════════════════
#  A  –  DATA INPUTS
# ═══════════════════════════════════════════════════════════════════════════
box(XA, 10.5, 2.4, 0.9, C_DATA, ['df_wide', '(items × dates)'], fs=8, bold_first=True)
box(XA, 9.0,  2.4, 0.9, C_IN,   ['Historical Sales', '(lookback × 1)'], fs=8, bold_first=True)
box(XA, 7.4,  2.4, 0.9, C_IN,   ['Calendar Features', '(lookback × cal_dim)'], fs=8, bold_first=True)

# ═══════════════════════════════════════════════════════════════════════════
#  B  –  GRAPH CONSTRUCTION  (repeated for each of the L timesteps)
# ═══════════════════════════════════════════════════════════════════════════
box(XB, 10.5, 2.4, 1.0, C_GRAPH,
    ['Node Feature', 'Computation',
     '──────────────',
     'raw 15-day window',
     '+ 8 stats (mean, std,',
     'slope, zero-ratio…)',
     '→ dim = 23'], fs=7.5, bold_first=True)

box(XB, 8.2, 2.4, 1.3, C_GRAPH,
    ['Ego-Graph Building',
     '──────────────────',
     'Pairwise similarity/distance',
     '(e.g. CID, Pearson …)',
     'Threshold → edges',
     'Star + within-star links',
     'central_node_idx = 0'], fs=7.2, bold_first=True)

box(XB, 5.8, 2.4, 0.7, C_GRAPH,
    ['PyG Data',
     '(x, edge_index,',
     ' edge_attr)'], fs=7.5, bold_first=True)

# Arrows inside B column
arrow(XA+1.2, 10.5, XB-1.2, 10.5, label='window dates')
arrow(XB, 9.95, XB, 9.05)            # feature → ego
arrow(XB, 7.55, XB, 6.15)            # ego → pyg data
arrow(XA+1.2, 9.0, XB-1.2, 8.8)     # history → ego

# ═══════════════════════════════════════════════════════════════════════════
#  Loop annotation
# ═══════════════════════════════════════════════════════════════════════════
ax.text(XB, 4.75, '× L timesteps', ha='center', va='center', fontsize=7.5,
        color='#7D6608', style='italic', zorder=6)
loop_rect = FancyBboxPatch((XB-1.35, 4.95), 2.7, 6.35,
                           boxstyle="round,pad=0.05",
                           facecolor='none', edgecolor='#D4AC0D',
                           linewidth=1.4, linestyle='--', zorder=2)
ax.add_patch(loop_rect)

# ═══════════════════════════════════════════════════════════════════════════
#  Batch.from_data_list  (B×L graphs → PyG Batch)
# ═══════════════════════════════════════════════════════════════════════════
box(XB, 3.8, 2.4, 0.75, C_GRAPH,
    ['Batch.from_data_list',
     '(B×L graphs)'], fs=8, bold_first=True)

arrow(XB, 5.45, XB, 4.2)

# ═══════════════════════════════════════════════════════════════════════════
#  C  –  GCN ENCODER
# ═══════════════════════════════════════════════════════════════════════════
arrow(XB+1.2, 3.8, XC-1.4, 3.8)

box(XC, 9.5, 2.5, 0.85, C_GRAPH,
    ['GCNConv 1',
     'in_ch → hidden_ch',
     'add_self_loops=True'], fs=8, bold_first=True)

box(XC, 8.2, 2.5, 0.7, C_GRAPH,
    ['ReLU + Dropout(p=0.2)'], fs=8)

box(XC, 7.0, 2.5, 0.85, C_GRAPH,
    ['GCNConv 2',
     'hidden_ch → out_ch',
     'add_self_loops=True'], fs=8, bold_first=True)

box(XC, 5.5, 2.5, 0.85, C_GRAPH,
    ['Node Embeddings',
     '(total_nodes,',
     ' gcn_out_ch)'], fs=8, bold_first=True)

box(XC, 3.8, 2.5, 0.85, C_GRAPH,
    ['Central Node',
     'Extraction',
     '──────────────',
     'ptr[:-1]  →  node_0',
     'per graph',
     'reshape →',
     '(B, L, gcn_out_ch)'], fs=7.5, bold_first=True)

# Arrows inside C
arrow(XC, 9.05, XC, 8.55)
arrow(XC, 7.85, XC, 7.43)
arrow(XC, 6.57, XC, 6.07)
arrow(XC, 5.07, XC, 4.62)

# Connect Batch → GCNConv1
ax.annotate('', xy=(XC-1.25, 9.5), xytext=(XC-1.25, 3.8),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.4,
                            connectionstyle='arc3,rad=0.0'), zorder=5)
ax.text(XC-1.55, 6.65, 'PyG Batch', fontsize=7, color='#555',
        rotation=90, va='center', style='italic', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════
#  D  –  FEATURE ASSEMBLY  (TS + Cal + z_target arrive here)
# ═══════════════════════════════════════════════════════════════════════════
# z_target from C
arrow(XC+1.25, 3.8, XD-1.4, 3.8)
ax.text((XC+XD)/2, 4.05, 'z_target\n(B, L, gcn_out_ch)', ha='center',
        fontsize=7, color='#555', style='italic', zorder=6)

# Historical sales → Feature assembly
arrow(XA+1.2, 9.0, XD-1.4, 7.1)
ax.text(5.9, 8.5, 'ts_seq\n(B, L, 1)', ha='left', fontsize=7, color='#555',
        style='italic', zorder=6)

# Calendar features → Feature assembly
arrow(XA+1.2, 7.4, XD-1.4, 5.8)
ax.text(4.5, 6.6, 'cal_seq\n(B, L, cal_dim)', ha='left', fontsize=7,
        color='#555', style='italic', zorder=6)

box(XD, 7.1, 2.5, 0.75, C_IN,
    ['ts_seq',
     '(B, L, ts_dim)'], fs=8, bold_first=True)

box(XD, 5.8, 2.5, 0.75, C_IN,
    ['cal_seq',
     '(B, L, cal_dim)'], fs=8, bold_first=True)

box(XD, 3.8, 2.5, 0.75, C_GRAPH,
    ['z_target',
     '(B, L, gcn_out_ch)'], fs=8, bold_first=True)

# ═══════════════════════════════════════════════════════════════════════════
#  E  –  FUSION  (concatenation)
# ═══════════════════════════════════════════════════════════════════════════
box(XE, 5.5, 2.4, 1.1, C_FUS,
    ['Concatenate',
     '[ts | cal | z_target]',
     '────────────────────',
     '(B, L, ts_dim',
     '   + cal_dim',
     '   + gcn_out_ch)'], fs=8, bold_first=True)

# Arrows from D to E
for yd in [7.1, 5.8, 3.8]:
    arrow(XD+1.25, yd, XE-1.2, 5.5)

# ═══════════════════════════════════════════════════════════════════════════
#  F  –  MLP FORECASTER
# ═══════════════════════════════════════════════════════════════════════════
arrow(XE+1.2, 5.5, XF-1.4, 5.5)

box(XF, 7.8, 2.6, 0.75, C_MLP,
    ['Flatten',
     '(B, L × concat_dim)'], fs=8, bold_first=True)

box(XF, 6.4, 2.6, 1.0, C_MLP,
    ['FC + ReLU + Dropout',
     '(×  n_layers)',
     'hidden_sizes = (64,32)'], fs=8, bold_first=True)

box(XF, 5.0, 2.6, 0.9, C_MLP,
    ['FC Output Layer',
     '→ horizon × out_dim'], fs=8, bold_first=True)

box(XF, 3.6, 2.6, 0.9, C_MLP,
    ['Reshape',
     '(B, horizon, 1)'], fs=8, bold_first=True)

# Connect fusion to flatten
ax.annotate('', xy=(XF-1.3, 7.8), xytext=(XF-1.3, 5.5),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.4), zorder=5)
ax.text(XF-1.6, 6.65, 'combined_seq', fontsize=7, color='#555',
        rotation=90, va='center', style='italic', zorder=6)

arrow(XF, 7.42, XF, 6.9)
arrow(XF, 5.9, XF, 5.45)
arrow(XF, 4.55, XF, 4.05)

# ═══════════════════════════════════════════════════════════════════════════
#  G  –  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
arrow(XF+1.3, 3.6, XG-0.8, 3.6)

box(XG, 3.6, 1.5, 0.85, C_OUT,
    ['Predictions',
     '(B, horizon, 1)'], fs=8.5, bold_first=True)

# ── Recursive-inference annotation ──────────────────────────────────────────
rec_rect = FancyBboxPatch((XE-1.25, 0.45), (XG+0.75-XE+1.25), 1.35,
                          boxstyle="round,pad=0.08",
                          facecolor='#FDFEFE', edgecolor='#884EA0',
                          linewidth=1.3, linestyle=':', zorder=2)
ax.add_patch(rec_rect)
ax.text((XE + XG)/2, 1.12,
        'Recursive Inference: output at step t  →  input target at step t+1\n'
        '(graph rebuilt each autoregressive step with updated predictions)',
        ha='center', va='center', fontsize=7.2, color='#6C3483', zorder=6)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(11, 12.3,
        'GCN + MLP Forecaster (SimpleGNN_MLP_Forecaster) — Architecture Block Diagram',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='#1A252F', zorder=6)

plt.tight_layout(pad=0.4)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_gcn_mlp.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved to {out}")
plt.show()
