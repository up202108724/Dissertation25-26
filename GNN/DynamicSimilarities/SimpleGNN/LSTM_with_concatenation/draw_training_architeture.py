"""
Training-pipeline architecture for SimpleGNNLSTMForecaster (concat fusion).

Depicts what `train.train_gnn_lstm` actually does, end-to-end:

    raw df  ─►  split train/val/test  ─►  MinMax fit on train target
            ─►  make_single_windows (build ego-graphs + ts_seq per sample)
            ─►  SingleGraphDataset + DataLoader (shuffle on train)

    per epoch:
        ── train loop ─────────────────────────────────────────────
        batch = (y, pyg_batch, ts_seq)
        pred  = model(pyg_batch, ptr[:-1], ts_seq)        # concat-fusion fwd
        loss  = MSE | MAE | Huber  (chosen by loss_type)
        loss.backward()  →  clip_grad_norm_(1.0)  →  AdamW step
                            (lr=1e-4, weight_decay=1e-3)

        ── val loop (no_grad) ─────────────────────────────────────
        val_loss = mean over val_loader
        if val_loss < best_val:
            best_state = state_dict.clone()
            torch.save → best_models/seed_<seed>/<loss>/gnn_lstm_product_<pid>.pth
            best_epoch = epoch ; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: early-stop

    return best model (loaded from best_state)

Outputs PNG + PDF next to this file.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Hyper-parameters / labels mirrored from train.TrainConfig + train_gnn_lstm
# ---------------------------------------------------------------------------
LOOKBACK         = 30
HORIZON_MODEL    = 1            # network predicts one step (recursive at inference)
WINDOW_GRAPH     = 15           # ego-graph time-series window
BATCH_SIZE       = 32
TRAIN_SIZE       = 455
VAL_SIZE         = 153
LR               = 1e-4
WEIGHT_DECAY     = 1e-3
EPOCHS           = 30
GRAD_CLIP        = 1.0
GCN_HIDDEN       = 32
GCN_OUT          = 16
LSTM_HIDDEN      = 64
LSTM_LAYERS      = 1
HEAD_HIDDEN      = 64
N_NODE_STATS     = 7
N_NODE_CAL       = 21
N_EXOG_LSTM      = 31
IN_CH            = N_NODE_STATS + N_NODE_CAL   # 28
LSTM_INPUT       = 1 + N_EXOG_LSTM             # 32
FUSED_DIM        = GCN_OUT + LSTM_HIDDEN       # 80


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COL_DATA    = "#E3F2FD"
COL_SPLIT   = "#B3E5FC"
COL_SCALE   = "#FFF3E0"
COL_WINDOW  = "#FFE082"
COL_LOADER  = "#FFD180"
COL_MODEL   = "#C8E6C9"
COL_LOSS    = "#FFCDD2"
COL_OPT     = "#D1C4E9"
COL_VAL     = "#B2DFDB"
COL_CKPT    = "#F8BBD0"
COL_LOOPBG  = "#F3E5F5"
EDGE        = "#37474F"
LOOP_EDGE   = "#6A1B9A"


def box(ax, xy, w, h, text, color, fontsize=9,
        boxstyle="round,pad=0.05,rounding_size=0.12"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                       linewidth=1.2, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color="#111")
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, text=None, rad=0.0, color=EDGE, fontsize=8,
          style="->", lw=1.4, ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom",
                fontsize=fontsize, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")

ax.set_title(
    "Training Pipeline — SimpleGNNLSTMForecaster (GCN + LSTM, concat fusion)",
    fontsize=14, fontweight="bold", pad=14, loc="center",
)
ax.text(10, 11.45,
        f"lookback={LOOKBACK}  model_horizon={HORIZON_MODEL}  "
        f"batch_size={BATCH_SIZE}  train={TRAIN_SIZE}  val={VAL_SIZE}  "
        f"epochs={EPOCHS}   AdamW(lr={LR}, wd={WEIGHT_DECAY})   clip={GRAD_CLIP}",
        ha="center", fontsize=9, color="#444")


# =====================================================================
# ROW 1 — Data preparation
# =====================================================================
y_data = 9.7
df_c    = box(ax, (0.4, y_data),  2.0, 0.95, "Raw df\n(target + exog cols)", COL_DATA)
split_c = box(ax, (3.0, y_data),  2.6, 0.95,
              f"Chronological split\ntrain[:{TRAIN_SIZE}]  |  val[{VAL_SIZE}]  |  test",
              COL_SPLIT)
scale_c = box(ax, (6.2, y_data),  2.6, 0.95,
              "MinMax scaler\n(fit on train target only)\nval/test: transform",
              COL_SCALE)
win_c   = box(ax, (9.4, y_data),  3.4, 0.95,
              "make_single_windows\nper sample build:\n"
              f"  • ego-graph (target + neighbours, w={WINDOW_GRAPH})\n"
              f"  • ts_seq (lookback={LOOKBACK}, dim={LSTM_INPUT})  • y (next step)",
              COL_WINDOW, fontsize=8)
load_c  = box(ax, (13.4, y_data), 3.2, 0.95,
              "SingleGraphDataset\n+ DataLoader\n"
              f"batch={BATCH_SIZE}   shuffle=True (train)\nshuffle=False (val)",
              COL_LOADER, fontsize=8)
seed_c  = box(ax, (17.2, y_data), 2.4, 0.95,
              "Seed fixed\n(NumPy / PyTorch / CUDA)\nreproducible split",
              COL_DATA, fontsize=8)

arrow(ax, (df_c[0]+1.0,    y_data+0.475), (split_c[0]-1.3, y_data+0.475))
arrow(ax, (split_c[0]+1.3, y_data+0.475), (scale_c[0]-1.3, y_data+0.475))
arrow(ax, (scale_c[0]+1.3, y_data+0.475), (win_c[0]-1.7,   y_data+0.475))
arrow(ax, (win_c[0]+1.7,   y_data+0.475), (load_c[0]-1.6,  y_data+0.475))


# =====================================================================
# ROW 2 — Per-epoch loop  (everything below this row lives inside the loop)
# =====================================================================
loop_frame = FancyBboxPatch(
    (0.4, 0.5), 19.2, 8.4,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    linewidth=1.4, edgecolor=LOOP_EDGE, facecolor=COL_LOOPBG,
    linestyle="--", alpha=0.55,
)
ax.add_patch(loop_frame)
ax.text(0.7, 8.7,
        f"For epoch in 1..{EPOCHS}     (early stop when val has not improved "
        f"for `patience` epochs)",
        fontsize=10, color=LOOP_EDGE, fontweight="bold")

# Arrow from DataLoader down into the loop
arrow(ax, (load_c[0], y_data), (load_c[0], 8.55), color=LOOP_EDGE, lw=1.5)


# ---------------------------------------------------------------------
# TRAIN sub-block (left half of the loop)
# ---------------------------------------------------------------------
ax.text(4.6, 8.25, "Train loop  (model.train())",
        fontsize=10, fontweight="bold", color="#1B5E20")

batch_c = box(ax, (1.0, 7.0), 3.6, 1.0,
              "Batch from train_loader\n(y, pyg_batch, ts_seq)\n→ .to(device)",
              COL_LOADER, fontsize=8)

# Model block (showing the two branches + concat + head)
model_box = FancyBboxPatch(
    (5.4, 4.6), 6.6, 3.55,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    linewidth=1.4, edgecolor=EDGE, facecolor="#F1F8E9",
)
ax.add_patch(model_box)
ax.text(8.7, 7.95, "SimpleGNNLSTMForecaster  (forward)",
        ha="center", fontsize=10, fontweight="bold", color="#1B5E20")

gcn_c = box(ax, (5.6, 5.95), 2.9, 1.0,
            f"GCN branch\nGCNConv {IN_CH}→{GCN_HIDDEN}\nReLU + Dropout\n"
            f"GCNConv {GCN_HIDDEN}→{GCN_OUT}\nz = h[ptr[:-1]]",
            COL_MODEL, fontsize=8)
lstm_c = box(ax, (8.9, 5.95), 2.9, 1.0,
             f"LSTM branch\ninput={LSTM_INPUT}, hidden={LSTM_HIDDEN}\n"
             f"layers={LSTM_LAYERS}, zero h₀/c₀\nh_T = out[:,-1,:]\n+ Dropout",
             COL_OPT, fontsize=8)

concat_c = box(ax, (6.4, 4.95), 5.2, 0.55,
               f"Concat fusion  [ LayerNorm(z) ‖ LayerNorm(h_T) ]  →  ℝ^{FUSED_DIM}",
               "#CE93D8", fontsize=8)
head_c   = box(ax, (6.4, 4.3), 5.2, 0.55,
               f"MLP head: Linear({FUSED_DIM}→{HEAD_HIDDEN}) + ReLU + Dropout → Linear({HEAD_HIDDEN}→{HORIZON_MODEL})  →  pred (B,1,1)",
               COL_LOSS, fontsize=8)

arrow(ax, (gcn_c[0],  5.95), (gcn_c[0],  5.50))
arrow(ax, (lstm_c[0], 5.95), (lstm_c[0], 5.50))
arrow(ax, (concat_c[0], 4.95), (head_c[0], 4.85))

# batch → model (split into the two branches)
arrow(ax, (batch_c[0]+1.8, 7.5), (gcn_c[0]-1.45, 6.45),
      text="pyg_batch + ptr[:-1]", rad=0.05, fontsize=7)
arrow(ax, (batch_c[0]+1.8, 7.3), (lstm_c[0]-1.45, 6.45),
      text="ts_seq", rad=-0.05, fontsize=7)

# pred → loss
loss_c = box(ax, (13.0, 6.7), 3.6, 1.0,
             "Loss\nMSE | MAE | Huber\n(loss_type)",
             COL_LOSS, fontsize=9)
arrow(ax, (head_c[0]+2.6, 4.55), (loss_c[0]-1.8, 7.0),
      text="pred", rad=-0.25, fontsize=7)
arrow(ax, (batch_c[0]+1.8, 7.7), (loss_c[0]-1.8, 7.3),
      text="y (target)", rad=0.25, fontsize=7)

# loss → backward → clip → AdamW
back_c = box(ax, (13.0, 5.3), 3.6, 1.0,
             "loss.backward()\nclip_grad_norm_(1.0)\nAdamW.step()\n"
             f"(lr={LR}, wd={WEIGHT_DECAY})",
             COL_OPT, fontsize=8)
arrow(ax, (loss_c[0], 6.7), (back_c[0], 6.3))

# backward → parameter update arrow back into the model
arrow(ax, (back_c[0]-1.8, 5.8), (head_c[0]+2.6, 4.6),
      text="∇θ  (update GCN + LSTM + head)",
      rad=0.25, color="#B71C1C", lw=1.3, fontsize=7)

# Accumulator note
box(ax, (17.0, 6.0), 2.4, 0.7,
    f"accumulate\ntrain_loss / N_train",
    COL_DATA, fontsize=8)


# ---------------------------------------------------------------------
# VAL sub-block (bottom of the loop)
# ---------------------------------------------------------------------
ax.plot([0.6, 19.4], [3.95, 3.95], color="#9E9E9E", lw=0.7, ls=":")
ax.text(2.5, 3.65, "Validation loop  (model.eval(), torch.no_grad())",
        fontsize=10, fontweight="bold", color="#004D40")

valbatch_c = box(ax, (1.0, 2.4), 3.6, 0.9,
                 "Batch from val_loader\nforward only\n(no grad, no shuffle)",
                 COL_VAL, fontsize=8)
valloss_c  = box(ax, (5.4, 2.4), 4.0, 0.9,
                 "val_loss  =  mean( loss_fn(pred, y) )\nover val_loader",
                 COL_LOSS, fontsize=9)
cmp_c      = box(ax, (10.2, 2.4), 4.2, 0.9,
                 "val_loss  <  best_val ?",
                 "#FFE0B2", fontsize=10)

arrow(ax, (valbatch_c[0]+1.8, 2.85), (valloss_c[0]-2.0, 2.85))
arrow(ax, (valloss_c[0]+2.0,  2.85), (cmp_c[0]-2.1, 2.85))

# YES branch — checkpoint
ckpt_c = box(ax, (15.4, 2.85), 4.2, 1.4,
             "YES  →  best_state = state_dict.clone()\n"
             "torch.save(best_state, .pth)\n"
             "best_models/seed_<seed>/<loss>/\n"
             "gnn_lstm_product_<pid>.pth\nbest_epoch = epoch ; no_improve = 0",
             COL_CKPT, fontsize=8)
arrow(ax, (cmp_c[0]+2.1, 3.05), (ckpt_c[0]-2.1, 3.30),
      text="yes", rad=0.15, color="#1B5E20", fontsize=8)

# NO branch — patience
pat_c = box(ax, (15.4, 1.05), 4.2, 1.3,
            "NO  →  no_improve += 1\n"
            "if no_improve ≥ patience:\n   early-stop (break)",
            "#FFCCBC", fontsize=8)
arrow(ax, (cmp_c[0]+2.1, 2.65), (pat_c[0]-2.1, 1.70),
      text="no", rad=-0.15, color="#B71C1C", fontsize=8)

# Loop-back arrow: end of epoch → next epoch
arrow(ax, (ckpt_c[0]+2.0, 3.55), (0.55, 8.55),
      text="next epoch", rad=-0.35, color=LOOP_EDGE, lw=1.4, ls="--", fontsize=8)


# =====================================================================
# Final outputs (outside the loop)
# =====================================================================
final_c = box(ax, (0.6, 0.05), 6.2, 0.55,
              "Return: model.load_state_dict(best_state) , scaler , "
              "train_losses , val_losses , best_epoch",
              "#DCEDC8", fontsize=9)

# =====================================================================
# Legend
# =====================================================================
legend_patches = [
    mpatches.Patch(facecolor=COL_DATA,   edgecolor=EDGE, label="Data / config"),
    mpatches.Patch(facecolor=COL_SPLIT,  edgecolor=EDGE, label="Split"),
    mpatches.Patch(facecolor=COL_SCALE,  edgecolor=EDGE, label="Scaler"),
    mpatches.Patch(facecolor=COL_WINDOW, edgecolor=EDGE, label="Windowing"),
    mpatches.Patch(facecolor=COL_LOADER, edgecolor=EDGE, label="DataLoader / batch"),
    mpatches.Patch(facecolor=COL_MODEL,  edgecolor=EDGE, label="GCN branch"),
    mpatches.Patch(facecolor=COL_OPT,    edgecolor=EDGE, label="LSTM / optimizer"),
    mpatches.Patch(facecolor="#CE93D8",  edgecolor=EDGE, label="Concat fusion"),
    mpatches.Patch(facecolor=COL_LOSS,   edgecolor=EDGE, label="Head / Loss"),
    mpatches.Patch(facecolor=COL_VAL,    edgecolor=EDGE, label="Validation"),
    mpatches.Patch(facecolor=COL_CKPT,   edgecolor=EDGE, label="Checkpoint"),
    mpatches.Patch(facecolor=COL_LOOPBG, edgecolor=LOOP_EDGE, label="Epoch loop"),
]
ax.legend(handles=legend_patches, loc="lower right",
          bbox_to_anchor=(0.998, 0.002), frameon=True, fontsize=8, ncol=4)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_out = os.path.join(out_dir, "gcn_lstm_concat_training_architecture.png")
pdf_out = os.path.join(out_dir, "gcn_lstm_concat_training_architecture.pdf")
plt.tight_layout()
plt.savefig(png_out, dpi=200, bbox_inches="tight")
plt.savefig(pdf_out,           bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_out}")
print(f"Saved: {pdf_out}")
