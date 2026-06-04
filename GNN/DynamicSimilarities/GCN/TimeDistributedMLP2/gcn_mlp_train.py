"""
Training loop for the GCN + MLP forecaster.

Mirrors ``LSTM_GCN_1_graph_per_lookback/train.py`` (same batch unpacking via
collate_pyg_ts) but classifies grad norms under ``mlp`` instead of ``lstm``
and drops the LSTM-specific logging vocabulary.
"""

from __future__ import annotations

import csv
import os
import time
import numpy as np
import torch


def train_model(
    seed,
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    criterion2,
    optimizer,
    device,
    best_model_path=None,
    scheduler=None,
    patience=10,
    grad_clip=1.0,
    log_every=10,
    diag_csv_path=None,
    diag_meta=None,
):
    """
    Jointly trains the GCN + MLP end-to-end on the forecasting loss.

    Parameters mirror ``LSTM_GCN_1_graph_per_lookback/train.train_model``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── diagnostics: capture z embedding via forward-hook on z_norm ───────
    _z_buf: list[torch.Tensor] = []

    def _capture_z(_module, _inp, out):
        _z_buf.append(out.detach())

    h_handle = model.z_norm.register_forward_hook(_capture_z) if hasattr(model, 'z_norm') else None

    DIAG_FIELDS = [
        "product_id", "store_id", "seed", "metric",
        "window_size", "step_size", "threshold", "percentile",
        "enable_edges", "enable_second_degree", "ablate_z",
        "num_edges_mean",
        "epoch", "train_loss", "val_loss",
        "z_norm_mean", "z_var_mean",
        "grad_norm_gcn", "grad_norm_mlp", "grad_ratio_gcn_mlp",
    ]
    if diag_csv_path is not None:
        os.makedirs(os.path.dirname(diag_csv_path) or ".", exist_ok=True)
        if not os.path.exists(diag_csv_path):
            with open(diag_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(DIAG_FIELDS)

    start_train_time = time.time()
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # ── training ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        grad_acc_gcn = 0.0
        grad_acc_mlp = 0.0
        grad_n       = 0

        for pyg_batch, ts_batch, y_batch, target_idx, _L in train_loader:
            pyg_batch  = pyg_batch.to(device)
            ts_batch   = ts_batch.to(device)
            y_batch    = y_batch.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()
            outputs = model(pyg_batch, target_idx, ts_batch)        # (B, H, 1)
            preds   = outputs[:, -1, 0].view(-1, 1)
            loss    = criterion(preds, y_batch)
            loss.backward()

            with torch.no_grad():
                gn_gcn_sq = 0.0
                gn_mlp_sq = 0.0
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = float(p.grad.detach().pow(2).sum().item())
                    if n.startswith("conv") or n.startswith("z_norm"):
                        gn_gcn_sq += g2
                    elif n.startswith("mlp"):
                        gn_mlp_sq += g2
                grad_acc_gcn += gn_gcn_sq ** 0.5
                grad_acc_mlp += gn_mlp_sq ** 0.5
                grad_n       += 1

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        if _z_buf:
            z_cat = torch.cat(_z_buf, dim=0)
            z_norm_mean = float(z_cat.norm(dim=-1).mean().item())
            z_var_mean  = float(z_cat.var(dim=0).mean().item())
            _z_buf.clear()
        else:
            z_norm_mean = float("nan")
            z_var_mean  = float("nan")

        avg_gn_gcn = grad_acc_gcn / max(grad_n, 1)
        avg_gn_mlp = grad_acc_mlp / max(grad_n, 1)
        grad_ratio = avg_gn_gcn / (avg_gn_mlp + 1e-12)

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for pyg_batch, ts_batch, y_batch, target_idx, _L in val_loader:
                pyg_batch  = pyg_batch.to(device)
                ts_batch   = ts_batch.to(device)
                y_batch    = y_batch.to(device)
                target_idx = target_idx.to(device)

                outputs = model(pyg_batch, target_idx, ts_batch)
                preds   = outputs[:, -1, 0].view(-1, 1)
                all_outputs.append(preds)
                all_targets.append(y_batch)

        all_outputs = torch.cat(all_outputs, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        val_loss    = criterion2(all_outputs, all_targets).item()
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if diag_csv_path is not None:
            meta = diag_meta or {}
            with open(diag_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    meta.get("product_id", ""),
                    meta.get("store_id", ""),
                    seed,
                    meta.get("metric", ""),
                    meta.get("window_size", ""),
                    meta.get("step_size", ""),
                    meta.get("threshold", ""),
                    meta.get("percentile", ""),
                    meta.get("enable_edges", ""),
                    meta.get("enable_second_degree", ""),
                    bool(getattr(model, "ablate_z", False)),
                    meta.get("num_edges_mean", ""),
                    epoch + 1,
                    avg_train_loss,
                    val_loss,
                    z_norm_mean,
                    z_var_mean,
                    avg_gn_gcn,
                    avg_gn_mlp,
                    grad_ratio,
                ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            patience_counter = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
            else:
                model.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)."
                )
                break

        if (epoch + 1) % log_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                f"||z||={z_norm_mean:.3f} Var(z)={z_var_mean:.4f} | "
                f"|grad_gcn|={avg_gn_gcn:.2e} |grad_mlp|={avg_gn_mlp:.2e} "
                f"ratio={grad_ratio:.2e}"
            )

    train_time = time.time() - start_train_time
    if h_handle is not None:
        h_handle.remove()

    if best_model_path is None and hasattr(model, "best_state_dict"):
        model.load_state_dict(model.best_state_dict)

    return model, train_losses, val_losses, best_epoch, train_time
