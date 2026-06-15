import torch
from dataclasses import dataclass
from time import time
import time
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import csv
import sys
# ── Paths & sys.path setup ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))  # DynamicSimilarities/



from MLP_Baseline.mlp import TimeDistributedMLPForecaster
def train_gcn_lstm_model(
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
    Jointly trains the GCN + LSTM end-to-end on the forecasting loss.

    Parameters
    ----------
    epochs            : int            – maximum number of training epochs
    model             : SimpleGCNLSTMForecaster
    train_loader      : DataLoader yielding (pyg_batch, ts_batch, y_batch, target_idx)
    val_loader        : DataLoader (same format)
    criterion         : training loss  (e.g. MSE)
    criterion2        : validation loss (e.g. MAE)
    optimizer         : optimiser already bound to ``model.parameters()``
    device            : torch.device
    best_model_path   : optional path to dump best state_dict
    scheduler         : optional LR scheduler with ``.step(val_loss)``
    patience          : early-stopping patience (epochs without improvement)
    grad_clip         : max-norm for global grad clipping (None to disable)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ── diagnostics: capture z embedding via forward-hook on z_norm ───────
    _z_buf: list[torch.Tensor] = []

    def _capture_z(_module, _inp, out):
        _z_buf.append(out.detach())

    # Models without a graph branch (e.g. AblationLSTMForecaster) have no
    # z_norm; skip the hook and report z stats as NaN for those rows.
    z_norm_mod = getattr(model, "z_norm", None)
    h_handle = (
        z_norm_mod.register_forward_hook(_capture_z)
        if z_norm_mod is not None
        else None
    )

    if diag_csv_path is not None:
        os.makedirs(os.path.dirname(diag_csv_path) or ".", exist_ok=True)

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
        grad_acc_gcn  = 0.0
        grad_acc_lstm = 0.0
        grad_n        = 0

        for pyg_batch, ts_batch, y_batch, target_idx, _L in train_loader:
            pyg_batch  = pyg_batch.to(device)
            ts_batch   = ts_batch.to(device)
            y_batch    = y_batch.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()
            outputs = model(pyg_batch, target_idx, ts_batch)   # (B, H, 1)
            preds   = outputs[:, -1, 0].view(-1, 1)             # last-step pred
            loss    = criterion(preds, y_batch)
            loss.backward()

            # capture per-branch grad norms BEFORE clipping / step
            with torch.no_grad():
                gn_gcn_sq  = 0.0
                gn_lstm_sq = 0.0
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = float(p.grad.detach().pow(2).sum().item())
                    if n.startswith("conv") or n.startswith("z_norm"):
                        gn_gcn_sq += g2
                    elif n.startswith("lstm"):
                        gn_lstm_sq += g2
                grad_acc_gcn  += gn_gcn_sq  ** 0.5
                grad_acc_lstm += gn_lstm_sq ** 0.5
                grad_n        += 1

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # aggregate z stats captured during training forwards
        if _z_buf:
            z_cat = torch.cat(_z_buf, dim=0)                       # (sum B*L, d_g)
            z_norm_mean = float(z_cat.norm(dim=-1).mean().item())
            z_var_mean  = float(z_cat.var(dim=0).mean().item())
            _z_buf.clear()
        else:
            z_norm_mean = float("nan")
            z_var_mean  = float("nan")

        avg_gn_gcn  = grad_acc_gcn  / max(grad_n, 1)
        avg_gn_lstm = grad_acc_lstm / max(grad_n, 1)
        grad_ratio  = avg_gn_gcn / (avg_gn_lstm + 1e-12)

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

        # ── write per-epoch diagnostics row ────────────────────────────────
        if diag_csv_path is not None:
            meta = diag_meta or {}
            with open(diag_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    meta.get("product_id", ""),
                    meta.get("store_id", ""),
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
                    avg_gn_lstm,
                    grad_ratio,
                ])

        # ── early stopping bookkeeping ─────────────────────────────────────
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
                f"|grad_gcn|={avg_gn_gcn:.2e} |grad_lstm|={avg_gn_lstm:.2e} "
                f"ratio={grad_ratio:.2e}"
            )

    train_time = time.time() - start_train_time

    if h_handle is not None:
        h_handle.remove()

    if best_model_path is None and hasattr(model, "best_state_dict"):
        model.load_state_dict(model.best_state_dict)

    return model, train_losses, val_losses, best_epoch, train_time


def train_gcn_mlpmodel(
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


# ── Expanding-window (walk-forward) cross-validation ────────────────────────
def _fit_no_val(
    seed, epochs, model, train_loader, criterion, optimizer, device,
    scheduler=None, grad_clip=1.0, log_every=10,
):
    """Train for a FIXED number of epochs with no validation / early stopping.

    Used for the "Final" expanding-window fit, which trains on *all* train+val
    days and therefore has no held-out block to early-stop on.  The epoch budget
    is fixed up front (typically the median best-epoch found across the CV
    folds).  Returns ``(model, train_losses, train_time)``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start = time.time()
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for pyg_batch, ts_batch, y_batch, target_idx, _L in train_loader:
            pyg_batch  = pyg_batch.to(device)
            ts_batch   = ts_batch.to(device)
            y_batch    = y_batch.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()
            outputs = model(pyg_batch, target_idx, ts_batch)
            preds   = outputs[:, -1, 0].view(-1, 1)
            loss    = criterion(preds, y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        train_losses.append(avg)
        # No val loss to plateau on; step the scheduler on the train loss so a
        # ReduceLROnPlateau still decays the LR over the fixed budget.
        if scheduler is not None:
            scheduler.step(avg)
        if (epoch + 1) % log_every == 0:
            print(f"[final] Epoch {epoch + 1}/{epochs} | Train: {avg:.6f}")

    return model, train_losses, time.time() - start


def train_gcn_expanding_window(
    seed,
    epochs,
    build_model,
    fold_loaders,
    head,
    device,
    criterion=None,
    criterion2=None,
    lr=1e-3,
    weight_decay=1e-2,
    grad_clip=1.0,
    patience=10,
    sched_factor=0.5,
    sched_patience=None,
    log_every=10,
    diag_csv_path=None,
    diag_meta=None,
    final_train_loader=None,
    final_model_path=None,
    final_epochs=None,
):
    """
    Expanding-window (walk-forward) CV around the existing per-fold trainers.

    The graph/dataset pipeline stays in the caller: it hands over one
    ``(train_loader, val_loader)`` per fold (training origin fixed at day 0,
    validation block sliding forward) plus, optionally, a ``final_train_loader``
    spanning *all* train+val days for the deployment refit.

    Parameters
    ----------
    build_model        : ``() -> nn.Module`` — a FRESH model per fold/final fit
                         (set ``.ablate_z`` inside the factory if needed).  Each
                         fold restarts from scratch; weights are never carried
                         across folds.
    fold_loaders       : ``list[(train_loader, val_loader)]`` in chronological
                         fold order (Fold 1 first).
    head               : ``'lstm'`` or ``'mlp'`` — selects the per-fold trainer
                         (``train_gcn_lstm_model`` vs ``train_gcn_mlpmodel``).
    final_train_loader : optional loader over all train+val days; when given, a
                         deployment model is retrained on it for ``final_epochs``
                         (default: median best-epoch across folds — there is no
                         held-out block to early-stop on) and saved to
                         ``final_model_path``.

    Returns
    -------
    dict with per-fold ``best_val_loss``/``best_epoch``/``val_losses``, the mean
    CV val score (``cv_val_mean``), the chosen ``final_epochs`` and, if a final
    loader was supplied, the trained ``final_model``.
    """
    if head not in ("lstm", "mlp"):
        raise ValueError(f"head must be 'lstm' or 'mlp', got {head!r}")
    _fold_trainer = train_gcn_lstm_model if head == "lstm" else train_gcn_mlpmodel
    criterion  = criterion  or nn.MSELoss()
    criterion2 = criterion2 or nn.MSELoss()
    sched_patience = sched_patience if sched_patience is not None else max(1, patience // 3)

    fold_best_val, fold_best_epoch, fold_val_curves = [], [], []

    for k, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
        print(f"\n{'-'*60}\nExpanding-window fold {k}/{len(fold_loaders)} ({head})\n{'-'*60}")
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor, patience=sched_patience,
        )
        fold_diag = (
            diag_csv_path.replace(".csv", f"_fold{k}.csv")
            if diag_csv_path is not None else None
        )
        _, _t_losses, _v_losses, best_epoch, _ = _fold_trainer(
            seed=seed, epochs=epochs, model=model,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, criterion2=criterion2, optimizer=optimizer,
            device=device, best_model_path=None, scheduler=scheduler,
            patience=patience, grad_clip=grad_clip, log_every=log_every,
            diag_csv_path=fold_diag, diag_meta=diag_meta,
        )
        best_val = min(_v_losses) if _v_losses else float("nan")
        fold_best_val.append(best_val)
        fold_best_epoch.append(best_epoch)
        fold_val_curves.append(_v_losses)
        print(f"Fold {k}: best_val={best_val:.6f} @ epoch {best_epoch}")

    cv_val_mean = float(np.nanmean(fold_best_val)) if fold_best_val else float("nan")
    print(f"\nCV mean val ({head}, {len(fold_loaders)} folds): {cv_val_mean:.6f}")

    result = {
        "head": head,
        "fold_best_val": fold_best_val,
        "fold_best_epoch": fold_best_epoch,
        "fold_val_curves": fold_val_curves,
        "cv_val_mean": cv_val_mean,
        "final_epochs": None,
        "final_model": None,
    }

    # ── Final deployment fit on all train+val days (no early stopping) ───────
    if final_train_loader is not None:
        if final_epochs is None:
            final_epochs = int(round(float(np.median(fold_best_epoch)))) if fold_best_epoch else epochs
            final_epochs = max(1, final_epochs)
        print(f"\nFinal fit on all train+val for {final_epochs} epochs "
              f"(median best-epoch across folds).")
        final_model = build_model().to(device)
        final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
        final_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            final_opt, mode="min", factor=sched_factor, patience=sched_patience,
        )
        final_model, _, _ = _fit_no_val(
            seed=seed, epochs=final_epochs, model=final_model,
            train_loader=final_train_loader, criterion=criterion,
            optimizer=final_opt, device=device, scheduler=final_sched,
            grad_clip=grad_clip, log_every=log_every,
        )
        if final_model_path:
            os.makedirs(os.path.dirname(final_model_path) or ".", exist_ok=True)
            torch.save(final_model.state_dict(), final_model_path)
        result["final_epochs"] = final_epochs
        result["final_model"]  = final_model

    return result