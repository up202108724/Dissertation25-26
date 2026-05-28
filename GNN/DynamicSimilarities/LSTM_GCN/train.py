"""
Training loop for the GCN + LSTM forecaster.

Mirrors the API of ``Graph2vec_FixedThreshold/LSTM/train.py`` so calling
code can swap one for the other.  Only the per-batch unpack changes:
batches now come from ``collate_pyg_ts`` and contain a PyG ``Batch`` of
ego-graphs, the temporal tensor and the labels.
"""

from __future__ import annotations

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
):
    """
    Jointly trains the GCN + LSTM end-to-end on the forecasting loss.

    Parameters
    ----------
    seed              : int            – RNG seed for reproducibility
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

        for pyg_batch, ts_batch, y_batch, target_idx in train_loader:
            pyg_batch  = pyg_batch.to(device)
            ts_batch   = ts_batch.to(device)
            y_batch    = y_batch.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()
            outputs = model(pyg_batch, target_idx, ts_batch)   # (B, H, 1)
            preds   = outputs[:, -1, 0].view(-1, 1)             # last-step pred
            loss    = criterion(preds, y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for pyg_batch, ts_batch, y_batch, target_idx in val_loader:
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
                f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}"
            )

    train_time = time.time() - start_train_time

    if best_model_path is None and hasattr(model, "best_state_dict"):
        model.load_state_dict(model.best_state_dict)

    return model, train_losses, val_losses, best_epoch, train_time
