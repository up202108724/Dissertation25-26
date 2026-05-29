"""
Training loop for the N-HiTS forecaster.

Designed for direct multi-horizon AND recursive (horizon=1) training: the
model produces a (B, H, 1) tensor, and the loss is computed on the same
shape regardless of H.  When the model has no future-exog channel (n_exog=0
or future_exog_len=0), batches may omit the future_exog tensor.

The loader is expected to yield either:
    (x, fut_exog, y)   when exogenous future features are used
    (x, y)             otherwise
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _unpack_batch(batch):
    """Allow batches to be either (x, fut, y) or (x, y)."""
    if len(batch) == 3:
        x, fut, y = batch
    elif len(batch) == 2:
        x, y = batch
        fut = None
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}: expected 2 or 3 tensors.")
    return x, fut, y


def train_nhits(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    grad_clip: float = 1.0,
    best_model_path: str | None = None,
    loss_type: str = "mse",
    log_every: int = 25,
) -> Tuple[list, list, int, float]:
    """
    Train an N-HiTS model with early stopping + LR plateau scheduling.

    Parameters
    ----------
    model            : NHITS instance (forward: (x, future_exog?) -> (B, H, 1))
    train_loader     : yields (x, fut_exog, y) or (x, y) tensors
    val_loader       : same format as train_loader
    device           : torch device
    epochs           : maximum number of epochs
    lr               : AdamW learning rate
    weight_decay     : AdamW weight decay
    patience         : early stopping patience (epochs without val improvement)
    grad_clip        : max-norm of grad clipping (None to disable)
    best_model_path  : path to dump best state_dict (None -> kept in-memory only)
    loss_type        : 'mse' | 'mae' | 'huber'
    log_every        : print stats every N epochs

    Returns
    -------
    train_losses, val_losses, best_epoch, best_val_loss
    """
    loss_type = loss_type.lower()
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "huber":
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss_type={loss_type!r}; use mse/mae/huber.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3),
    )

    if best_model_path is not None:
        os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = 0
    best_state_in_memory = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── training ───────────────────────────────────────────────────────
        model.train()
        run_loss, n_seen = 0.0, 0
        for batch in train_loader:
            x, fut, y = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            fut = fut.to(device) if fut is not None else None

            optimizer.zero_grad()
            preds = model(x, fut) if fut is not None else model(x)
            loss = criterion(preds, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            n_seen   += x.size(0)
        train_loss = run_loss / max(n_seen, 1)
        train_losses.append(train_loss)

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        run_loss, n_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, fut, y = _unpack_batch(batch)
                x = x.to(device)
                y = y.to(device)
                fut = fut.to(device) if fut is not None else None
                preds = model(x, fut) if fut is not None else model(x)
                run_loss += criterion(preds, y).item() * x.size(0)
                n_seen   += x.size(0)
        val_loss = run_loss / max(n_seen, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
            else:
                best_state_in_memory = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val={best_val:.6f})")
                break

        if epoch % log_every == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.6f} val={val_loss:.6f}")

    # Restore best weights if we did not persist them
    if best_model_path is None and best_state_in_memory is not None:
        model.load_state_dict(best_state_in_memory)

    return train_losses, val_losses, best_epoch, best_val
