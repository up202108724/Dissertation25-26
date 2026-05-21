import time
import torch
import numpy as np
import torch.nn as nn


def train_model_with_embedding(
    seed,
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    criterion2,
    optimizer,
    device,
    best_model_path,
    scheduler=None,
    patience=10,
):
    """
    Training loop for models whose forward signature is
    ``model(x_ts, emb_idx) -> (B, 1)``, i.e. LSTMWithEmbedding and
    MLPWithEmbedding.

    The data loaders must yield 3-tuples ``(x_ts, emb_idx, y)`` as
    produced by ``TimeSeriesEmbIdxDataset``.

    Parameters
    ----------
    seed            : random seed set at the start of training
    epochs          : maximum number of training epochs
    model           : LSTMWithEmbedding or MLPWithEmbedding
    train_loader    : DataLoader yielding (x_ts, emb_idx, y)
    val_loader      : DataLoader yielding (x_ts, emb_idx, y)
    criterion       : training loss (e.g. nn.MSELoss())
    criterion2      : validation loss (e.g. nn.MSELoss())
    optimizer       : optimiser (e.g. torch.optim.Adam)
    device          : torch device
    best_model_path : path to save the best checkpoint; pass None to keep
                      the best state dict in memory
    scheduler       : optional LR scheduler (ReduceLROnPlateau)
    patience        : early-stopping patience

    Returns
    -------
    model, train_losses, val_losses, best_epoch, train_time
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    train_losses, val_losses = [], []
    best_val_loss    = float('inf')
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(epochs):
        # ------------------------------------------------------------------ #
        #  Training                                                            #
        # ------------------------------------------------------------------ #
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_idx, batch_y in train_loader:
            batch_x   = batch_x.to(device)
            batch_idx = batch_idx.to(device)
            batch_y   = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, batch_idx)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ------------------------------------------------------------------ #
        #  Validation                                                          #
        # ------------------------------------------------------------------ #
        model.eval()
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_idx, batch_y in val_loader:
                batch_x   = batch_x.to(device)
                batch_idx = batch_idx.to(device)
                batch_y   = batch_y.to(device)

                outputs = model(batch_x, batch_idx)
                all_outputs.append(outputs)
                all_targets.append(batch_y)

        all_outputs = torch.cat(all_outputs, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        val_loss    = criterion2(all_outputs, all_targets).item()
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        # ------------------------------------------------------------------ #
        #  Early stopping & checkpointing                                      #
        # ------------------------------------------------------------------ #
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
            else:
                model.best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs)."
            )
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {avg_train:.6f} | Val: {val_loss:.6f}"
            )

    train_time = time.time() - start_train_time

    if not best_model_path and hasattr(model, 'best_state_dict'):
        model.load_state_dict(model.best_state_dict)

    return model, train_losses, val_losses, best_epoch, train_time
