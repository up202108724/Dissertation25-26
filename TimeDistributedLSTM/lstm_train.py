import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Training: dense supervision for the time-distributed head, last-step only for
# seq-to-one. Validation is ALWAYS monitored on the last-step MSE so the two
# heads are directly comparable and early stopping is consistent.
# -----------------------------------------------------------------------------
def train(seed, model, train_loader, val_loader, criterion, optimizer, device,
          best_model_path, scheduler=None, patience=10, epochs=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    val_criterion = nn.MSELoss()
    td = model.time_distributed

    start = time.time()
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:                 # by: (B, seq, 1)
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            if td:
                loss = criterion(out, by)           # all steps
            else:
                loss = criterion(out, by[:, -1, :])  # last step only
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation: last-step prediction, MSE, for both heads.
        model.eval()
        outs, tgts = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                last = out[:, -1, :] if td else out
                outs.append(last)
                tgts.append(by[:, -1, :])
        outs = torch.cat(outs, dim=0).view(-1)
        tgts = torch.cat(tgts, dim=0).view(-1)
        val_loss = val_criterion(outs, tgts).item()
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train {avg_train:.6f} | Val {val_loss:.6f}")

    return model, train_losses, val_losses, best_epoch, time.time() - start