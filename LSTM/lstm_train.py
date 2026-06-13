import copy
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def _train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Run a single training epoch and return the average batch loss."""
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def _validate(model, val_loader, criterion2, device):
    """Return the validation loss over the whole loader."""
    model.eval()
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            all_outputs.append(model(batch_x))
            all_targets.append(batch_y)
    all_outputs = torch.cat(all_outputs, dim=0).view(-1)
    all_targets = torch.cat(all_targets, dim=0).view(-1)
    return criterion2(all_outputs, all_targets).item()


def _mean_curves(fold_train_curves, fold_val_curves):
    """Average the per-fold train/val curves epoch-by-epoch (truncated to the shortest fold)."""
    n = min(len(c) for c in fold_val_curves)
    mean_train = [float(np.mean([c[e] for c in fold_train_curves])) for e in range(n)]
    mean_val = [float(np.mean([c[e] for c in fold_val_curves])) for e in range(n)]
    return mean_train, mean_val


def _select_epochs_from_mean_curve(mean_val_curve, patience):
    """Early-stopping selection on the mean fold curve. Returns (best_epoch, best_mean_val)."""
    best_mean = float('inf')
    best_epoch = 0
    counter = 0
    for e, mv in enumerate(mean_val_curve):
        if mv < best_mean:
            best_mean = mv
            best_epoch = e + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return best_epoch, best_mean


def _refit(model, refit_data, refit_exog, seq_length, batch_size, epochs_to_train,
           criterion, optimizer, device, dataset_class, initial_model_state,
           initial_optim_state, final_model_path):
    """
    Refit a fresh model (pristine weights/optimizer) on the given data for a fixed number
    of epochs, then save it. No scheduler/validation is used here (constant LR), matching
    the project's 'combined' refit convention.
    """
    refit_dataset = dataset_class(refit_data, refit_exog, seq_length)
    refit_loader = DataLoader(refit_dataset, batch_size=batch_size, shuffle=False)
    model.load_state_dict(initial_model_state)
    optimizer.load_state_dict(initial_optim_state)
    for epoch in range(epochs_to_train):
        avg_train_loss = _train_one_epoch(model, refit_loader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == epochs_to_train - 1:
            print(f"  [refit] Epoch {epoch+1}/{epochs_to_train} | Train Loss: {avg_train_loss:.6f}")
    torch.save(model.state_dict(), final_model_path)
    return model


def train_model_expanding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 3: Expanding Window (Growing Window) Cross-Validation.

    The train window grows by val_step_size each fold; the immediately following block is the
    validation set. Each fold is INDEPENDENT (model weights, optimizer and scheduler are reset
    to their pristine initial state before every fold), so this is true walk-forward CV rather
    than continual learning. Every fold runs the full `epochs` budget and its per-epoch
    validation curve is recorded.

    Model selection: the per-fold val curves are averaged epoch-by-epoch and the best epoch
    count E* is chosen by early-stopping (`patience`) on that MEAN curve. A single fresh model
    is then refit on ALL the train+val data for E* epochs -> this is the deployed model.

    Returns the mean train/val curves (for plotting), E*, and the wall-clock time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)

    # Pristine states so every fold starts identically and independently.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optim_state = copy.deepcopy(optimizer.state_dict())
    initial_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None

    fold_train_curves = []
    fold_val_curves = []

    current_train_end = initial_train_size
    window_idx = 0
    # Walk through the data by expanding the train window limit repeatedly
    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Expanding Window {window_idx}: Train 0->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")

        train_data = full_train_scaled[:current_train_end]

        val_start = current_train_end - seq_length
        val_end = current_train_end + val_step_size
        val_data = full_train_scaled[val_start:val_end]

        exog_train = exog_scaled[:current_train_end] if exog_scaled is not None else None
        exog_val = exog_scaled[val_start:val_end] if exog_scaled is not None else None

        train_dataset = dataset_class(train_data, exog_train, seq_length)
        val_dataset = dataset_class(val_data, exog_val, seq_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_end += val_step_size
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Independent fold: reset to pristine weights/optimizer/scheduler
        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optim_state)
        if scheduler is not None:
            scheduler.load_state_dict(initial_sched_state)

        # Run the full epoch budget so every fold's curve is aligned for averaging.
        train_curve, val_curve = [], []
        for epoch in range(epochs):
            avg_train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _validate(model, val_loader, criterion2, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            train_curve.append(avg_train_loss)
            val_curve.append(val_loss)

        fold_train_curves.append(train_curve)
        fold_val_curves.append(val_curve)

        current_train_end += val_step_size
        window_idx += 1

    if len(fold_val_curves) == 0:
        train_time = time.time() - start_train_time
        return model, [], [], 0, train_time

    # Average across folds and pick E* on the mean validation curve.
    mean_train_curve, mean_val_curve = _mean_curves(fold_train_curves, fold_val_curves)
    best_overall_epoch, best_mean = _select_epochs_from_mean_curve(mean_val_curve, patience)
    print(f"\\n[Expanding CV] {len(fold_val_curves)} folds | E* = {best_overall_epoch} epochs | mean val loss = {best_mean:.6f}")

    # Refit one fresh model on ALL train+val data for E* epochs -> deployed model.
    model = _refit(model, full_train_scaled, exog_scaled, seq_length, batch_size,
                   best_overall_epoch, criterion, optimizer, device, dataset_class,
                   initial_model_state, initial_optim_state, final_model_path)

    train_time = time.time() - start_train_time
    return model, mean_train_curve, mean_val_curve, best_overall_epoch, train_time


def train_model_sliding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 4: Sliding Window Cross-Validation.

    The train window has a FIXED size (initial_train_size) and slides forward by val_step_size
    each fold; the immediately following block is the validation set. Each fold is INDEPENDENT
    (weights, optimizer and scheduler reset to pristine state per fold) and runs the full
    `epochs` budget, recording its per-epoch validation curve.

    Model selection: per-fold val curves are averaged epoch-by-epoch and the best epoch count
    E* is chosen by early-stopping (`patience`) on that MEAN curve. A single fresh model is
    then refit for E* epochs on the MOST RECENT window only (the last initial_train_size days),
    keeping the fixed-window philosophy -> this is the deployed model.

    Returns the mean train/val curves (for plotting), E*, and the wall-clock time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)

    # Pristine states so every fold starts identically and independently.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optim_state = copy.deepcopy(optimizer.state_dict())
    initial_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None

    fold_train_curves = []
    fold_val_curves = []

    current_train_start = 0
    current_train_end = initial_train_size
    window_idx = 0

    while current_train_end + val_step_size <= total_len:
        print(f"\\n--- Sliding Window {window_idx}: Train {current_train_start}->{current_train_end}, Val {current_train_end}->{current_train_end + val_step_size} ---")

        train_data = full_train_scaled[current_train_start:current_train_end]

        val_start = current_train_end - seq_length
        val_end = current_train_end + val_step_size
        val_data = full_train_scaled[val_start:val_end]

        exog_train = exog_scaled[current_train_start:current_train_end] if exog_scaled is not None else None
        exog_val = exog_scaled[val_start:val_end] if exog_scaled is not None else None

        train_dataset = dataset_class(train_data, exog_train, seq_length)
        val_dataset = dataset_class(val_data, exog_val, seq_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Skipping window due to insufficient data for seq_length.")
            current_train_start += val_step_size
            current_train_end += val_step_size
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Independent fold: reset to pristine weights/optimizer/scheduler
        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optim_state)
        if scheduler is not None:
            scheduler.load_state_dict(initial_sched_state)

        # Run the full epoch budget so every fold's curve is aligned for averaging.
        train_curve, val_curve = [], []
        for epoch in range(epochs):
            avg_train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _validate(model, val_loader, criterion2, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            train_curve.append(avg_train_loss)
            val_curve.append(val_loss)

        fold_train_curves.append(train_curve)
        fold_val_curves.append(val_curve)

        current_train_start += val_step_size
        current_train_end += val_step_size
        window_idx += 1

    if len(fold_val_curves) == 0:
        train_time = time.time() - start_train_time
        return model, [], [], 0, train_time

    # Average across folds and pick E* on the mean validation curve.
    mean_train_curve, mean_val_curve = _mean_curves(fold_train_curves, fold_val_curves)
    best_overall_epoch, best_mean = _select_epochs_from_mean_curve(mean_val_curve, patience)
    print(f"\\n[Sliding CV] {len(fold_val_curves)} folds | E* = {best_overall_epoch} epochs | mean val loss = {best_mean:.6f}")

    # Refit one fresh model for E* epochs on the MOST RECENT window only -> deployed model.
    refit_start = max(0, total_len - initial_train_size)
    refit_data = full_train_scaled[refit_start:total_len]
    refit_exog = exog_scaled[refit_start:total_len] if exog_scaled is not None else None
    model = _refit(model, refit_data, refit_exog, seq_length, batch_size,
                   best_overall_epoch, criterion, optimizer, device, dataset_class,
                   initial_model_state, initial_optim_state, final_model_path)

    train_time = time.time() - start_train_time
    return model, mean_train_curve, mean_val_curve, best_overall_epoch, train_time

def train_model(seed, epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=10):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    
    start_train_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                all_outputs.append(outputs)
                all_targets.append(batch_y)

        all_outputs = torch.cat(all_outputs, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        
        val_loss = criterion2(all_outputs, all_targets).item()
        val_losses.append(val_loss)
        
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter on improvement
            torch.save(model.state_dict(), best_model_path)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
        patience_counter +=1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    train_time = time.time() - start_train_time

    return model, train_losses, val_losses, best_epoch, train_time



def train_model_combined(seed, epochs_to_train, model, combined_loader, criterion, optimizer, device, final_model_path):
    """
    Strategy 2: Train an existing model further (such as when combining Train + Val).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    start_train_time = time.time()
    train_losses = []

    for epoch in range(epochs_to_train):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(combined_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(combined_loader)
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == epochs_to_train - 1:
            print(f"Epoch {epoch+1}/{epochs_to_train} | Combined Train Loss: {avg_train_loss:.6f}")

    # Save the retrained model
    torch.save(model.state_dict(), final_model_path)
        
    train_time = time.time() - start_train_time

    return model, train_losses, train_time

