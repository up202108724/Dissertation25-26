import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model_expanding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 3: Expanding Window (Growing Window) Training.
    The model trains progressively on an expanding dataset array, validating on the immediate next block.
    After each window, the model retains weights to fine-tune across expanding windows, simulating real longitudinal training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)
    current_train_end = initial_train_size
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

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
        
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation Step
            model.eval()
            all_outputs = []
            all_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    all_outputs.append(outputs)
                    all_targets.append(batch_y)
            
            all_outputs = torch.cat(all_outputs, dim=0).view(-1)
            all_targets = torch.cat(all_targets, dim=0).view(-1)
            val_loss = criterion2(all_outputs, all_targets).item()
            
            if scheduler is not None:
                scheduler.step(val_loss)
                
            # Keep globally lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_overall_epoch = epoch + 1
                torch.save(model.state_dict(), final_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} in window {window_idx}.")
                break
                
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(val_loss)
                
        current_train_end += val_step_size
        window_idx += 1
        
    train_time = time.time() - start_train_time
    return model, all_train_losses, all_val_losses, best_overall_epoch, train_time


def train_model_sliding_window(seed, epochs, model, full_train_scaled, exog_scaled, seq_length, initial_train_size, val_step_size, batch_size, criterion, criterion2, optimizer, device, final_model_path, dataset_class, scheduler=None, patience=10):
    """
    Strategy 4: Sliding Window Training.
    The model trains progressively on a sliding dataset window, validating on the immediate next block.
    The training window size remains fixed, but slides forward on each iteration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_train_time = time.time()
    total_len = len(full_train_scaled)
    
    current_train_start = 0
    current_train_end = initial_train_size
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_overall_epoch = 0

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
        
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation Step
            model.eval()
            all_outputs = []
            all_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    all_outputs.append(outputs)
                    all_targets.append(batch_y)
            
            all_outputs = torch.cat(all_outputs, dim=0).view(-1)
            all_targets = torch.cat(all_targets, dim=0).view(-1)
            val_loss = criterion2(all_outputs, all_targets).item()
            
            if scheduler is not None:
                scheduler.step(val_loss)
                
            # Keep globally lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_overall_epoch = epoch + 1
                torch.save(model.state_dict(), final_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} in window {window_idx}.")
                break
                
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(val_loss)
                
        current_train_start += val_step_size
        current_train_end += val_step_size
        window_idx += 1
        
    train_time = time.time() - start_train_time
    return model, all_train_losses, all_val_losses, best_overall_epoch, train_time

def train_model(epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=10):
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


def train_model_best_train_loss(seed, epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=10):
    """
    Strategy 1: Records the epoch of the best validation model, but saves the model 
    with the best training loss.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    start_train_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_epoch = 0
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
        
        # Save model based on best training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)

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

        # Track early stopping and epochs based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            patience_counter = 0  # Reset patience on improvement
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    train_time = time.time() - start_train_time

    # Return best_val_epoch as requested
    return model, train_losses, val_losses, best_val_epoch, train_time


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

'''
def train_model_selected_epochs(seed, selected_epochs, model, combined_loader, criterion, optimizer, device, final_model_path):
    """
    New Strategy: Trains the model using the combined training + validation data 
    for a explicitly selected number of epochs from scratch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    start_train_time = time.time()
    train_losses = []
    best_train_loss = float('inf')
    best_model_epoch = 0
    for epoch in range(selected_epochs):
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
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_model_epoch = epoch + 1
            torch.save(model.state_dict(), final_model_path)
        
        if (epoch + 1) % 10 == 0 or epoch == selected_epochs - 1:
            print(f"Epoch {epoch+1}/{selected_epochs} | Train+Val Train Loss: {avg_train_loss:.6f}")

    train_time = time.time() - start_train_time

    return model, train_losses, best_train_loss, best_model_epoch, train_time

'''
