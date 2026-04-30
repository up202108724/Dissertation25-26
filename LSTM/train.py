import time
import torch
import numpy as np
import torch.nn as nn

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
