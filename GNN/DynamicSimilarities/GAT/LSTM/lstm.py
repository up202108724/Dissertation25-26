import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # only meaningful if >1
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = self.drop(lstm_out[:, -1, :])
        return self.linear(last)
    
class TimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length):
        """
        Dataset for time series with optional exogenous variables.
        
        Args:
            target_data: Scaled target variable data
            exog_data: Scaled exogenous variables data (can be None)
            seq_length: Length of input sequences
        """
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.has_exog = exog_data is not None
    
    def __len__(self):
        return len(self.target_data) - self.seq_length
    
    def __getitem__(self, idx):
        # Target sequence and label
        target_seq = self.target_data[idx:idx+self.seq_length]
        y = self.target_data[idx+self.seq_length]
        
        if self.has_exog:
            # Combine target with exogenous variables
            exog_seq = self.exog_data[idx+1:idx+self.seq_length+1]  # Align exog with target sequence
            # Stack target and exog features: shape (seq_length, 1 + n_exog)
            x = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x = target_seq.reshape(-1, 1)
        
        return torch.FloatTensor(x), torch.FloatTensor([y])



def train_lstm(seed, epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=150):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    

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


    return model, train_losses, val_losses, best_epoch


def recursive_inference_lstm(model, test_start_idx, seq_length, 
                       val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                       scaler, exog_scaler, df_product, device, exog_cols,
                       forecast_window, seed, strategy, item_id, store_id, loss_type, script_dir):
    model.eval()
    
    current_seq = val_scaled[-seq_length:].tolist()
    current_exog_seq = []
    if exog_cols and len(exog_cols) > 0:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
    
    forecast = []
    
    dynamic_features = []
    if exog_cols:
        for idx, col in enumerate(exog_cols):
            try:
                if col.startswith("rolling_mean_excl_"):
                    dynamic_features.append((idx, "rolling_mean_excl", int(col.split("_")[-1])))
                elif col.startswith("rolling_mean_"):
                    dynamic_features.append((idx, "rolling_mean", int(col.split("_")[-1])))
                elif col.startswith("lag_"):
                    dynamic_features.append((idx, "lag", int(col.split("_")[-1])))
            except ValueError:
                pass
                    
    
        
    with torch.no_grad():
        for step in range(forecast_window):
            if exog_cols and len(exog_cols) > 0:
                current_seq_arr = np.array(current_seq).reshape(-1, 1)
                current_exog_arr = np.array(current_exog_seq)
                x_np = np.column_stack([current_seq_arr, current_exog_arr])
            else:
                x_np = np.array(current_seq).reshape(-1, 1)

            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            forecast.append(pred)

            pred_unscaled = scaler.inverse_transform([[pred]])[0, 0]
            x_str = str(x_np.tolist()).replace('"', "'")
            
            if exog_cols and len(exog_cols) > 0:
                last_exog_scaled = current_exog_arr[-1]
                last_exog_raw = exog_scaler.inverse_transform(last_exog_scaled.reshape(1, -1))[0]
                last_exog_unscaled_str = ",".join([str(v) for v in last_exog_raw.tolist()])
                last_exog_scaled_str = ",".join([str(v) for v in last_exog_scaled.tolist()])
                
           

            current_seq = current_seq[1:] + [pred]

            if exog_cols and len(exog_cols) > 0 and step + 1 < forecast_window:
                next_exog_raw = exog_test[step + 1].copy()

                if len(dynamic_features) > 0:
                    max_w = max([w for _, _, w in dynamic_features])
                    # current_seq[-1] is the prediction for 'step'. We are preparing exog for 'step + 1'
                    hist_unscaled = scaler.inverse_transform(np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                    
                    for idx, feat_type, w in dynamic_features:
                        if feat_type == "lag":
                            if w <= len(hist_unscaled):
                                next_exog_raw[idx] = hist_unscaled[-w]
                            else:
                                next_exog_raw[idx] = hist_unscaled[0] if len(hist_unscaled) > 0 else 0.0
                        elif feat_type == "rolling_mean_excl":
                            # For step+1, shift(1) means the window ends at step.
                            window_values = hist_unscaled[-w:]
                            next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0
                        elif feat_type == "rolling_mean":
                            # For step+1, rolling included the future target at step+1, which we don't have.
                            # Approximating by averaging the available window up to step.
                            window_values = hist_unscaled[-w:]
                            next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0

                next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]
                
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    
    return forecast