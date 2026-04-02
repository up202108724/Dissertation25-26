import torch
import numpy as np
import pandas as pd
import os
import time

def test_model(
    model, df, date_col, scaler, exog_scaler, test_start_idx, seq_length, forecast_window, 
    device, item_id, store_id, seed, criterion, val_scaled,
    exog_val_scaled=None, exog_test_scaled=None, exog_test_raw=None, exog_cols=None, save_plot_path=None
):
    
    model.eval()
    start_inference_time = time.time()
    
    # Get the start date of the test set
    test_start_date = df[date_col].iloc[test_start_idx]
    print(f"Test set starts on: {test_start_date.date()}")

    # Use the last seq_length points from validation data as initial input
    current_seq = val_scaled[-seq_length:].tolist()
    if exog_cols and len(exog_cols) > 0:
        # Shift exog by 1 to match training dataloader
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
        
    current_date_seq = df[date_col].iloc[test_start_idx - seq_length : test_start_idx].dt.date.tolist()
    
    forecast = []
    
    # Setup inference log file
    inf_log_dir = f'inference_logs/seed_{seed}/{criterion}/item_{item_id}_store_{store_id}'
    os.makedirs(inf_log_dir, exist_ok=True)
    inf_log_path = f'{inf_log_dir}/inference_item{item_id}_store{store_id}.csv'
    
    with open(inf_log_path, 'w') as inf_log_file:
        
        # Dynamically identify all rolling mean features
        rolling_features = [] # List of tuples: (index, window_size)
        if exog_cols:
            for idx, col in enumerate(exog_cols):
                if col.startswith("rolling_mean_"):
                    try:
                        window = int(col.split("_")[-1])
                        rolling_features.append((idx, window))
                    except ValueError:
                        pass
        
        # Write header that includes exogenous columns if available
        header_str = "Step,X,Predicted_Y_Scaled,Predicted_Y_Unscaled"
        if exog_cols and len(exog_cols) > 0:
            exog_cols_unscaled_str = ",".join([f"{col}_Unscaled" for col in exog_cols])
            exog_cols_scaled_str = ",".join([f"{col}_Scaled" for col in exog_cols])
            header_str += f",{exog_cols_unscaled_str},{exog_cols_scaled_str}"
        inf_log_file.write(header_str + "\n")
        
        with torch.no_grad():
            for step in range(forecast_window):
                # Build model input from current history
                if exog_cols and len(exog_cols) > 0:
                    current_seq_arr = np.array(current_seq).reshape(-1, 1)
                    current_exog_arr = np.array(current_exog_seq)
                    x_np = np.column_stack([current_seq_arr, current_exog_arr])
                else:
                    x_np = np.array(current_seq).reshape(-1, 1)

                x = torch.FloatTensor(x_np).unsqueeze(0).to(device)

                # Predict next value
                pred = model(x).cpu().numpy()[0, 0]
                forecast.append(pred)

                # Log the unscaled array snapshot for validation right after predicting
                x_str = str(x_np.tolist()).replace('"', "'")
                
                # Unscale the prediction so it's readable in the log
                pred_unscaled = scaler.inverse_transform([[pred]])[0, 0]
                
                # Attempt to get the unscaled current exogenous features used for this prediction
                if exog_cols and len(exog_cols) > 0:
                    # Inverse transform just the last exogenous vector so we can read the raw values in the log
                    last_exog_scaled = current_exog_arr[-1]
                    last_exog_raw = exog_scaler.inverse_transform(last_exog_scaled.reshape(1, -1))[0]
                    
                    last_exog_unscaled_str = ",".join([str(v) for v in last_exog_raw.tolist()])
                    last_exog_scaled_str = ",".join([str(v) for v in last_exog_scaled.tolist()])
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled},{last_exog_unscaled_str},{last_exog_scaled_str}\n')
                else:
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled}\n')


                y_date = df[date_col].iloc[test_start_idx + step].date()
                if step % 10 == 0:
                    print(f"Step {step}: Predicting for Date: {y_date}")

                # Update target sequence with prediction
                current_seq = current_seq[1:] + [pred]
                current_date_seq = current_date_seq[1:] + [y_date]

                # Update exogenous sequence for the row being appended
                if exog_cols and len(exog_cols) > 0 and step + 1 < forecast_window:
                    # Use raw exog for the next predicted date
                    next_exog_raw = exog_test_raw[step + 1].copy()

                    # Recompute rolling means dynamically from past values only
                    if len(rolling_features) > 0:
                        max_w = max([w for _, w in rolling_features])
                        # Convert current sequence back to original scale (only what we strictly need)
                        hist_unscaled = scaler.inverse_transform(np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                        
                        for idx, w in rolling_features:
                            window_values = hist_unscaled[-w:]
                            next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0

                    # Scale after overwriting the rolling features
                    next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]

                    # Slide exog window
                    current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]
                    
    # Inverse transform predictions
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    print(f"Forecasted values {forecast[:5]} ...")
    inference_time = time.time() - start_inference_time

    return forecast, inference_time