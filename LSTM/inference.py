import torch
import numpy as np
import pandas as pd
import os
import time

    
def recursive_inference(model, test_start_idx, seq_length, 
                       val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                       scaler, exog_scaler, df_product, device, exog_cols,
                       forecast_window, seed, strategy, item_id, store_id, loss_type, script_dir):
    model.eval()
    
    current_seq = val_scaled[-seq_length:].tolist()
    current_exog_seq = []
    if exog_cols and len(exog_cols) > 0:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
    
    forecast = []
    
    rolling_features = []
    if exog_cols:
        for idx, col in enumerate(exog_cols):
            if col.startswith("rolling_mean_"):
                try:
                    rolling_features.append((idx, int(col.split("_")[-1])))
                except ValueError:
                    pass
                    
    # Setup inference log file
    inf_log_dir = os.path.join(script_dir, f'inference_logs/seed_{seed}/{loss_type}/{strategy}')
    os.makedirs(inf_log_dir, exist_ok=True)
    inf_log_path = os.path.join(inf_log_dir, f'inference_item{item_id}_store{store_id}.csv')
                    
    start_inference_time = time.time()
    
    with open(inf_log_path, 'w') as inf_log_file:
        header_str = "Step,X,Predicted_Y_Scaled,Predicted_Y_Unscaled"
        if exog_cols and len(exog_cols) > 0:
            exog_cols_unscaled_str = ",".join([f"{col}_Unscaled" for col in exog_cols])
            exog_cols_scaled_str = ",".join([f"{col}_Scaled" for col in exog_cols])
            header_str += f",{exog_cols_unscaled_str},{exog_cols_scaled_str}"
        inf_log_file.write(header_str + "\n")
        
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
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled},{last_exog_unscaled_str},{last_exog_scaled_str}\n')
                else:
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled}\n')

                current_seq = current_seq[1:] + [pred]

                if exog_cols and len(exog_cols) > 0 and step + 1 < forecast_window:
                    next_exog_raw = exog_test[step + 1].copy()

                    if len(rolling_features) > 0:
                        max_w = max([w for _, w in rolling_features])
                        hist_unscaled = scaler.inverse_transform(np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                        
                        for idx, w in rolling_features:
                            window_values = hist_unscaled[-w:]
                            next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0

                    next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                    current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]
                
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    inference_time = time.time() - start_inference_time
    
    return forecast, inference_time