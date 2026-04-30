import torch
import numpy as np
import pandas as pd
import os
import time

    
def simple_autoregressive_inference(model, seq_length, forecast_window, device, 
                                  val_scaled, test_scaled, exog_val_scaled, exog_test_scaled, 
                                  exog_test_raw, exog_cols, scaler, exog_scaler, date_col, df, test_start_idx):
    model.eval()
    forecast = []
    
    current_seq = val_scaled[-seq_length:].tolist()
    if exog_cols and len(exog_cols) > 0:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
    else:
        current_exog_seq = []
        
    start_time = time.time()
    with torch.no_grad():
        for step in range(forecast_window):
            features_to_stack = [np.array(current_seq).reshape(-1, 1)]
            if exog_cols and len(exog_cols) > 0:
                features_to_stack.append(np.array(current_exog_seq))
                
            x_np = np.column_stack(features_to_stack)
            x_tensor = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            
            pred = model(x_tensor).cpu().numpy()[0, 0]
            forecast.append(pred)
            current_seq.pop(0)
            current_seq.append(pred)
            
            if exog_cols and len(exog_cols) > 0:
                current_exog_seq.pop(0)
                if step + 1 < forecast_window:
                    current_exog_seq.append(exog_test_scaled[step + 1].tolist())
                else:
                    current_exog_seq.append(exog_test_scaled[-1].tolist())
                    
    inf_time = time.time() - start_time
    forecast_unscaled = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    return forecast_unscaled, inf_time