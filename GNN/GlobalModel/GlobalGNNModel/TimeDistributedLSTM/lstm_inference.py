import torch
import numpy as np
import pandas as pd
import os
import time

    
# =============================================================================
# Recursive multi-step forecasting for the whole cluster
# =============================================================================
def multivariate_recursive_inference(model, seq_length, val_scaled,
                                     exog_val_scaled, exog_test_scaled,
                                     scaler, device, forecast_window):
    """Roll the model forward ``forecast_window`` steps, feeding back its own
    K-dimensional predictions. Returns an inverse-scaled (H, K) forecast.

    Exogenous features (if any) are KNOWN future calendar variables, so they are
    taken from ``exog_test_scaled`` rather than predicted.
    """
    model.eval()
    td = model.time_distributed
    has_exog = exog_test_scaled is not None

    current_seq = list(val_scaled[-seq_length:])          # list of (K,) rows
    if has_exog:
        current_exog = list(exog_val_scaled[-seq_length + 1:]) + [exog_test_scaled[0]]

    forecast = []
    start = time.time()
    with torch.no_grad():
        for step in range(forecast_window):
            seq_arr = np.array(current_seq)               # (seq, K)
            if has_exog:
                x_np = np.column_stack([seq_arr, np.array(current_exog)])
            else:
                x_np = seq_arr

            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            out = model(x)
            pred = (out[0, -1, :] if td else out[0, :]).cpu().numpy()   # (K,)
            forecast.append(pred)

            current_seq = current_seq[1:] + [pred]

            if has_exog and step + 1 < forecast_window:
                current_exog = current_exog[1:] + [exog_test_scaled[step + 1]]

    forecast = scaler.inverse_transform(np.array(forecast))             # (H, K)
    return forecast, time.time() - start
