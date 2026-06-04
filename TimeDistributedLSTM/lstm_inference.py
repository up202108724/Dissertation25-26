import torch
import numpy as np
import pandas as pd
import os
import time

    
# -----------------------------------------------------------------------------
# Recursive inference. Reads the LAST timestep's output for either head, so it
# is identical to the seq-to-one pipeline. Dynamic lag features are recomputed
# from the running prediction history, mirroring LSTM.lstm_inference.
# -----------------------------------------------------------------------------
def recursive_inference(model, seq_length, val_scaled, exog_val_scaled,
                        exog_test_scaled, exog_test, scaler, exog_scaler,
                        device, exog_cols, forecast_window):
    model.eval()
    td = model.time_distributed

    current_seq = val_scaled[-seq_length:].tolist()
    current_exog_seq = []
    if exog_cols:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]

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

    forecast = []
    start = time.time()
    with torch.no_grad():
        for step in range(forecast_window):
            if exog_cols:
                x_np = np.column_stack([np.array(current_seq).reshape(-1, 1),
                                        np.array(current_exog_seq)])
            else:
                x_np = np.array(current_seq).reshape(-1, 1)

            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            out = model(x)
            pred = (out[0, -1, 0] if td else out[0, 0]).cpu().item()
            forecast.append(pred)

            current_seq = current_seq[1:] + [pred]

            if exog_cols and step + 1 < forecast_window:
                next_exog_raw = exog_test[step + 1].copy()
                if dynamic_features:
                    max_w = max(w for _, _, w in dynamic_features)
                    hist = scaler.inverse_transform(
                        np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                    for idx, feat_type, w in dynamic_features:
                        if feat_type == "lag":
                            next_exog_raw[idx] = hist[-w] if w <= len(hist) else (hist[0] if len(hist) else 0.0)
                        else:  # rolling_mean / rolling_mean_excl
                            win = hist[-w:]
                            next_exog_raw[idx] = np.mean(win) if len(win) else 0.0
                next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    return forecast, time.time() - start
