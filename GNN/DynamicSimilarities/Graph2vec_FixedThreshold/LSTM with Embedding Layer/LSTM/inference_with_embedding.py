import time
import numpy as np
import torch


def inference_with_embedding(
    model,
    device,
    seq_length,
    forecast_window,
    test_start_idx,
    val_scaled,
    exog_val_scaled,
    exog_test_scaled,
    test_scaled,
    scaler,
    exog_cols,
    emb_table_size,
    warmup_steps=0,
):
    """
    Recursive (autoregressive) inference for LSTMWithEmbedding /
    MLPWithEmbedding.

    Instead of building dynamic graphs at test time, the model looks up
    the fine-tuned embedding directly from the embedding table using the
    ABSOLUTE TIMESTEP INDEX of each window position.  For positions within
    the training / validation range the fine-tuned weights are used; for
    future positions beyond the training range the table retains its
    Graph2Vec initialisation (no data leakage — graph structure is an
    input feature, not a label).

    At each autoregressive step ``i``:
      * window covers absolute indices  [test_start_idx - seq_length + i,
                                         ...,  test_start_idx + i - 1]
      * indices are clamped to [0, emb_table_size - 1] for safety
      * the predicted scaled value is fed back as input for step i+1

    Parameters
    ----------
    model           : LSTMWithEmbedding or MLPWithEmbedding (already on device)
    device          : torch.device
    seq_length      : lookback window  (L)
    forecast_window : number of future steps to predict
    test_start_idx  : absolute index of the first test day in the full df
    val_scaled      : full validation target array  (used to seed the window)
    exog_val_scaled : full validation exog array    (can be None)
    exog_test_scaled: full test exog array           (can be None)
    test_scaled     : full test target array (used during warmup); None if
                      ground-truth should not be used
    scaler          : fitted MinMaxScaler for the target (inverse_transform)
    exog_cols       : list of exog column names (or [] / None)
    emb_table_size  : total number of entries in the embedding table
    warmup_steps    : number of initial steps where ground-truth replaces
                      the model prediction (mirrors the original warmup logic)

    Returns
    -------
    forecast        : np.ndarray  shape (forecast_window,)   unscaled values
    inference_time  : float  seconds
    """
    model.eval()
    start = time.time()

    use_exog = exog_cols is not None and len(exog_cols) > 0

    # Seed the rolling window with the last seq_length steps of validation
    current_seq  = val_scaled[-seq_length:].tolist()
    if use_exog:
        # Shift exog forward by 1 (same convention as training dataset)
        current_exog = (
            exog_val_scaled[-seq_length + 1:].tolist()
            + [exog_test_scaled[0].tolist()]
        )

    forecast = []

    with torch.no_grad():
        for step in range(forecast_window):
            # -- Embedding indices for the current window --
            window_start = test_start_idx - seq_length + step
            emb_idx = np.arange(
                window_start, window_start + seq_length, dtype=np.int64
            )
            emb_idx = np.clip(emb_idx, 0, emb_table_size - 1)

            # -- Build x_ts --
            x_ts = np.array(current_seq, dtype=np.float32).reshape(-1, 1)
            if use_exog:
                x_ts = np.column_stack([x_ts, np.array(current_exog, dtype=np.float32)])

            x_tensor   = torch.FloatTensor(x_ts).unsqueeze(0).to(device)    # (1, L, F)
            idx_tensor = torch.LongTensor(emb_idx).unsqueeze(0).to(device)  # (1, L)

            pred_scaled = model(x_tensor, idx_tensor).cpu().numpy()[0, 0]

            # -- Warmup: replace model output with ground-truth --
            if (
                step < warmup_steps
                and test_scaled is not None
                and step < len(test_scaled)
            ):
                val = test_scaled[step]
                pred_scaled = float(val.item() if hasattr(val, 'item') else val)

            # -- Unscale and record --
            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            forecast.append(pred_unscaled)

            # -- Roll the window forward --
            current_seq = current_seq[1:] + [pred_scaled]
            if use_exog and (step + 1) < forecast_window:
                current_exog = current_exog[1:] + [exog_test_scaled[step + 1].tolist()]

    inference_time = time.time() - start
    return np.array(forecast, dtype=np.float64), inference_time
