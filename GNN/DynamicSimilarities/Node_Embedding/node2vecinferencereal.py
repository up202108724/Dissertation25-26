import torch
import numpy as np
import pandas as pd
import os
import time
import networkx as nx
from node2vec import Node2Vec
from GNN.DynamicSimilarities.GraphAnalysis.graph_construction_similarity import compute_similarities_1vsAll

def build_dynamic_graph(target_id, target_preds, df_wide, cat_labels, date_cols, metric, threshold):
    G = nx.Graph()
    cat = cat_labels.get(target_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
    G.add_node(target_id, cat_label=cat)
    
    # Extract data for the window
    window_data = df_wide[date_cols]
    all_ts = window_data.values
    item_ids = window_data.index.values
    
    target_ts = np.array(target_preds)
    
    similarities = compute_similarities_1vsAll(target_ts, all_ts, metric=metric)
    
    active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
    valid_mask = (item_ids != target_id) & active_items_mask
    
    if np.sum(np.abs(target_ts)) == 0:
        valid_mask = np.zeros_like(valid_mask, dtype=bool)
        
    valid_item_ids = item_ids[valid_mask]
    valid_similarities = similarities[valid_mask]
    valid_original_idxs = np.arange(len(item_ids))[valid_mask]
    
    mask = valid_similarities >= threshold
    selected_similarities = valid_similarities[mask]
    selected_ids = valid_item_ids[mask]
    selected_orig_idxs = valid_original_idxs[mask]
    
    neighbor_indices = []
    for orig_idx, sim_val, other_id in zip(selected_orig_idxs, selected_similarities, selected_ids):
        neighbor_indices.append((orig_idx, other_id))
        if not G.has_node(other_id):
            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
            G.add_node(other_id, cat_label=cat_other)
        G.add_edge(target_id, other_id, weight=float(sim_val))
            
    if len(neighbor_indices) > 1:
        n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
        n_ids = [oid for _, oid in neighbor_indices]
        
        for i, n_id1 in enumerate(n_ids):
            n1_ts = n_ts_array[i]
            n_sims = compute_similarities_1vsAll(n1_ts, n_ts_array, metric=metric)
            for j in range(i + 1, len(n_ids)):
                if n_sims[j] >= threshold:
                    if np.sum(np.abs(n1_ts)) > 0 and np.sum(np.abs(n_ts_array[j])) > 0:
                        G.add_edge(n_id1, n_ids[j], weight=float(n_sims[j]))
    return G

def get_dynamic_embedding(G, target_id, dimensions=64):
    if G.number_of_edges() == 0:
        return np.zeros(dimensions)
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=10, num_walks=50, workers=4, quiet=True)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    if str(target_id) in model.wv:
        return model.wv[str(target_id)]
    return np.zeros(dimensions)
    
def node2vec_inference(
    metric, window_size, step_size, threshold, model, df, df_wide, cat_labels, date_col, scaler, exog_scaler, test_start_idx, seq_length, forecast_window, 
    device, item_id, store_id, seed, criterion, val_scaled,
    exog_val_scaled=None, exog_test_scaled=None, exog_test_raw=None, exog_cols=None, save_plot_path=None,
    node_embeddings=None, freeze_graph=True
):
    
    model.eval()
    start_inference_time = time.time()
    
    # Get the start date of the test set
    test_start_date = df[date_col].iloc[test_start_idx]
    print(f"Test set starts on: {test_start_date.date()}")

    # Determine date columns matching the wide dataframe 
    all_dates = df[date_col].dt.strftime('%Y-%m-%d').tolist()
    # Ensure they match the format of df_wide columns (assuming strings or timestamp properties)
    df_wide_cols = df_wide.columns.astype(str) if df_wide is not None else []

    # Use the last seq_length points from validation data as initial input
    current_seq = val_scaled[-seq_length:].tolist()
    if exog_cols and len(exog_cols) > 0:
        # Shift exog by 1 to match training dataloader
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
        
    current_date_seq = df[date_col].iloc[test_start_idx - seq_length : test_start_idx].dt.date.tolist()

    if node_embeddings is not None:
        # Extract the sequence of embeddings corresponding to the last seq_length days of validation
        current_emb_seq = node_embeddings[test_start_idx - seq_length : test_start_idx].tolist()
    else:
        # User disabled embeddings
        current_emb_seq = None
    
    forecast = []
    warmup_preds_unscaled = []
    total_steps = forecast_window
    
    # Setup inference log file
    inf_log_dir = f'inference_logs/seed_{seed}/{criterion}/item_{item_id}_store_{store_id}'
    os.makedirs(inf_log_dir, exist_ok=True)
    inf_log_path = f'{inf_log_dir}/inference_item{item_id}_store{store_id}.csv'
    
    with open(inf_log_path, 'w') as inf_log_file:
        
        rolling_features = []
        if exog_cols:
            for idx, col in enumerate(exog_cols):
                if col.startswith("rolling_mean_"):
                    try:
                        window = int(col.split("_")[-1])
                        rolling_features.append((idx, window))
                    except ValueError:
                        pass
        
        header_str = "Step,X,Predicted_Y_Scaled,Predicted_Y_Unscaled"
        if exog_cols and len(exog_cols) > 0:
            exog_cols_unscaled_str = ",".join([f"{col}_Unscaled" for col in exog_cols])
            exog_cols_scaled_str = ",".join([f"{col}_Scaled" for col in exog_cols])
            header_str += f",{exog_cols_unscaled_str},{exog_cols_scaled_str}"
        inf_log_file.write(header_str + "\n")
        
        with torch.no_grad():
            for step in range(total_steps):
                
                if step >= window_size and not freeze_graph and current_emb_seq is not None:
                    # Time to update embedding dynamic graph
                    # The dates for the window
                    window_date_start_idx = test_start_idx + step - window_size
                    window_date_end_idx = test_start_idx + step
                    
                    window_dates = all_dates[window_date_start_idx:window_date_end_idx]
                    
                    # We need the wide columns for these dates. If format varies, convert appropriately
                    matched_cols = [c for c in df_wide_cols if any(wd in c for wd in window_dates)]
                    if len(matched_cols) == window_size:
                        # Use the real values of the target product from df_wide instead of predictions
                        target_real_values = df_wide.loc[item_id, matched_cols].values
                        
                        G = build_dynamic_graph(item_id, target_real_values, df_wide, cat_labels, matched_cols, metric, threshold)
                        new_emb = get_dynamic_embedding(G, item_id, dimensions=len(current_emb_seq[0]))
                        
                        # Replace the last element (which would normally be just the frozen one) with our new dynamic one
                        current_emb_seq[-1] = new_emb.tolist()
                
                # Build model input from current history
                current_seq_arr = np.array(current_seq).reshape(-1, 1)
                
                features_to_stack = [current_seq_arr]
                
                if exog_cols and len(exog_cols) > 0:
                    current_exog_arr = np.array(current_exog_seq)
                    features_to_stack.append(current_exog_arr)
                    
                if node_embeddings is not None or current_emb_seq is not None:
                    current_emb_arr = np.array(current_emb_seq)
                    features_to_stack.append(current_emb_arr)

                x_np = np.column_stack(features_to_stack)

                x = torch.FloatTensor(x_np).unsqueeze(0).to(device)

                # Predict next value
                pred = model(x).cpu().numpy()[0, 0]
                
                pred_unscaled = scaler.inverse_transform([[pred]])[0, 0]
                warmup_preds_unscaled.append(pred_unscaled)
                
                # Pad the first window_size days with NaN so the plot starts on the window_size-th day
                if step < window_size:
                    forecast.append(np.nan)
                else:
                    forecast.append(pred)

                # Logging
                x_str = str(x_np.tolist()).replace('"', "'")
                if exog_cols and len(exog_cols) > 0:
                    last_exog_scaled = current_exog_arr[-1]
                    last_exog_raw = exog_scaler.inverse_transform(last_exog_scaled.reshape(1, -1))[0]
                    last_exog_unscaled_str = ",".join([str(v) for v in last_exog_raw.tolist()])
                    last_exog_scaled_str = ",".join([str(v) for v in last_exog_scaled.tolist()])
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled},{last_exog_unscaled_str},{last_exog_scaled_str}\n')
                else:
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled}\n')


                if (test_start_idx + step) < len(df):
                    y_date = df[date_col].iloc[test_start_idx + step].date()
                else:
                    y_date = current_date_seq[-1] # fallback if beyond df length
                    
                if step % 10 == 0:
                    print(f"Step {step}: Predicting for Date: {y_date}")

                # Update target sequence with prediction
                current_seq = current_seq[1:] + [pred]
                current_date_seq = current_date_seq[1:] + [y_date]

                # Update node sequences
                if current_emb_seq is not None and step + 1 < total_steps:
                    next_emb = current_emb_seq[-1] # default keep same (frozen mode or pending update)
                    current_emb_seq = current_emb_seq[1:] + [next_emb]

                # Update exogenous sequence
                if exog_cols and len(exog_cols) > 0 and step + 1 < total_steps:
                    if (step + 1) < len(exog_test_raw):
                        next_exog_raw = exog_test_raw[step + 1].copy()
                    else:
                        next_exog_raw = exog_test_raw[-1].copy() # fallback if beyond length

                    if len(rolling_features) > 0:
                        max_w = max([w for _, w in rolling_features])
                        hist_unscaled = scaler.inverse_transform(np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                        
                        for idx, w in rolling_features:
                            window_values = hist_unscaled[-w:]
                            next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0

                    next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                    current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]
                    
    # Inverse transform predictions
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    print(f"Forecasted values {forecast[:5]} ...")
    inference_time = time.time() - start_inference_time

    return forecast, inference_time