import torch
import numpy as np
import pandas as pd
import os
import time
import networkx as nx

from GNN.DynamicSimilarities.GraphAnalysis.utils import compute_similarities_1vsAll, compute_distances_1vsAll, plot_dynamic_graphs

def build_dynamic_graph_with_calculated_threshold(target_id, target_preds, df_wide, cat_labels, date_cols, metric, fixed_threshold, enable_edges_within_star=True, enable_second_degree=False):
    G = nx.Graph()
    cat = cat_labels.get(target_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
    G.add_node(target_id, cat_label=cat)
    
    # Extract data for the window
    window_data = df_wide[date_cols]
    all_ts = window_data.values
    item_ids = window_data.index.values
    
    target_ts = np.array(target_preds)
    distance_metrics=['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
    similarity_metrics=['pearson', 'spearman', 'kendall']
   
    is_distance = metric in distance_metrics

    # Compute metric-specific scores
    if is_distance:
        scores = compute_distances_1vsAll(target_ts, all_ts, metric=metric)
    else:
        scores = compute_similarities_1vsAll(target_ts, all_ts, metric=metric)
        
    active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
    valid_mask = (item_ids != target_id) & active_items_mask
    
    if np.sum(np.abs(target_ts)) == 0:
        valid_mask = np.zeros_like(valid_mask, dtype=bool)
        
    valid_item_ids = item_ids[valid_mask]
    valid_scores = scores[valid_mask]
    valid_original_idxs = np.arange(len(item_ids))[valid_mask]
    
    
    if is_distance:
        final_mask = valid_scores <= fixed_threshold
    else:
        final_mask = valid_scores >= fixed_threshold
        
    selected_scores = valid_scores[final_mask]
    selected_ids = valid_item_ids[final_mask]
    selected_orig_idxs = valid_original_idxs[final_mask]
    
    neighbor_indices = []
    for orig_idx, score_val, other_id in zip(selected_orig_idxs, selected_scores, selected_ids):
        neighbor_indices.append((orig_idx, other_id))
        if not G.has_node(other_id):
            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
            G.add_node(other_id, cat_label=cat_other)
        G.add_edge(target_id, other_id, weight=float(score_val))
            
    if enable_edges_within_star and len(neighbor_indices) > 1:
        n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
        n_ids = [oid for _, oid in neighbor_indices]
        
        for i, n_id1 in enumerate(n_ids):
            n1_ts = n_ts_array[i]
            
            if is_distance:
                n_scores = compute_distances_1vsAll(n1_ts, n_ts_array, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(n1_ts, n_ts_array, metric=metric)
                
            for j in range(i + 1, len(n_ids)):
                    # Here we apply the same calculated global threshold
                condition = (n_scores[j] <= fixed_threshold) if is_distance else (n_scores[j] >= fixed_threshold)
                if condition:
                    if np.sum(np.abs(n1_ts)) > 0 and np.sum(np.abs(n_ts_array[j])) > 0:
                        G.add_edge(n_id1, n_ids[j], weight=float(n_scores[j]))
                        
    if enable_second_degree and len(neighbor_indices) > 0:
        for orig_idx, _ in neighbor_indices:
            target_neighbor_ts = all_ts[orig_idx]
            
            if is_distance:
                n_scores = compute_distances_1vsAll(target_neighbor_ts, all_ts, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(target_neighbor_ts, all_ts, metric=metric)
                
            for valid_idx, is_valid in enumerate(valid_mask):
                if is_valid and valid_idx != orig_idx:  
                    val_sub = n_scores[valid_idx]
                    other_id = item_ids[valid_idx]
                    
                    add_edge = False
                    if is_distance:
                        if val_sub <= fixed_threshold:
                            add_edge = True
                    else:
                        if val_sub >= fixed_threshold:
                            add_edge = True
                            
                    if add_edge:
                        if not G.has_node(other_id):
                            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                            G.add_node(other_id, cat_label=cat_other)
                        if not G.has_edge(item_ids[orig_idx], other_id):
                            G.add_edge(item_ids[orig_idx], other_id, weight=float(val_sub))
                            
    return G


def get_dynamic_embedding(G, graph2vec_model, dimensions=64):
    if G.number_of_nodes() == 0:
        return np.zeros(dimensions)
    if graph2vec_model is not None:
        emb = graph2vec_model.infer([G])
        return emb[0]
    return np.zeros(dimensions)
    
def graph2vec_inference(
    metric, window_size, step_size, threshold, model, df, df_wide, cat_labels, date_col, scaler, exog_scaler, test_start_idx, seq_length, forecast_window, 
    device, item_id, store_id, seed, criterion, val_scaled, test_scaled=None,
    exog_val_scaled=None, exog_test_scaled=None, exog_test_raw=None, exog_cols=None, save_plot_path=None,
    node_embeddings=None, graph2vec_model=None, enable_edges_within_star=True, enable_second_degree=False, percentile=None,
    create_plots=False
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
        # Aligned correctly with target slicing without the +1 shift
        emb_slice = node_embeddings[test_start_idx - seq_length : test_start_idx].tolist()
        
        # If node_embeddings is missing the inference steps, pad with the last known embedding
        while len(emb_slice) < seq_length:
            emb_slice.append(emb_slice[-1])
        
        current_emb_seq = emb_slice
        emb_dim = len(current_emb_seq[0]) if current_emb_seq else 64
    else:
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
            
        if node_embeddings is not None or current_emb_seq is not None:
            emb_cols_str = ",".join([f"Emb_{i}" for i in range(emb_dim)])
            header_str += f",{emb_cols_str}"
            
        inf_log_file.write(header_str + "\n")
        
        with torch.no_grad():
            for step in range(total_steps):
                
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
                
                # If we are in the warmup period and have actual test data, use the ground truth
                if step < window_size and test_scaled is not None and step < len(test_scaled):
                    pred = test_scaled[step].item() if hasattr(test_scaled[step], 'item') else test_scaled[step]
                
                pred_unscaled = scaler.inverse_transform([[pred]])[0, 0]
                warmup_preds_unscaled.append(pred_unscaled)
                
                # Pad the first window_size days with NaN so the plot starts on the window_size-th day
                if step < window_size:
                    forecast.append(np.nan)
                else:
                    forecast.append(pred)

                # Logging
                x_str = str(x_np.tolist()).replace('"', "'")
                row_str = f'{step},"{x_str}",{pred},{pred_unscaled}'
                
                if exog_cols and len(exog_cols) > 0:
                    last_exog_scaled = current_exog_arr[-1]
                    last_exog_raw = exog_scaler.inverse_transform(last_exog_scaled.reshape(1, -1))[0]
                    last_exog_unscaled_str = ",".join([str(v) for v in last_exog_raw.tolist()])
                    last_exog_scaled_str = ",".join([str(v) for v in last_exog_scaled.tolist()])
                    row_str += f',{last_exog_unscaled_str},{last_exog_scaled_str}'
                    
                if node_embeddings is not None or current_emb_seq is not None:
                    last_emb = current_emb_arr[-1]
                    last_emb_str = ",".join([str(v) for v in last_emb.tolist()])
                    row_str += f',{last_emb_str}'
                    
                inf_log_file.write(row_str + '\n')


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
                    end_date_idx = test_start_idx + step + 1
                    start_date_idx = end_date_idx - window_size
                    
                    if start_date_idx >= 0 and end_date_idx <= len(df_wide_cols):
                        date_cols_window = df_wide_cols[start_date_idx : end_date_idx]
                        
                        actual_history_needed = window_size - len(warmup_preds_unscaled)
                        if actual_history_needed > 0:
                            # We need some history from before the test_start_idx
                            history_dates = df_wide_cols[start_date_idx : test_start_idx]
                            history_unscaled = df_wide.loc[item_id, history_dates].tolist()
                            target_preds_window = history_unscaled + warmup_preds_unscaled
                        else:
                            target_preds_window = warmup_preds_unscaled[-window_size:]

                        new_G = build_dynamic_graph_with_calculated_threshold(
                            target_id=item_id,
                            target_preds=target_preds_window, 
                            df_wide=df_wide, 
                            cat_labels=cat_labels,
                            date_cols=date_cols_window,
                            metric=metric,
                            fixed_threshold=threshold,
                            enable_edges_within_star=enable_edges_within_star,
                            enable_second_degree=enable_second_degree
                        )
                        
                        new_emb = get_dynamic_embedding(new_G, graph2vec_model, dimensions=emb_dim)
                        current_emb_seq = current_emb_seq[1:] + [new_emb.tolist()]
                        
                        if create_plots:
                            # Save the generated graph plot for this inference step
                            curr_dir = os.path.dirname(os.path.abspath(__file__))
                            
                            # Incorporate seed and itertools parameters into the folder path
                            label_params = f"pct_{percentile}_w_{window_size}_s_{step_size}_e_{enable_edges_within_star}_2nd_{enable_second_degree}"
                            inf_plot_output_dir = os.path.join(curr_dir, 'grid_search_plots', f'seed_{seed}', 'InferredGraphs', str(item_id), metric, label_params)
                            os.makedirs(inf_plot_output_dir, exist_ok=True)
                            
                            start_date_str = str(date_cols_window[0])
                            end_date_str = str(date_cols_window[-1])
                            new_G.graph['start_date'] = start_date_str
                            
                            # Identify the target day being inferred using this graph
                            if (test_start_idx + step + 1) < len(df):
                                next_y_date = df[date_col].iloc[test_start_idx + step + 1].date()
                            else:
                                next_y_date = (pd.to_datetime(y_date) + pd.Timedelta(days=1)).date()
                                
                            new_G.graph['end_date'] = f"{end_date_str}_inferring_{next_y_date}"
                            
                            try:
                                plot_dynamic_graphs(
                                    graphs=[new_G],
                                    product_id=item_id,
                                    metric=metric,
                                    plot_dir=inf_plot_output_dir,
                                    residuals=False,
                                    enable_edges_within_star=enable_edges_within_star,
                                    enable_second_degree=enable_second_degree,
                                    num_plots=None,
                                    window_size=window_size,
                                    step_size=step_size,
                                    threshold=threshold,
                                    percentile=None
                                )
                            except Exception as e:
                                print(f"Failed to plot inference graph for date {end_date_str}: {e}")

                    else:
                        next_emb = current_emb_seq[-1]
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