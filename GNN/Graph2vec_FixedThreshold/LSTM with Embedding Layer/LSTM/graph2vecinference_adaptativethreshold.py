import torch
import numpy as np
import pandas as pd
import os
import time
import networkx as nx
import holidays
from utils import compute_similarities_1vsAll, compute_distances_1vsAll

def generate_exogenous_features(df, exog_cols, date_col='date', target_col='value', group_cols=None):
    """
    Generates specific calendar, cyclical, and holiday exogenous features for a DataFrame
    based on the provided `exog_cols` list.
    """
    df = df.copy()
    
    if group_cols is None:
        group_cols = [c for c in ['item_id', 'store_id'] if c in df.columns]
        if not group_cols:
            group_cols = None
    
    # -----------------------------------------------------------------------------
    # FEATURE BUILDER DICTIONARY
    # -----------------------------------------------------------------------------
    def _get_holidays():
        us_holidays = holidays.US()
        return us_holidays, pd.to_datetime(sorted(us_holidays.keys()))

    _holidays_cache = None
    def get_holiday_dates():
        nonlocal _holidays_cache
        if _holidays_cache is None:
            _holidays_cache = _get_holidays()
        return _holidays_cache

    builders = {
        # BASIC CALENDAR PARTS
        "day_of_week": lambda d: d[date_col].dt.dayofweek.astype(int),
        "day_of_month": lambda d: d[date_col].dt.day.astype(int),
        "month": lambda d: d[date_col].dt.month.astype(int),
        "moy": lambda d: (d[date_col].dt.month - 1).astype(int),
        "quarter": lambda d: d[date_col].dt.quarter.astype(int),
        "doy": lambda d: (d[date_col].dt.dayofyear - 1).astype(int),
        "week_of_year": lambda d: d[date_col].dt.isocalendar().week.astype(int),
        "year": lambda d: d[date_col].dt.year.astype(int),
        
        "is_weekend": lambda d: d[date_col].dt.dayofweek.isin([5, 6]).astype(int),
        "is_monday": lambda d: (d[date_col].dt.dayofweek == 0).astype(int),
        "is_friday": lambda d: (d[date_col].dt.dayofweek == 4).astype(int),

        "is_month_start": lambda d: d[date_col].dt.is_month_start.astype(int),
        "is_month_end": lambda d: d[date_col].dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d[date_col].dt.is_quarter_start.astype(int),
        "is_quarter_end": lambda d: d[date_col].dt.is_quarter_end.astype(int),
        "week_of_month": lambda d: ((d[date_col].dt.day - 1) // 7 + 1).astype(int),

            # CYCLICAL ENCODINGS — base harmonics
        "dow_sin": lambda d: np.sin(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos": lambda d: np.cos(2 * np.pi * d[date_col].dt.dayofweek / 7),

        "dom_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),
        "dom_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),

        "wom_sin": lambda d: np.sin(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),
        "wom_cos": lambda d: np.cos(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),

        "month_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "quarter_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),
        "quarter_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),

        "woy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # CYCLICAL ENCODINGS — 2nd harmonics
        "dow_sin2": lambda d: np.sin(4 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos2": lambda d: np.cos(4 * np.pi * d[date_col].dt.dayofweek / 7),

        "month_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "woy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # EXACT HOLIDAYS
        "is_holiday": lambda d: d[date_col].isin(get_holiday_dates()[1]).astype(int),
        "is_christmas": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 25)).astype(int),
        "is_thanksgiving": lambda d: d[date_col].apply(lambda x: 1 if get_holiday_dates()[0].get(x) == "Thanksgiving Day" else 0),
        "is_black_friday": lambda d: d[date_col].isin(
            pd.to_datetime([day for day, name in get_holiday_dates()[0].items() if name == "Thanksgiving Day"]) + pd.Timedelta(days=1)
        ).astype(int),
        "is_christmas_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 24)).astype(int),
        "is_new_year_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 31)).astype(int),

        # PROMOTIONS
        "promo_type_FRPG": lambda d: d.get("promo_type_FRPG", pd.Series(0, index=d.index)).astype(int),
        "promo_value_FRPG": lambda d: d.get("promo_value_FRPG", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_GAS": lambda d: d.get("promo_type_GAS", pd.Series(0, index=d.index)).astype(int),
        "promo_value_GAS": lambda d: d.get("promo_value_GAS", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_BOGO": lambda d: d.get("promo_type_BOGO", pd.Series(0, index=d.index)).astype(int),
        "promo_value_BOGO": lambda d: d.get("promo_value_BOGO", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_DISC": lambda d: d.get("promo_type_DISC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_DISC": lambda d: d.get("promo_value_DISC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRC": lambda d: d.get("promo_type_CIRC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRC": lambda d: d.get("promo_value_CIRC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRE": lambda d: d.get("promo_type_CIRE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRE": lambda d: d.get("promo_value_CIRE", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CLCP": lambda d: d.get("promo_type_CLCP", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CLCP": lambda d: d.get("promo_value_CLCP", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_LFPE": lambda d: d.get("promo_type_LFPE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_LFPE": lambda d: d.get("promo_value_LFPE", pd.Series(0.0, index=d.index)).astype(float),
        
        #Trend indicators could be added here as well, but we will handle them separately in the main function to avoid data leakage
        # TREND
        "time_idx": lambda d: (d[date_col] - d[date_col].min()).dt.days.astype(int),
        "time_idx_sq": lambda d: ((d[date_col] - d[date_col].min()).dt.days.astype(float) ** 2),
        
    }

    # Generate only the requested features
    for col in exog_cols:
        if col in builders:
            df[col] = builders[col](df)
        elif col.startswith("is_pre_holiday_") or col.startswith("is_post_holiday_"):
            parts = col.split("_")
            lag = int(parts[-1])
            is_pre = "pre" in parts
            
            _, holiday_dates = get_holiday_dates()
            
            # Efficiently compute proximity using sets and exact matches (avoiding loop row-by-row)
            df[col] = 0
            for h in holiday_dates:
                target_date = h - pd.Timedelta(days=lag) if is_pre else h + pd.Timedelta(days=lag)
                df.loc[df[date_col] == target_date, col] = 1
                
        elif col.startswith("lag_"):
            lag = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].shift(lag).fillna(0)
            else:
                df[col] = df[target_col].shift(lag).fillna(0)
                
        elif col.startswith("rolling_mean_excl_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()).fillna(0)
            else:
                df[col] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
                
        elif col.startswith("rolling_mean_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean()).fillna(0)
            else:
                df[col] = df[target_col].rolling(window=window, min_periods=1).mean().fillna(0)

        elif col == "is_bridge_day":
            _, holiday_dates = get_holiday_dates()
            holiday_set = set(holiday_dates)
            
            df[col] = 0
            for idx in df.index:
                d = df.at[idx, date_col]
                prev_day = d - pd.Timedelta(days=1)
                next_day = d + pd.Timedelta(days=1)
                if ((prev_day in holiday_set and d.dayofweek == 4) or 
                    (next_day in holiday_set and d.dayofweek == 0)):
                    df.at[idx, col] = 1

        elif col in df.columns:
            # If the column exists in the dataset natively (e.g., store_id, cat_label) and isn't a builder key, just pass
            pass
                    
        else:
            print(f"Warning: Builder for feature '{col}' not found.")

    return df



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
    create_plots=False, product_scalers=None
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

                x_np = np.column_stack(features_to_stack)
                x = torch.FloatTensor(x_np).unsqueeze(0).to(device)

                # Embeddings are passed separately so the model's trainable
                # projection layer can transform them before the LSTM.
                if current_emb_seq is not None:
                    emb_np = np.array(current_emb_seq)          # (seq_len, emb_dim)
                    emb = torch.FloatTensor(emb_np).unsqueeze(0).to(device)
                else:
                    emb = torch.zeros(1, x.shape[1], 0).to(device)

                # Predict next value
                pred = model(x, emb).cpu().numpy()[0, 0]
                
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
                    last_emb = current_emb_seq[-1]
                    last_emb_str = ",".join([str(v) for v in (last_emb.tolist() if hasattr(last_emb, 'tolist') else last_emb)])
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
                        
                        distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                        is_distance = metric in distance_metrics

                        if actual_history_needed > 0:
                            # We need some history from before the test_start_idx
                            history_dates = df_wide_cols[start_date_idx : test_start_idx]
                            history_vals = df_wide.loc[item_id, history_dates].tolist() # Context value matches passed dataframe
                            
                            if is_distance and product_scalers is not None and item_id in product_scalers:
                                warmup_preds_scaled_z = product_scalers[item_id].transform(np.array(warmup_preds_unscaled).reshape(-1, 1)).flatten().tolist()
                                target_preds_window = history_vals + warmup_preds_scaled_z
                            else:
                                target_preds_window = history_vals + warmup_preds_unscaled
                        else:
                            if is_distance and product_scalers is not None and item_id in product_scalers:
                                warmup_preds_scaled_z = product_scalers[item_id].transform(np.array(warmup_preds_unscaled[-window_size:]).reshape(-1, 1)).flatten().tolist()
                                target_preds_window = warmup_preds_scaled_z
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
                                from plots import plot_dynamic_graphs
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