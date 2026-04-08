import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

def scale_exogenous_features(df, train_slice, categorical_cols=None, continuous_cols=None, binary_cols=None):
    df_scaled = df.copy()
    final_exog_cols = []
    
    cat_scaler = None
    cont_scaler = None
    
    # Scale categorical columns with OneHotEncoder
    if categorical_cols:
        cat_train = df.iloc[train_slice][categorical_cols].values
        cat_scaler = MinMaxScaler()  # Use MinMaxScaler to convert categories to a 0-1 range before OHE
        cat_scaler.fit(cat_train)
        
        cat_full_scaled = cat_scaler.transform(df[categorical_cols].values)
        cat_feature_names = cat_scaler.get_feature_names_out(categorical_cols)
        
        # Add new OHE columns and drop the originals
        df_cat = pd.DataFrame(cat_full_scaled, columns=cat_feature_names, index=df.index)
        df_scaled = pd.concat([df_scaled.drop(columns=categorical_cols), df_cat], axis=1)
        final_exog_cols.extend(cat_feature_names)
        
    # Scale continuous columns
    if continuous_cols:
        cont_train = df.iloc[train_slice][continuous_cols].values
        cont_scaler = MinMaxScaler()
        cont_scaler.fit(cont_train)
        
        cont_full_scaled = cont_scaler.transform(df[continuous_cols].values)
        df_scaled[continuous_cols] = cont_full_scaled
        final_exog_cols.extend(continuous_cols)
        
    # Keep binary columns as they are
    if binary_cols:
        final_exog_cols.extend(binary_cols)
        
    return df_scaled, final_exog_cols, cat_scaler, cont_scaler

def build_dynamic_exog_row(next_date, target_history, exog_cols,
                           lag_cols=None, rolling_cols=None, df_row=None):
    """
    Build one RAW exogenous row for next_date using current RAW target history.
    """
    row = {}

    if lag_cols:
        for col in lag_cols:
            lag = int(col.split('_')[1])
            row[col] = target_history[-lag] if len(target_history) >= lag else 0.0

    if rolling_cols:
        for col in rolling_cols:
            window = int(col.split('_')[2])
            if len(target_history) >= window:
                row[col] = float(np.mean(target_history[-window:]))
            else:
                row[col] = float(np.mean(target_history)) if len(target_history) > 0 else 0.0

    if df_row is not None:
        for col in exog_cols:
            if col not in row and col in df_row.index:
                row[col] = df_row[col]

    return pd.DataFrame([[row.get(col, 0.0) for col in exog_cols]], columns=exog_cols)



def transform_exog_row(row_df, categorical_cols=None, continuous_cols=None, binary_cols=None,
                       cat_scaler=None, cont_scaler=None, final_exog_cols=None):
    parts = []

    if categorical_cols:
        # Force column order
        cat_arr = cat_scaler.transform(row_df[categorical_cols].values)
        cat_names = cat_scaler.get_feature_names_out(categorical_cols)
        parts.append(pd.DataFrame(cat_arr, columns=cat_names, index=row_df.index))

    if continuous_cols:
        # Force column order precisely as fitted
        cont_arr = cont_scaler.transform(row_df[continuous_cols].values)
        parts.append(pd.DataFrame(cont_arr, columns=continuous_cols, index=row_df.index))

    if binary_cols:
        # Force column order
        parts.append(row_df[binary_cols].copy())

    out = pd.concat(parts, axis=1)

    if final_exog_cols is not None:
        # Reindex to ensure strictly ordered columns matching the training tensors
        out = out.reindex(columns=final_exog_cols, fill_value=0.0)

    return out




from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_test, y_pred):
    
    def POCID(y_test, y_pred):
        diff_original = y_test[1:] - y_test[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        is_positive = (diff_original * diff_pred) > 0
        return is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    pocid = POCID(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "bias": bias, "score": score, "pocid": pocid}

def analyze_distance_distribution(df, product_id, window_size, metrics, plot_dir):
    """
    Computes pair-wise distances for a sample window and plots a histogram
    to help determine percentiles and thresholds.
    """
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    print(f"\n--- Analyzing Distance Distributions for {product_id} ---")
    
    # Pick a sample window with some activity
    # Let's just use the first window that has non-zero sales for the product
    time_steps = df.shape[1]
    window_data = None
    start_idx_used = 0
    
    for start_idx in range(0, time_steps - window_size + 1):
        end_idx = start_idx + window_size
        candidate_data = df.iloc[:, start_idx:end_idx]
        target_ts = candidate_data.loc[product_id].values
        
        if np.sum(np.abs(target_ts)) > 0:
            window_data = candidate_data
            start_idx_used = start_idx
            break
            
    if window_data is None:
        print("Warning: Could not find a sample window where target product has >0 sales. Using first window.")
        window_data = df.iloc[:, 0:window_size]
        
    start_date = str(window_data.columns[0]).split('T')[0]
    end_date = str(window_data.columns[-1]).split('T')[0]
    
    print(f"Selected sample window: {start_date} to {end_date} (Indices {start_idx_used} to {start_idx_used+window_size})")

    target_ts = window_data.loc[product_id].values
    all_ts = window_data.values
    item_ids = window_data.index.values
    
    # Filter active items exactly like in graph construction
    active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
    valid_mask = (item_ids != product_id) & active_items_mask
    valid_all_ts = all_ts[valid_mask]
    
    os.makedirs(plot_dir, exist_ok=True)
    
    distributions = {}
    
    for metric in metrics:
        print(f"Computing distances for {metric}...")
        try:
            # We use valid_all_ts instead of all_ts to only analyze items we'd actually consider
            dists = compute_distances_1vsAll(target_ts, valid_all_ts, metric=metric)
            # Remove entirely infinite or NaN distances if any exist
            valid_dists = dists[np.isfinite(dists) & ~np.isnan(dists)]
            
            if len(valid_dists) > 0:
                distributions[metric] = valid_dists
                
                # Calculate key percentiles
                p1 = np.percentile(valid_dists, 1)
                p5 = np.percentile(valid_dists, 5)
                p10 = np.percentile(valid_dists, 10)
                p25 = np.percentile(valid_dists, 25)
                p50 = np.percentile(valid_dists, 50)
                
                print(f"  {metric.upper()} Percentiles:")
                print(f"    1st : {p1:.4f}")
                print(f"    5th : {p5:.4f}")
                print(f"    10th: {p10:.4f}")
                print(f"    Median: {p50:.4f}")
                
                # Plot Histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(valid_dists, bins=50, kde=True)
                
                # Add percentile lines
                plt.axvline(p1, color='red', linestyle='dashed', linewidth=2, label=f'1st % ({p1:.2f})')
                plt.axvline(p5, color='orange', linestyle='dashed', linewidth=2, label=f'5th % ({p5:.2f})')
                plt.axvline(p10, color='green', linestyle='dashed', linewidth=2, label=f'10th % ({p10:.2f})')
                
                plt.title(f'Distance Distribution: {metric.upper()}\nTarget {product_id} ({start_date} to {end_date})')
                plt.xlabel('Distance')
                plt.ylabel('Frequency (Number of Items)')
                plt.legend()
                
                dist_plot_path = os.path.join(plot_dir, f'dist_histogram_{metric}.png')
                plt.savefig(dist_plot_path)
                plt.close()
                print(f"  Saved histogram to {dist_plot_path}")
            else:
                print(f"  Warning: No valid distances returned for {metric}")
                
        except Exception as e:
            print(f"  Error computing {metric}: {e}")
            
    return distributions