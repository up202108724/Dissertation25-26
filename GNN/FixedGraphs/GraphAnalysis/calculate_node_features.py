import pandas as pd 
import numpy as np

def compute_window_node_features(df_pivot: pd.DataFrame):
        num_items = df_pivot.shape[1]
        window_size = df_pivot.shape[0]
        features = []
        
        def calc_mk_tau(ts):
            n = len(ts)
            if n < 2:
                return 0.0
            idx = np.triu_indices(n, 1)
            diff = ts[idx[1]] - ts[idx[0]]
            s = np.sum(np.sign(diff))
            return float(s / (n * (n - 1) / 2))

        for j in range(num_items):
            item_ts = df_pivot.iloc[:, j].values
            last_demand = item_ts[-1] if window_size > 0 else 0
            mean7 = np.mean(item_ts[-7:]) if window_size >= 7 else np.mean(item_ts)
            mean28 = np.mean(item_ts[-28:]) if window_size >= 28 else np.mean(item_ts)
            std28 = np.std(item_ts[-28:]) if window_size >= 28 else np.std(item_ts)
            if window_size >= 28:
                zero_ratio28 = np.mean(item_ts[-28:] == 0)
                slope28 = np.polyfit(np.arange(28), item_ts[-28:], 1)[0]
                min_28 = np.min(item_ts[-28:])
                max_28 = np.max(item_ts[-28:])
            elif window_size > 1:
                zero_ratio28 = np.mean(item_ts == 0)
                slope28 = np.polyfit(np.arange(window_size), item_ts, 1)[0]
                min_28 = np.min(item_ts)
                max_28 = np.max(item_ts)
            else:
                zero_ratio28 = np.mean(item_ts == 0)
                slope28 = 0.0
                min_28 = item_ts[0] if window_size > 0 else 0
                max_28 = item_ts[0] if window_size > 0 else 0
            features.append([last_demand, mean7, mean28, std28, zero_ratio28, slope28, min_28, max_28])
        return np.array(features)