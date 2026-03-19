import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class GraphAwareTimeSeriesDataset(Dataset):
    def __init__(
        self,
        panel_df,              # pd.DataFrame shape: (T, N), index=dates, columns=global_items
        seq_length,
        horizon,
        graph_window_info,
        item_to_idx,
        exog_df=None,          # pd.DataFrame shape: (T, F_exog), index=dates
    ):
        """
        Retail Panel dataset that extracts temporally valid target and exogenous sequences
        for each item, mapped natively to its dynamic graph index.
        """
        self.panel = panel_df.values.astype(np.float32)
        self.dates = pd.to_datetime(panel_df.index)
        self.items = list(panel_df.columns)
        self.item_to_idx = item_to_idx
        self.seq_length = seq_length
        self.horizon = horizon
        
        self.has_exog = exog_df is not None
        if self.has_exog:
            self.exog = exog_df.values.astype(np.float32)

        self.graph_end_dates = pd.to_datetime(
            [w["end_date"] for w in graph_window_info]
        )

        self.samples = []

        if len(self.graph_end_dates) == 0:
            return

        T, N = self.panel.shape

        # We construct dataset starting ONLY from points where the history is long enough for the LSTM
        # AND enough time has passed that a valid graph is available.
        for t in range(seq_length - 1, T - horizon):
            last_observed_date = self.dates[t]

            valid_graph_idx = np.where(self.graph_end_dates <= last_observed_date)[0]
            if len(valid_graph_idx) == 0:
                continue # PREVENT LEAKAGE: Wait until at least one graph exists

            graph_idx = valid_graph_idx[-1]

            # Append a distinct sample per item in the network!
            for item_idx in range(N):
                self.samples.append((t, item_idx, graph_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We unpack our precalculated pointer coordinates!
        t, item_idx, graph_idx = self.samples[idx]

        # Extract sequence from time t - L + 1 up to t, for the specific item
        ts_x = self.panel[t - self.seq_length + 1 : t + 1, item_idx:item_idx+1]
        
        # Extract target horizons up to t + horizon, for the specific item
        y = self.panel[t + 1 : t + 1 + self.horizon, item_idx]

        # Align exogenous variables perfectly natively inside the feature block
        if self.has_exog:
            exog_x = self.exog[t - self.seq_length + 1 : t + 1]
            ts_x = np.column_stack([ts_x, exog_x])

        return {
            "ts_x": torch.tensor(ts_x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "graph_idx": torch.tensor(graph_idx, dtype=torch.long),
            "target_node_idx": torch.tensor(item_idx, dtype=torch.long),
        }
