"""
Recursive (one-step-at-a-time) inference for the PER-STEP GCN + LSTM forecaster.

The per-step model expects *L* graphs per sample (one per lookback day), so at
inference we maintain a rolling deque of length L of ego-graphs alongside the
LSTM lookback window.  At each forecasting step we:

  1. Run the model on the current (L-graph deque, L-row ts_seq).
  2. Roll the LSTM window forward (drop oldest target value, append ŷ;
     advance the exogenous row to the step we are about to predict).
  3. Roll the graph deque forward: drop the oldest graph, append a new
     "current" graph.  If the caller supplies a sequence of *future*
     ego-graphs via ``future_graphs`` we use those (one per recursive
     step); otherwise we reuse the last graph and only refresh its
     target-node feature row from the rolling target-window values.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from GNN.DynamicSimilarities.LSTM_GCN_1_graph_per_lookback.gcn_lstm_dataset import _window_node_features  # internal helper


@torch.no_grad()
def recursive_forecast(
    model,
    initial_ts_seq: np.ndarray,
    initial_graphs: Sequence[Data],
    target_node_idx: int,
    horizon: int,
    device: torch.device,
    target_window_values: Optional[np.ndarray] = None,
    future_graphs: Optional[Sequence[Data]] = None,
    update_target_node_features: bool = True,
):
    """
    Roll-out ``horizon`` predictions in scaled space.

    Parameters
    ----------
    model                  : SimpleGCNLSTMForecaster (per-step variant)
    initial_ts_seq         : (L, F) np.ndarray – last observed LSTM window
                             (column 0 must be the scaled target value)
    initial_graphs         : list of L Data ego-graphs, one per lookback day,
                             aligned exactly as the dataset would have served
                             them (oldest first, newest last).
    target_node_idx        : index of the target node inside each graph
                             (0 by convention)
    horizon                : number of recursive steps to roll out
    device                 : torch device
    target_window_values   : optional 1-D array of length ``graph_window_size``
                             with the most recent target-node raw values.  Used
                             to refresh the target-node feature row of the new
                             graph at each step when ``future_graphs`` is None.
    future_graphs          : optional sequence of length ``horizon`` of
                             pre-built ego-graphs for the forecast steps.  When
                             provided, these are appended to the deque in
                             order; otherwise the last graph is reused (with an
                             updated target-node feature row).
    update_target_node_features
                           : when True and ``future_graphs`` is None, refresh
                             the target-node row of the reused graph from the
                             rolling target window.

    Returns
    -------
    np.ndarray of shape (horizon,) — scaled predictions.
    """
    model.eval()

    ts = np.asarray(initial_ts_seq, dtype=np.float32).copy()       # (L, F)
    if ts.ndim != 2:
        raise ValueError("initial_ts_seq must be 2-D (L, F)")
    L = ts.shape[0]
    if len(initial_graphs) != L:
        raise ValueError(
            f"initial_graphs must have length L={L}, got {len(initial_graphs)}"
        )

    graphs: "deque[Data]" = deque((g.clone() for g in initial_graphs), maxlen=L)

    if target_window_values is not None:
        win = np.asarray(target_window_values, dtype=np.float32).copy()
    else:
        win = None

    preds: List[float] = []

    for step in range(horizon):
        ts_t  = torch.from_numpy(ts).unsqueeze(0).to(device)             # (1, L, F)
        batch = Batch.from_data_list(list(graphs)).to(device)            # B*L = L
        tidx  = batch.ptr[:-1].to(device)
        out   = model(batch, tidx, ts_t)                                 # (1, H, 1)
        y_hat = float(out[0, -1, 0].detach().cpu().item())
        preds.append(y_hat)

        # roll the LSTM window: shift left, append a new last row carrying ŷ
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # roll graph deque
        if future_graphs is not None:
            new_graph = future_graphs[step].clone()
        else:
            new_graph = graphs[-1].clone()
            if update_target_node_features and win is not None:
                win = np.concatenate([win[1:], [y_hat]])
                new_feats = _window_node_features(win[None, :])           # (1, 8)
                new_graph.x[target_node_idx] = torch.from_numpy(new_feats[0])
        graphs.append(new_graph)   # deque auto-drops the oldest

    return np.array(preds, dtype=np.float32)

