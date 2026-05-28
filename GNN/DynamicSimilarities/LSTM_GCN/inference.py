"""
Recursive (one-step-at-a-time) inference for the GCN + LSTM forecaster.

At each forecasting step we:
  1. Roll the LSTM input window forward by replacing the oldest target
     value with the previous prediction.
  2. Roll the *target-node* feature row in the most-recent ego-graph
     forward as well (neighbour nodes keep their last observed window
     stats — this matches the recursive convention used elsewhere in the
     project, since we do not have ground-truth future observations for
     the neighbours).
  3. Run the model on this updated (pyg_graph, ts_seq) pair to obtain
     the next prediction.  Predictions are returned in scaled space; the
     caller applies the inverse MinMax transform.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from GNN.DynamicSimilarities.LSTM_GCN.gcn_lstm_dataset import _window_node_features  # internal helper


@torch.no_grad()
def recursive_forecast(
    model,
    initial_ts_seq: np.ndarray,
    initial_graph: Data,
    target_node_idx: int,
    horizon: int,
    device: torch.device,
    target_window_values: Optional[np.ndarray] = None,
    update_target_node_features: bool = True,
):
    """
    Roll-out `horizon` predictions in scaled space.

    Parameters
    ----------
    model                  : SimpleGCNLSTMForecaster (already loaded + eval mode)
    initial_ts_seq         : (L, F) np.ndarray   – last observed LSTM window
                             (column 0 must be the scaled target value)
    initial_graph          : ``Data`` ego-graph aligned to the last observed step
    target_node_idx        : index of the target node inside the graph
                             (typically 0 by convention)
    horizon                : number of recursive steps to roll out
    device                 : torch device
    target_window_values   : optional 1-D array of length ``graph_window_size``
                             of the most recent target-node raw values.  When
                             provided and ``update_target_node_features`` is
                             True, we update the target row of ``graph.x`` at
                             each step using the new prediction.
    update_target_node_features
                           : if False, the graph is kept frozen across the
                             roll-out (graph snapshot semantics).

    Returns
    -------
    np.ndarray of shape (horizon,) — scaled predictions.
    """
    model.eval()

    ts = np.asarray(initial_ts_seq, dtype=np.float32).copy()       # (L, F)
    if ts.ndim != 2:
        raise ValueError("initial_ts_seq must be 2-D (L, F)")

    # snapshot of the rolling target window (for node-feature updates)
    if target_window_values is not None:
        win = np.asarray(target_window_values, dtype=np.float32).copy()
    else:
        win = None

    graph = initial_graph.clone()
    preds = []

    for _ in range(horizon):
        ts_t  = torch.from_numpy(ts).unsqueeze(0).to(device)             # (1, L, F)
        batch = Batch.from_data_list([graph]).to(device)
        tidx  = batch.ptr[:-1].to(device)

        out   = model(batch, tidx, ts_t)                                  # (1, H, 1)
        y_hat = float(out[0, -1, 0].detach().cpu().item())
        preds.append(y_hat)

        # roll the LSTM window: shift left, append new target value
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # optionally roll the target-node feature row in the graph
        if update_target_node_features and win is not None:
            win = np.concatenate([win[1:], [y_hat]])
            new_feats = _window_node_features(win[None, :])               # (1, 8)
            graph = graph.clone()
            graph.x[target_node_idx] = torch.from_numpy(new_feats[0])

    return np.array(preds, dtype=np.float32)
