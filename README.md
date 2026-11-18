# GCN / GAT / Graph2Vec + LSTM / Time-Distributed MLP

This folder contains the graph-augmented forecasters and their experiment runners.

## 1. Overview

For each product and each day, a similarity graph is built: the target product is connected to the products whose recent sales windows look similar. That graph is turned into a vector `z`, either by a trainable GNN (GCN or GAT) or by a frozen Graph2Vec embedding. `z` is then concatenated to the target's own time-series features at every lookback step and fed to an LSTM or a time-distributed MLP.

Per-step forecaster input:

```
[ y_t ‖ exog_t ‖ z_t ]     shape (B, L, 1 + n_exog + d_g)
```

```
sales windows ──► similarity graph (per day) ──► encoder ──► z_t ─┐
                                                                  ├─► concat ─► LSTM / TD-MLP ─► forecast
target series y_t, exog_t ────────────────────────────────────────┘
```

The forecasters are the combinations of {GCN, GAT, Graph2Vec} x {LSTM, MLP, DLinear, PatchTST}.

## 2. File map

| File | Role |
|---|---|
| `generate_graph2vecwithadaptativethreshold.py` | Builds the dynamic similarity graphs per window |
| `node_feature_builders.py` | Node-feature construction (stats, catch22/24, raw) |
| `_catch24_manual.py` | Fallback catch22/24 implementation if `pycatch22` is missing |
| `graph2vec.py` | `CustomGraph2Vec`: Weisfeiler-Lehman + Doc2Vec embeddings |
| `graph2vec_assemble.py` | Graph2Vec helpers: embedding to 1-node graph, dynamic graph rebuild, recursive inference |
| `gcn_lstm_dataset.py` | NetworkX to PyG conversion, per-step dataset and collate |
| `gat_tdmlp_dataset.py` | Dataset variant for the GAT / time-distributed MLP path |
| `gcn_models.py` | The six LSTM / MLP hybrids and `EdgeWeightedGATConv` |
| `gcn_models_modern.py` | The six DLinear / PatchTST hybrids (section 11) |
| `train_methods.py` | Training loops |
| `inference_strategies.py` | Recursive multi-step inference for the GCN/GAT models |
| `review_baselines.py` | Seasonal-naive, neighbour baselines and the static graph (section 10) |
| `graph2vec_gcn_gat_assmble_*.py` | Experiment runners (sequential, parallel workers, scalability) |
| `ablationlstm.py`, `ablationmlp.py` | Graph-free LSTM / MLP heads (ablation and neighbour baselines) |
| `attention_interpretability_analysis.py`, `representation_quality_analysis.py`, `winner_products.py` | Analysis of results |

Baselines live one level up, one folder per model: `GNN/LSTMBaseline`, `GNN/MLP_Baseline`, `GNN/DLinearBaseline`, `GNN/PatchTSTBaseline`. `GNN/window_forecasting.py` holds the training loop and recursive forecast shared by DLinear, PatchTST and the neighbour baselines. The input `dataset/data_andre_classified.feather` is produced by `EDA/classify_fulfilled.py`.

## 3. Graph construction

`build_dynamic_graph_with_calculated_threshold` (in `graph2vec_assemble.py`):

- For a window of `W` days, it computes the distance or similarity between the target and every other product. Supported metrics include `cid`, `dtw`, `euclidean`, `manhattan`, `hamming`, `sbd`, `msm`, `edr`, `lcss`, `lorentzian` and others.
- Products that pass a threshold become neighbours (`<=` for distances, `>=` for similarities). This gives a star graph centred on the target.
- Edges between neighbours (`enable_edges_within_star`) and second-degree neighbours (`enable_second_degree`) are optional.
- The edge weight is the similarity score. Distances are mapped to `1/(1+d)` (`utils.edge_weight_from_metric`), so bigger always means more similar.
- Products with an all-zero window are excluded, and an all-zero target gets no neighbours.

## 4. Node features (`node_feature_builders.py`)

Used by the GCN and GAT path. Each node is a product, described by a summary of its window of sales values.

- **Scalar stats:** `mean`, `std`, `min`, `max`, `first`, `last`, `slope`, `sum`, `zero_fraction`.
- **Blocks:** `raw` (the whole window, width = window size), `catch22` (22 time-series shape features), `catch24` (catch22 plus `DN_Mean` and `DN_Spread_Std`).
- **Presets** (`_MODE_FEATURE_LISTS`): `raw`, `stats` (8 stats), `catch22`, `catch24`, `catch24_minmax`, `catch24_minmaxlast`.
- **Extra stats:** the bare `catch22` and `catch24` presets get `min`, `max` and `zero_fraction` appended when `extra_stats=True` (so `catch24` has width 27). Explicit combos are taken verbatim.
- `resolve_feature_names` is the single source of truth, so the dataset build, the inference patching and the ablation widths agree by construction.
- NaN and Inf values are replaced by 0, and degenerate series give 0 for catch features.

`nx_window_to_pyg` (in `gcn_lstm_dataset.py`) converts a graph plus its window values into a PyG `Data` object:

- The target is always row 0.
- Edges are added in both directions with the similarity as `edge_attr`.
- An isolated target gets a self-loop.

## 5. Graph2Vec (`graph2vec.py`)

The frozen alternative. It uses no node features.

1. Each node is labelled with its product ID (`feature`), and `nx.weisfeiler_lehman_node_hashes` refines the labels for `wl_iterations` (default 2). These hashes act as the "words" describing the graph.
2. The sorted hashes form one document per graph. Doc2Vec in PV-DBOW mode (`dm=0`, gensim) learns one vector per document.
3. Identical graphs share one training document, which saves time and keeps results deterministic. Seeds, `workers=1` and a fixed MD5 hash function are used for the same reason.
4. `infer()` embeds new graphs at inference time with `infer_vector`.

The embedding is computed offline and does not change during forecasting training.

## 6. Encoders and forecasters (`gcn_models.py`)

| Encoder | Layers | Notes |
|---|---|---|
| GCN | `GCNConv -> ReLU -> Dropout -> GCNConv -> LayerNorm` | Similarity is used as `edge_weight`; `z` is the target node's output |
| GAT | `EdgeWeightedGATConv` (4 heads, concat) `-> ELU -> Dropout -> EdgeWeightedGATConv` (1 head) `-> LayerNorm` | Self-loop weight is 1.0 |
| Graph2Vec | `Linear -> ReLU -> Dropout -> LayerNorm` | The frozen embedding is carried as the single node of a 1-node graph |

`EdgeWeightedGATConv` is a custom `GATConv`. Stock GAT only uses `edge_attr` as an additive bias in the attention logits. Here each softmax attention coefficient is also multiplied by the edge weight and renormalised per target node. A low-similarity neighbour is therefore always down-weighted, while the attention stays learned.

### Heads

- **LSTM head:** the LSTM runs over `[y_t ‖ exog_t ‖ z_t]`. The last output goes through dropout and a linear layer.
- **Time-distributed MLP head:** a shared MLP (default hidden sizes `(64, 32)`) is applied to each timestep independently. The model concatenates the last step and the mean over steps, then applies a linear layer.

### Classes

| | LSTM | MLP |
|---|---|---|
| GCN | `SimpleGCNLSTMForecaster` | `SimpleGCNMLPForecaster` |
| GAT | `SimpleGATLSTMForecaster` | `SimpleGATMLPForecaster` |
| Graph2Vec | `SimpleGraph2VecLSTMForecaster` | `SimpleGraph2VecMLPForecaster` |

All share the interface `forward(pyg_batch, target_node_indices, ts_seq) -> (B, horizon, 1)`, so they use the same dataset, collate, training and inference code. Setting `model.ablate_z = True` zeroes `z`, which tests whether the graph helps.

## 7. How the graph is linked to the time series

- **One graph per lookback step.** For a sample at index `idx`, lookback step `i` uses `pyg_graphs[idx + 1 + i]`, the graph built on the `W` days before that day.
- **Batching.** `collate_pyg_ts` flattens `B` samples x `L` steps into one PyG `Batch` of `B*L` small graphs. `target_idx = ptr[:-1]` gives each graph's target node.
- **Encoding.** The encoder produces one `z` per graph. This is reshaped to `(B, L, d_g)` and concatenated with `ts_seq`.
- **Padding.** The first `graph_window_size` days have no graph and are filled with zero-feature padding graphs.
- **Training.** For GCN and GAT, the encoder and the head train jointly on the forecasting loss. For Graph2Vec, only the small projection layer trains.

## 8. Recursive inference

Multi-step forecasts are produced one step at a time (`recursive_forecast_graph2vec_perstep` in `graph2vec_assemble.py`, and the GCN/GAT equivalent in `inference_strategies.py`). After each prediction:

1. The prediction is appended to the rolling target window (in the metric's scale).
2. For Graph2Vec, the target's graph is rebuilt from that window, strictly before the forecast day, and re-embedded with `infer`. For GCN and GAT, the target node's features are patched.
3. The new graph is pushed onto a deque of `L` graphs.
4. Lag and rolling exogenous columns are recomputed from the prediction-augmented history, so the target's future true values never leak into its own inputs.

Neighbour products' observed values over the horizon are used by design (the panel / nowcasting assumption).

## 9. Experiment runners

The `graph2vec_gcn_gat_assmble_*.py` scripts build the graphs and datasets, train each variant against the baselines, and append one row per (product, seed, variant) to `<metric>.csv`. They import `LSTMBaseline.{dataset, lstm, lstm_train, lstm_inference}` and `MLP_Baseline.{train, inference, utils, mlp}` from `GNN/`.

| Script | Purpose |
|---|---|
| `..._parallelized_workers.py` | Main runner. Parallel by seed or product; `PARALLEL = False` runs in-process. Only this runner has the reviewer baselines and static variants. |
| `..._sequential.py` | Older sequential runner |
| `..._scalability_analysis_copy.py` | Scalability analysis |

Rows already present in `<metric>.csv` or `<metric>_w*.csv` are skipped on rerun (resume). `REDISTRIBUTE_STAGES` narrows the product list to the leftovers of an interrupted run: set it to `None` for a full run.

## 10. Reviewer baselines (`review_baselines.py`)

All of them use the hybrids' split, MinMax target scaling, validation-slice early stopping and 153-step recursive rollout. They write rows to the same CSV with blank graph columns. Toggle them with the `RUN_*` flags in the workers runner.

| `model_variant` | What it is | Question it answers |
|---|---|---|
| `seasonal_naive` | `y[t] = y[t-7]` from the last observed week | Floor for MAE and POCID |
| `dlinear` | DLinear (`GNN/DLinearBaseline`), univariate, moving-average kernel 7 | Modern linear forecaster |
| `patchtst` | PatchTST (`GNN/PatchTSTBaseline`), univariate, RevIN, patch 8 / stride 4 | Modern Transformer forecaster |
| `nbr_lstm`, `nbr_mlp` | The hybrids' graph-free heads plus the top-`NBR_TOP_K` neighbours' demand as plain input channels. Neighbours are ranked with the config metric on the training period only. | Does the gain come from graph topology, or just from neighbour data? |
| `nbr_lstm_snaive`, `nbr_mlp_snaive` | Same, but the neighbours' test-period demand is replaced by their own seasonal-naive forecast | Fully deployable version: no future neighbour demand |
| `static_gcn_*`, `static_gat_*` | Same encoders as the dynamic variants, on one ego-graph of the most persistent training-window neighbours, with mean edge weights. Its size is density-matched to the product's mean dynamic neighbourhood (`STATIC_GRAPH_TOP_K = None`). Node features still roll daily. | Is rebuilding the graph every day necessary? |

A fixed frequency cut-off is not used for the static graph because neighbourhoods turn over very fast. For product 278: about 12 neighbours per window, 571 distinct over the training period, and the most persistent appears in only 19% of windows.

The neighbour channel for day `t` carries the neighbours' value on day `t`, never on the predicted day. This matches what the per-step ego-graph built on `[t-W+1, t]` exposes, so `nbr_*` and the graph models see the same information.

## 11. DLinear / PatchTST hybrids (`gcn_models_modern.py`)

`{graph2vec,gcn,gat}_{dlinear,patchtst}` pair the same three graph encoders (layer for layer as in `gcn_models.py`) with DLinear or PatchTST in place of the LSTM / MLP. Each is compared with the `dlinear` / `patchtst` baseline.

- **How z enters.** The temporal input is `[y_t ‖ z_t]`. DLinear adds one linear map over the lookback per z channel. PatchTST patches z with the same windows as the target and adds its projection to each patch embedding. RevIN normalises the target only.
- **Zero-initialised covariate path.** At initialisation a hybrid outputs exactly what its baseline outputs. It moves away only if z lowers the training loss, so any difference from the baseline is attributable to the graph.
- **No exog.** Like their baselines, these hybrids read only the target column of `ts_seq`, so hybrid vs baseline differ by z alone.
- **Shared plumbing.** They use the LSTM path's dataset, `train_gcn_lstm_model` and recursive inference (`_data_head`). LR `LEARNING_RATE_MODERN`, dropout `DROPOUT_MODERN` (PatchTST and encoder), scheduler `USE_LR_SCHEDULER_MODERN`, all identical to the baselines.
- **No ablation run.** With z zeroed the hybrid *is* the baseline.
- **Static versions.** `static_gat_dlinear` etc. work too: add them to `MODEL_VARIANTS`.
