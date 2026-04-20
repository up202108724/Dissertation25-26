# Functional Specification & Technical Prompt  
## GraphSAGE–LSTM Spatial-Temporal Architecture for Multi-Step Recursive Forecasting on Dynamic Graphs

---

## 1. System Purpose

Design, train, and deploy a spatial-temporal machine learning system that predicts the future daily sales of a **target product (SKU)** over a configurable multi-step horizon. The distinguishing capability of the system is the **dynamic graph structure**: at each prediction step, a directed similarity graph is (re-)constructed from the current sliding time window (which may contain predicted values in the inference phase), and an inductive **GraphSAGE** model converts this graph into a compact embedding vector. That vector is injected as an **exogenous contextual feature** into the per-day cell of an **LSTM** recurrent network which performs the final sales regression.

---

## 2. Terminology

| Symbol | Meaning |
|---|---|
| $i^*$ | The target SKU being forecasted |
| $\mathcal{N}$ | Set of all SKUs in the catalogue; $i^* \in \mathcal{N}$ |
| $W$ | Graph sliding window size (e.g., 7 days) |
| $L$ | LSTM input sequence length (lookback window) |
| $H$ | Forecast horizon (number of future days to predict) |
| $T_{last}$ | Index of the last day with known ground-truth sales |
| $G_t$ | Graph snapshot built from ground-truth data within $[t-W,\, t-1]$ |
| $G'_{T+k}$ | Predicted graph snapshot built from a window that contains predicted values |
| $\mathbf{e}_t \in \mathbb{R}^d$ | GraphSAGE embedding of node $i^*$ inside graph $G_t$ |
| $\mathbf{x}_t$ | Raw feature vector for day $t$: $[\text{sales}_{i^*,t}, \text{exog}_t]$ |

---

## 3. Primary Component Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                       INFERENCE / TRAINING LOOP                       │
│                                                                       │
│   Day t:   sliding_window ──► Graph Builder ──► G_t                  │
│                                                    │                  │
│                                              GraphSAGE (frozen/fit)   │
│                                                    │                  │
│                                              e_t ∈ ℝ^d               │
│                                                    │                  │
│   LSTM input at step t:  [ sales_{i*,t}  |  exog_t  |  e_t ]        │
│                                                    │                  │
│                                              LSTM ──► fc ──► ŷ_{t+1} │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.1 GraphSAGE Feature Extractor

**Role**: Inductive neighbour-aggregation GNN.  
**Why GraphSAGE**: Unlike transductive methods (GCN, GAT on a fixed adjacency), GraphSAGE can operate on **any unseen graph topology** without retraining, because it learns a neighbourhood aggregation function rather than node-specific weights. This is essential for inference, where the graph is reconstructed each day from predicted data.

**Architecture** (maps to `GraphSAGE` class in `GraphSAGE.py`):

```
Input: raw_features  ∈ ℝ^{N×F_in}   (per-node time-series features, e.g. sales in [t-W, t-1])
       adj_lists: Dict[int, Set[int]] (neighbour sets derived from the similarity graph)

Layer 1 – SageLayer(input_size=F_in,  out_size=sage_hidden)
Layer 2 – SageLayer(input_size=sage_hidden, out_size=sage_out=d)

Output: node_embeddings ∈ ℝ^{N×d}
Target embedding: node_embeddings[i*_idx] ∈ ℝ^d
```

**Aggregation**: Mean aggregation (default `agg_func='MEAN'`). Max-pooling is available as an alternative.  
**Weight matrices**: Xavier-uniform initialised via `SageLayer.init_params()`.  
**GCN mode**: When `gcn=True`, self-features are not concatenated but only neighbour aggregates are used (equivalent to vanilla GCN). Default `gcn=False` (standard GraphSAGE: concatenate self + aggregated).

### 3.2 LSTM Temporal Model

**Role**: Sequence-to-one regression over a lookback window of length $L$.  
**Architecture** (maps to the pattern in `graph2vecdataset.py` + `graphsagetrain.py`):

```
Input per step t: [ sales_{i*,t}  |  exog_t  |  e_t ]
                   (1-dim)          (E-dim)     (d-dim)
Total input dim = 1 + E + d

LSTM(input_size = 1+E+d, hidden_size = H_lstm, num_layers = num_layers, batch_first=True)
→ take output at final time step: h_L ∈ ℝ^{H_lstm}
→ Dropout(p)
→ Linear(H_lstm, 1)  →  ŷ  (next-day sales, scaled)
```

**Training signal**: MSE (or MAE, configurable) between $\hat{y}$ and the true next-day sales.

---

## 4. Graph Construction Protocol

### 4.1 Node Feature Matrix

For a graph snapshot $G_t$, every node $j \in \mathcal{N}$ is represented by its sales time series within the preceding window:

$$\mathbf{f}_{j,t} = \text{sales}_{j,\, t-W : t-1} \in \mathbb{R}^{W}$$

This is the raw-feature array passed as `raw_features` to GraphSAGE.

### 4.2 Edge Construction (Similarity Metric)

Edges are built by computing a pairwise **similarity score** between the target product $i^*$ and every candidate peripheral SKU $j \neq i^*$:

$$\text{sim}(j, i^*) = f_{\text{metric}}\!\left(\mathbf{f}_{j,t},\, \mathbf{f}_{i^*,t}\right)$$

Supported metrics:

| Metric | Notes |
|---|---|
| **DTW** (Dynamic Time Warping) | Captures shape similarity even under phase shifts; preferred for seasonal product sales |
| **Pearson Correlation** | Fast; sensitive to linear co-movement |
| **Euclidean Distance** (inverted) | Simplest baseline |

An undirected edge $(j, i^*)$ is added if $\text{sim}(j, i^*)$ exceeds the $p$-th similarity percentile over all candidate pairs (e.g., `p = 0.10` → top 10% most similar SKUs become neighbours).

**Adjacency list format** (compatible with `GraphSAGE.adj_lists`):

```python
adj_lists: Dict[int, Set[int]]  # node_idx → set of neighbour node_indices
```

### 4.3 Graph Snapshot Timing Convention

A graph snapshot $G_t$ is associated with day $t$ but is built **strictly from days $[t-W,\; t-1]$**, i.e. it uses only information observable *before* day $t$. This avoids future data leakage.

---

## 5. Offline Training Workflow

### 5.1 End-to-End Pipeline Summary

```
Historical data (train + val)
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │   PHASE A: Graph Sequence Construction        │
  │                                              │
  │   For t = W  to  T_last:                    │
  │     1. Extract sales window [t-W, t-1]       │
  │     2. Compute pairwise similarities          │
  │     3. Apply percentile edge threshold        │
  │     4. Store G_t (as adj_lists + raw_features)│
  └──────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │   PHASE B: GraphSAGE Pre-training (Optional) │
  │                                              │
  │   Train GraphSAGE unsupervised on the        │
  │   corpus of graphs {G_W, …, G_{T_last}}     │
  │   using the UnsupervisedLoss (random walks + │
  │   negative sampling) defined in GraphSAGE.py │
  │   → Optimise neighbourhood-preserving        │
  │     embedding quality.                       │
  │   → Save checkpoint: graphsage_pretrained.pt │
  └──────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │   PHASE C: Embedding Cache Generation        │
  │                                              │
  │   Load pretrained GraphSAGE (frozen).        │
  │   For t = W  to  T_last:                    │
  │     Forward-pass G_t through GraphSAGE.      │
  │     Extract e_t = node_embeddings[i*_idx]    │
  │     ∈ ℝ^d.                                   │
  │   Save: cached_embeddings[t] = e_t           │
  │   Shape of cache: (T_last - W + 1, d)        │
  └──────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────┐
  │   PHASE D: LSTM Training with Exogenous      │
  │            Graph Embedding                   │
  │                                              │
  │   Dataset (TimeSeriesDataset /               │
  │   SingleItemGraphDataset):                   │
  │     For each training sample at position t:  │
  │       x_seq[t] = [sales_{i*,t}|exog_t|e_t]  │
  │       y = sales_{i*, t+1}                    │
  │                                              │
  │   Train LSTM end-to-end to minimise          │
  │   MSE(ŷ, y) with early stopping on val set.  │
  └──────────────────────────────────────────────┘
```

### 5.2 Dataset Construction Detail

The `TimeSeriesDataset` class (`graph2vecdataset.py`) already supports this pattern via the `embeddings` parameter. The zero-padding logic ensures correct temporal alignment:

```
cached_embeddings[t]  →  represents graph G_t  →  built from window [t-W, t-1]
                      →  must be used as exog input for the LSTM step that predicts sales at t+1
```

The class zero-pads the first $W$ days so that `embeddings[t] = e_t` is aligned with LSTM input at step $t$.

**Feature stacking at each time step** (column order must be consistent across training and inference):

```
[target_sales (1-dim)] | [exogenous features (E-dim)] | [graph embedding (d-dim)]
```

### 5.3 GraphSAGE Pre-training (Unsupervised)

**Objective**: Learn neighbourhood-preserving node embeddings via the loss defined in `UnsupervisedLoss` (GraphSAGE.py):

$$\mathcal{L} = -\log \sigma(\mathbf{z}_u \cdot \mathbf{z}_v) - Q \cdot \mathbb{E}_{v_n \sim P_n} \left[\log \sigma(-\mathbf{z}_u \cdot \mathbf{z}_{v_n})\right]$$

where $(u, v)$ is a positive pair from a short random walk and $v_n$ is a far-away negative sample.

**Training graphs**: All graph snapshots $\{G_t\}_{t=W}^{T_{last}}$ from the training period.  
**Input features**: Normalised sales window $\mathbf{f}_{j,t} \in \mathbb{R}^W$ per node.  
**Key hyperparameters**:
- `num_layers`: 2 (standard two-hop neighbourhood aggregation)
- `input_size`: $W$ (window size = raw feature dimension)
- `out_size`: $d$ (embedding dimension, e.g. 32 or 64)
- `agg_func`: `'MEAN'` or `'MAX'`
- `num_sample`: 10 (neighbourhood sample size per hop, controls receptive field)

---

## 6. Recursive Inference Workflow

This section defines the **precise daily loop** for multi-step forecasting beyond $T_{last}$.

### 6.1 Initialisation (Before Loop)

```python
# Active sliding window for the TARGET product and ALL peripheral SKUs
# Shape: (W, |N|) - each row is one day, each column is one SKU
active_window = ground_truth_sales[T_last - W + 1 : T_last + 1, :]   # [T-W+1 .. T_last]

# Active LSTM input sequence (last L observations)
current_seq   = scaled_sales[T_last - L + 1 : T_last + 1]    # shape (L,)
current_exog  = scaled_exog [T_last - L + 1 : T_last + 1]    # shape (L, E)
current_emb   = cached_embeddings[T_last - L + 1 : T_last + 1]  # shape (L, d)

predictions = []
```

### 6.2 Per-Step Loop (for k = 1, …, H)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP k: Predict sales for day T_last + k                           │
│                                                                     │
│  1. PREDICT                                                         │
│     x = column_stack([current_seq, current_exog, current_emb])     │
│         shape (L, 1 + E + d)                                       │
│     ŷ_k = LSTM(x)  → scalar (scaled)                               │
│     predictions.append(ŷ_k)                                        │
│                                                                     │
│  2. RECURSIVE WINDOW UPDATE                                         │
│     Insert ŷ_k (inverse-transformed to original scale) as the      │
│     sales of i* on day T_last + k into active_window.              │
│     Displace the oldest row (day T_last - W + k):                  │
│       active_window = roll left by 1 row, append new row           │
│     New window covers days [T_last - W + k + 1, T_last + k]        │
│     For peripheral SKUs at day T_last + k, use:                    │
│       - Known future exogenous data if available, OR               │
│       - Last known value (frozen peripheral assumption), OR         │
│       - A separately trained peripheral forecast (advanced)        │
│                                                                     │
│  3. BUILD PREDICTED GRAPH G'_{T+k}                                 │
│     Using active_window (which now contains predicted value for i*):│
│     a. Extract sales window for every SKU j:                       │
│          f_{j} = active_window[:, j]   ∈ ℝ^W                      │
│     b. Compute sim(j, i*) for all j ≠ i* using the SAME metric     │
│          and threshold as used during training.                     │
│     c. Build adj_lists and raw_features from this window.          │
│     G'_{T+k}.raw_features shape: (|N|, W)                         │
│     G'_{T+k}.adj_lists: Dict[int, Set[int]]                        │
│                                                                     │
│  4. EMBED PREDICTED GRAPH (ON-THE-FLY INDUCTIVE INFERENCE)         │
│     Load frozen GraphSAGE checkpoint.                              │
│     with torch.no_grad():                                          │
│       node_embs = graphsage(G'_{T+k}.raw_features,                │
│                              G'_{T+k}.adj_lists)                   │
│     e'_{T+k} = node_embs[i*_idx]   ∈ ℝ^d                         │
│                                                                     │
│  5. SLIDE LSTM INPUT WINDOW                                         │
│     current_seq  = slide right: drop oldest, append ŷ_k (scaled)  │
│     current_exog = slide right: drop oldest, append next exog       │
│     current_emb  = slide right: drop oldest, append e'_{T+k}       │
│                                                                     │
│  ──► Go to STEP k+1                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Peripheral SKU Handling During Inference

The **peripheral SKUs** (all $j \neq i^*$) populate the other columns of `active_window`. Three strategies exist for filling their future rows (ranked by realism):

| Strategy | Description | Trade-off |
|---|---|---|
| **Frozen Peripheral** | Repeat the last known sales value for all future days | No extra model needed; conservative; may distort similarity for long horizons |
| **Known Future** | If exogenous/catalogue data provides future peripheral values (e.g., promotional flags) | Ideal but rarely available for all SKUs |
| **Predicted Peripheral** | Maintain a separate trained forecaster per peripheral SKU | Highest accuracy; substantial engineering overhead |

**Default recommendation for dissertation scope**: Frozen Peripheral. This is conservative and avoids a circular inference dependency that requires co-forecasting all SKUs.

### 6.4 Critical Design Constraint: Metric Consistency

The **identical similarity metric, distance function, and edge threshold** used to build $G_t$ during training must be applied unchanged when building $G'_{T+k}$ during inference. Any deviation will cause a distribution shift between the training-time and inference-time embeddings, degrading the quality of $\mathbf{e}'_{T+k}$ relative to the trained LSTM expectation.

This means:
- The same `percentile` value (e.g., 0.10) must be re-evaluated **empirically per window** at inference time. It is not a global constant; it is the $p$-th percentile of the similarity scores computed from the current window.
- The same feature pre-processing (MinMax or Standard scaling of the window) must be applied before computing similarities.

---

## 7. Data Leakage Prevention

| Phase | Potential Leak | Mitigation |
|---|---|---|
| Graph construction | Using sales at day $t$ to build $G_t$ | $G_t$ uses window $[t-W, t-1]$ strictly; day $t$ excluded |
| Embedding cache | Using test-period graphs | Only compute cached embeddings for train+val days |
| LSTM dataset | Exog at day $t+1$ known at day $t$ | Exog is shifted by +1 in `TimeSeriesDataset` (line: `exog_data[idx+1:idx+seq_length+1]`) |
| Validation split | Graphs built with validation targets | Graph window must not overlap with the first validation day |

---

## 8. Module Interface Contracts

### 8.1 `GraphBuilder`

```python
def build_graph_snapshot(
    sales_window: np.ndarray,     # shape (W, |N|) — columns ordered by SKU index
    target_sku_idx: int,          # column index of i* in sales_window
    metric: str,                  # 'dtw' | 'correlation' | 'euclidean'
    percentile: float,            # edge inclusion threshold (e.g. 0.10)
) -> Tuple[Dict[int, Set[int]], np.ndarray]:
    """
    Returns:
        adj_lists     — neighbour sets for GraphSAGE
        raw_features  — shape (|N|, W), each row = sales_window[:, j].T
    """
```

### 8.2 `GraphSAGEEmbedder`

```python
def embed_graph(
    graphsage: GraphSAGE,          # pretrained, frozen model
    raw_features: np.ndarray,      # shape (|N|, W)
    adj_lists: Dict[int, Set[int]],
    target_idx: int,
) -> np.ndarray:                   # shape (d,)
    """
    Inductive, gradient-free forward pass.
    Returns the embedding vector for the target node only.
    """
```

### 8.3 `RecursiveForecaster`

```python
def recursive_forecast(
    lstm_model: nn.Module,         # trained LSTM (frozen)
    graphsage: GraphSAGE,          # pretrained GraphSAGE (frozen)
    active_sales_window: np.ndarray,   # shape (W, |N|), ground truth up to T_last
    initial_lstm_seq: np.ndarray,      # shape (L, 1 + E + d), last L days
    horizon: int,                      # H — number of future days
    target_sku_idx: int,
    metric: str,
    percentile: float,
    scaler,                            # sklearn-compatible inverse_transform
    exog_future: Optional[np.ndarray], # shape (H, E) or None
    peripheral_strategy: str,          # 'frozen' | 'known' | 'predicted'
    device: torch.device,
) -> np.ndarray:                   # shape (H,), unscaled predictions
```

---

## 9. Hyperparameter Registry

| Group | Parameter | Suggested Range | Notes |
|---|---|---|---|
| Graph | `window_size` W | 7, 14, 21 | Larger = more stable similarity, less responsive |
| Graph | `metric` | `'dtw'`, `'correlation'` | DTW preferred; correlation is ~10× faster |
| Graph | `percentile` | 0.05 – 0.25 | Top 5–25% similar SKUs become edges |
| GraphSAGE | `num_layers` | 1, 2 | 2-hop captures indirect influence |
| GraphSAGE | `out_size` (d) | 16, 32, 64 | Larger = richer spatial context but slower inference |
| GraphSAGE | `num_sample` | 5, 10 | Neighbourhood sub-sampling; trade-off speed vs. quality |
| GraphSAGE | `agg_func` | `'MEAN'`, `'MAX'` | MEAN is typically more stable |
| LSTM | `seq_length` (L) | 14 – 60 | Lookback in days |
| LSTM | `lstm_hidden` | 32 – 128 | |
| LSTM | `num_layers` | 1, 2 | Multi-layer needs dropout to avoid overfitting |
| LSTM | `dropout` | 0.1 – 0.3 | |
| Training | `lr` | 1e-3, 5e-4 | Adam; cosine LR schedule optional |
| Training | `patience` | 10 – 20 | Early stopping on scaled MAE |

---

## 10. File Layout (GraphSAGE Subfolder)

```
GraphSAGE/
├── GraphSAGE.py              # SageLayer, GraphSAGE, UnsupervisedLoss, Classification
├── graphsagetrain.py         # train_model() — pure LSTM training loop
├── graph_sageinference.py    # graphsage_inference() — recursive forecasting loop
├── graph2vecdataset.py       # TimeSeriesDataset — handles embedding alignment & padding
├── graph2vecloader.py        # generate_graph2vec_embeddings() — offline embedding cache
├── lstm.py                   # GAT_LSTM stub (to be refactored → GraphSAGE_LSTM)
├── SPEC_GraphSAGE_LSTM_Recursive.md   # this specification
└── best_models/              # saved checkpoints
```

**Refactoring note for `lstm.py`**:  
The current `lstm.py` imports `GATFeatureExtractor` from `gat`. For this architecture, replace the spatial extractor with the `GraphSAGE` model (`GraphSAGE.py`). The `forward()` interface changes fundamentally: instead of accepting a list of PyTorch Geometric `Batch` objects per timestep, the LSTM receives a pre-stacked tensor `[batch, L, 1+E+d]` where the graph embedding is already concatenated as a column of features (embedding-as-exogenous pattern). This eliminates the need for an end-to-end differentiable graph encoder and relies on the frozen, inductively-pre-trained GraphSAGE.

---

## 11. Comparison with Current Codebase Patterns

| Pattern | Current Code | This Specification |
|---|---|---|
| Graph representation | PyTorch Geometric `Data`/`Batch` objects (GAT path) | `adj_lists` dict + `raw_features` numpy array (GraphSAGE-native) |
| Spatial encoder | `GATFeatureExtractor` (transductive, needs re-training for new graphs) | `GraphSAGE` (inductive — runs on any new graph at inference) |
| Embedding injection | End-to-end differentiable per-timestep spatial pass inside LSTM forward | Pre-cached embedding as extra input feature (`graph2vecdataset.py` pattern) |
| Inference graph update | `freeze_graph=True/False` flag (binary) | Full dynamic graph reconstruction + on-the-fly inductive embedding per step |
| Peripheral SKU strategy | Not modelled | Explicit frozen / known / predicted strategy contract |
| Dataset | `SingleItemGraphDataset` (graph-object-aware) or `TimeSeriesDataset` (embedding-column-aware) | `TimeSeriesDataset` with `embeddings` param (simpler; no PyG dependency at LSTM training time) |

---

## 12. Execution Order Summary

```
1.  Pre-process historical sales + exogenous features.
2.  [Train/Val/Test split — no leakage from test into graphs or scalers.]
3.  Build graph snapshot sequence {G_t} for t = W..T_train_end.
4.  (Optional) Pre-train GraphSAGE unsupervised on {G_t}.
5.  Cache graph embeddings {e_t} for t = W..T_val_end using frozen GraphSAGE.
6.  Align embedding cache with TimeSeriesDataset (zero-pad first W positions).
7.  Train LSTM with embedded features. Save best checkpoint.
8.  [At inference time:]
9.  Initialise active_window and current_seq from last W / last L known days.
10. For k = 1..H:
    a. Forward LSTM → ŷ_k
    b. Update active_window (insert ŷ_k for i*, fill peripherals per strategy)
    c. Reconstruct G'_{T+k} applying same metric + threshold
    d. Inductive GraphSAGE forward → e'_{T+k}
    e. Slide current_seq, current_exog, current_emb windows
11. Inverse-transform all ŷ_k. Return forecast array shape (H,).
```
