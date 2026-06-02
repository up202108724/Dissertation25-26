import torch
import numpy as np
from typing import Optional
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
import networkx as nx
import pandas as pd
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


def recursive_inference(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,  
    future_exog: np.ndarray,     
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- Graph2Vec Parameters ---
    graph2vec_model=None,
    graph_window_size: int = 15,
    recent_embeddings: Optional[np.ndarray] = None,  # (L, emb_dim)
    df_wide: pd.DataFrame = None,
    cat_labels: dict = None,
    target_id: int = None,
    metric: str = 'cid',
    fixed_threshold: float = 0.5,
    enable_edges_within_star: bool = False,
    enable_second_degree: bool = False,
    past_dates: list = None,
    future_dates: list = None,
    return_graphs: bool = False,
) -> np.ndarray:
    """
    Recursively forecasts `horizon` steps using a 1-step-ahead model.
    Uses known `future_exog` for the next step's input.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    horizon = len(future_exog)
    
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    # Identificar índices exógenos de forma segura
    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Escalar o histórico recente (apenas o target, assumindo exogs já escaladas)
    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )
    
    if graph2vec_model is not None and recent_embeddings is not None:
        input_window = np.column_stack([current_x_scaled, recent_embeddings])
        num_base_features = C_in
        all_dates = list(past_dates) + list(future_dates)
        all_unscaled_targets = list(recent_history[:, target_channel])
    else:
        input_window = current_x_scaled.copy()
        
    preds_scaled = []
    preds_unscaled = []
    model = model.to(device).eval()

    # Alinhar janela inicial com a mesma lógica do make_windows
    # X[t] = target[t], exog[t+1], emb[t]
    if len(exog_indices) > 0:
        # Fazer shift das exógenas
        input_window[:-1, exog_indices] = input_window[1:, exog_indices]
        # Inserir exógena do dia que vamos prever no último step
        input_window[-1, exog_indices] = future_exog[0]
    
    gathered_inference_graphs = []
    
    with torch.no_grad(): # BOA PRÁTICA: Desligar gradientes na inferência
        for i in range(horizon):

            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device) # (1, L, C_in)
            
            y_pred = model(x_tensor)
            val_pred = y_pred.item()
            preds_scaled.append(val_pred)
            
            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # --- Shift da Janela ---
            input_window = np.roll(input_window, -1, axis=0)
            
            # 1. O novo target no último step é a previsão que acabámos de fazer
            input_window[-1, target_channel] = val_pred
            
            # 2. Inserir as exógenas do PRÓXIMO dia no último step
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]
                
            # 3. Inferir embedding para o dia que acabámos de prever (se graph2vec ativo)
            if graph2vec_model is not None:
                all_unscaled_targets.append(unscaled_val)
                # O índice do dia t (o dia que acabámos de prever) em all_dates é:
                current_t_idx = len(past_dates) + i
                
                # As datas da janela terminam em t
                window_dates = all_dates[current_t_idx - graph_window_size + 1 : current_t_idx + 1]
                target_preds = all_unscaled_targets[-graph_window_size:]
                
                G = build_dynamic_graph_with_calculated_threshold(
                    target_id=target_id,
                    target_preds=target_preds,
                    df_wide=df_wide,
                    cat_labels=cat_labels,
                    date_cols=window_dates,
                    metric=metric,
                    fixed_threshold=fixed_threshold,
                    enable_edges_within_star=enable_edges_within_star,
                    enable_second_degree=enable_second_degree
                )
                
                gathered_inference_graphs.append(G)
                
                new_emb = graph2vec_model.infer([G])[0]
                input_window[-1, num_base_features:] = new_emb
            
    # Reshape para fazer inverse_transform
    # Em vez de inverter novamente, podemos simplesmente devolver os preds_unscaled que já acumulámos
    if return_graphs:
        return np.array(preds_unscaled).flatten(), gathered_inference_graphs
    return np.array(preds_unscaled).flatten()