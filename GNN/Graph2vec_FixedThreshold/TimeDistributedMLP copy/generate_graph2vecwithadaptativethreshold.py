import os
import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from graph2vec import CustomGraph2Vec as Graph2Vec
from utils import neighbourhood_graph, compute_distances_1vsAll, compute_similarities_1vsAll
DISTANCE_METRICS = {
    'euclidean', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss', 'manhattan', 'twed', 'erp', 'stid'
}
SIMILARITY_METRICS = {'pearson', 'spearman', 'kendall'}


def infer_metric_type(metric, metric_type=None):
    if metric_type is not None:
        if metric_type not in {'distance', 'similarity'}:
            raise ValueError("metric_type must be either 'distance' or 'similarity'")
        return metric_type
    if metric in DISTANCE_METRICS:
        return 'distance'
    if metric in SIMILARITY_METRICS:
        return 'similarity'
    raise ValueError(
        f"Metric {metric} not supported. "
        f"Distance metrics: {sorted(DISTANCE_METRICS)}; "
        f"similarity metrics: {sorted(SIMILARITY_METRICS)}"
    )

def get_graph2vec_embeddings(graphs, dimensions=20, wl_iterations=2, workers=1, epochs=100, min_count=1, window=0, seed=42):
    """
    Generates Graph2Vec embeddings for a sequence of graphs.
    Returns the graph embeddings.
    """
    start_time = time.time()
    
    # Filter out empty graphs or those with no edges as Graph2Vec needs at least some structure
    valid_graphs = []
    valid_indices = []
    
    for i, G in enumerate(graphs):
        if G.number_of_nodes() > 0:
            valid_graphs.append(G)
            valid_indices.append(i)
            
    # Initialize and fit Graph2Vec
    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations, workers=workers, epochs=epochs, min_count=min_count, seed=seed)
    model.fit(valid_graphs)
    
    # Get embeddings
    emb = model.get_embedding()
    
    # Reconstruct full array keeping zeros for empty graphs
    graph_embeddings = np.zeros((len(graphs), dimensions))
    for idx, valid_idx in enumerate(valid_indices):
        graph_embeddings[valid_idx] = emb[idx]
        
    
    elapsed_time = time.time() - start_time
    
    return graph_embeddings, model # Graph2Vec does not produce node-level embeddings

def load_or_generate_embeddings(product_id, metric, window_size, step_size, threshold, percentile, 
            dimensions=20, wl_iterations=2, walk_length=10, num_walks=50, workers=1, epochs=100, min_count=1, window=0, seed=42, 
            use_residuals=False, model_type='ridge', enable_edges_within_star=True, enable_second_degree=False, run_id=None, 
        overwrite_embeddings=False, graphs=None, df=None, metric_type=None, cat_labels=None, train_end_idx=None, save_embeddings=True
            ):

    metric_type = infer_metric_type(metric, metric_type)

    prefix = "" if enable_edges_within_star else "star_"
    if enable_second_degree:
        prefix = "2nddegree_" + prefix
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    dir_label = f"pct{percentile}" if percentile is not None else f"th{threshold}"
    if use_residuals and percentile is not None:
        dir_label = f"top{percentile}pct"
        
    plot_dir_name = f"{prefix}{dir_label}" if prefix else dir_label
    
    if use_residuals:
        base_dir = os.path.join(curr_dir, '..', 'GraphAnalysis', 'DynamicGraphPkls', f'residuals_{model_type}', str(product_id), metric, str(window_size), str(step_size), plot_dir_name)
        emb_base_dir = os.path.join(curr_dir, 'embeddings', f'residuals_{model_type}', str(product_id), metric, str(window_size), str(step_size), plot_dir_name)
    else:
        base_dir = os.path.join(curr_dir, '..', 'GraphAnalysis', 'DynamicGraphPkls', str(product_id), metric, str(window_size), str(step_size), plot_dir_name)
        emb_base_dir = os.path.join(curr_dir, 'embeddings', str(product_id), metric, str(window_size), str(step_size), plot_dir_name)
        
    pkl_path = ""
    # Assuming the PKL name is the same as before if calculated_threshold_strategy is passed. 
    # Actually, in graph_construction, the PKL name is kept the exact same as original, just placed inside the strat folder!
    if threshold is not None:
        pkl_filename = f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_th{threshold}.pkl"
    if percentile is not None:
        if use_residuals:
            pkl_filename = f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_top{percentile}pct.pkl"
        else:
            pkl_filename = f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_pct{percentile}.pkl"
            
    pkl_path = os.path.join(base_dir, pkl_filename)
    emb_pkl_path = os.path.join(emb_base_dir, pkl_filename.replace('dynamic_graphs_', 'embeddings_').replace('.pkl', f'_dim{dimensions}_ep{epochs}_wl{wl_iterations}_seed{seed}{run_suffix}.pkl'))

    model_pkl_path = emb_pkl_path.replace('embeddings_', 'graph2vec_model_')
    lock_path = emb_pkl_path + ".lock"

    if os.path.exists(emb_pkl_path) and os.path.exists(model_pkl_path) and not overwrite_embeddings:
        with open(emb_pkl_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        graph_embeddings = embeddings_data['graph_embeddings']
        
        # Recover our fixed training threshold, deducing from pickled graphs if it's an older run
        saved_threshold = embeddings_data.get('fixed_global_threshold')
        if saved_threshold is None and threshold is None:
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as pf:
                    g_temp = pickle.load(pf)
                all_w = [d.get('weight', 0) for g in (g_temp[:train_end_idx] if train_end_idx else g_temp) for _,_,d in g.edges(data=True)]
                if all_w:
                    saved_threshold = np.max(all_w) if metric_type == 'distance' else np.min(all_w)
                else:
                    saved_threshold = float('inf') if metric_type == 'distance' else -float('inf')
            else:
                saved_threshold = 0.0
        elif saved_threshold is None:
            saved_threshold = threshold
        
        with open(model_pkl_path, 'rb') as f:
            graph2vec_model = pickle.load(f)
            
        csv_path = emb_pkl_path.replace('.pkl', '.csv')
        if not os.path.exists(csv_path):
            import pandas as pd
            df_embs = pd.DataFrame(graph_embeddings)
            df_embs.to_csv(csv_path, index=False)
            
        return graph_embeddings, graph2vec_model, csv_path, 0.0, 0.0, saved_threshold

    if os.path.exists(lock_path):
        return None, None, None, 0.0, 0.0, threshold if threshold is not None else 0.0

    build_graph_time = 0.0
    get_embedding_time = 0.0

    try:
        # Create a lock file to claim this configuration
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, 'w') as f: f.write(str(time.time()))

        build_graph_start = time.time()
        if graphs is None:
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        graphs = pickle.load(f)
                    
                    # Recover our fixed training threshold, deducing from pickled graphs
                    if threshold is None:
                        all_w = [d.get('weight', 0) for g in (graphs[:train_end_idx] if train_end_idx else graphs) for _,_,d in g.edges(data=True)]
                        if all_w:
                            fixed_global_threshold = np.max(all_w) if metric_type == 'distance' else np.min(all_w)
                        else:
                            fixed_global_threshold = float('inf') if metric_type == 'distance' else -float('inf')
                    else:
                        fixed_global_threshold = threshold
                    
                except (EOFError, pickle.UnpicklingError) as e:
                    graphs = None  # Force regeneration

            if graphs is None and df is not None:
                import sys
                utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GraphAnalysis'))
                if utils_path not in sys.path:
                    sys.path.append(utils_path)
                
                
                compute_func = compute_distances_1vsAll if metric_type == 'distance' else compute_similarities_1vsAll
                
                graphs, fixed_global_threshold = neighbourhood_graph(
                    product_id=product_id,
                    df=df,
                    metric=metric,
                    metric_type=metric_type,
                    window_size=window_size,
                    compute_func=compute_func,
                    threshold=threshold,
                    percentile=percentile,
                    step_size=step_size,
                    cat_labels=cat_labels,
                    plot_dir=None,
                    residuals=use_residuals,
                    enable_edges_within_star=enable_edges_within_star,
                    enable_second_degree=enable_second_degree,
                    train_end_idx=train_end_idx
                )
            elif graphs is None:
                raise FileNotFoundError(f"Cannot find graphs at {pkl_path} and no dataset (df) was provided to build them dynamically.")
        else:
            fixed_global_threshold = threshold # Default if provided in memory
        
        build_graph_time = time.time() - build_graph_start
        
        use_weights = False
        
        get_emb_start = time.time()
        graph_embeddings, graph2vec_model = get_graph2vec_embeddings(
            graphs, dimensions=dimensions, wl_iterations=wl_iterations, workers=workers, epochs=epochs, min_count=min_count, window=window, seed=seed
        )
        get_embedding_time = time.time() - get_emb_start
        
        if save_embeddings:
            # Save the embeddings to a pickle file
            os.makedirs(os.path.dirname(emb_pkl_path), exist_ok=True)
            with open(emb_pkl_path, 'wb') as f:
                pickle.dump({
                    'graph_embeddings': graph_embeddings,
                    'fixed_global_threshold': fixed_global_threshold
                }, f)
                
            # Save the trained model to a pickle file
            with open(model_pkl_path, 'wb') as f:
                pickle.dump(graph2vec_model, f)
                
            csv_path = emb_pkl_path.replace('.pkl', '.csv')
            if not os.path.exists(csv_path):
                import pandas as pd
                df_embs = pd.DataFrame(graph_embeddings)
                df_embs.to_csv(csv_path, index=False)
        else:
            csv_path = emb_pkl_path.replace('.pkl', '.csv')

        
    finally:
        # Always remove the lock file when done or if it crashes
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except Exception as e:
                pass
    
    return graph_embeddings, graph2vec_model, csv_path, build_graph_time, get_embedding_time, fixed_global_threshold


def analyze_embeddings(embeddings, output_csv_path=None):
    """
    Analyzes the generated embeddings to check for repetitions, missing values, and uniqueness.
    Optionally exports the metrics (including variance per dimension) to a CSV file.
    """
    import pandas as pd
    total_graphs = embeddings.shape[0]
    
    # Check for all-zero embeddings (usually corresponding to empty graphs)
    zero_rows = np.all(embeddings == 0, axis=1)
    num_zeros = np.sum(zero_rows)
    
    analysis_results = {
        'total_graphs': total_graphs,
        'zero_embeddings': num_zeros,
        'zero_embeddings_pct': (num_zeros / total_graphs) * 100 if total_graphs > 0 else 0
    }
    
    # Analyze non-zero embeddings
    non_zero_embs = embeddings[~zero_rows]
    if len(non_zero_embs) > 0:
        unique_embs, counts = np.unique(non_zero_embs, axis=0, return_counts=True)
        num_unique = len(unique_embs)
        repeats = len(non_zero_embs) - num_unique
        
        mean_norm = np.mean(np.linalg.norm(non_zero_embs, axis=1))
        total_variance = np.var(non_zero_embs)
        
        analysis_results.update({
            'non_zero_embeddings': len(non_zero_embs),
            'unique_embeddings': num_unique,
            'unique_embeddings_pct': (num_unique / len(non_zero_embs)) * 100,
            'repeats': repeats,
            'mean_norm': mean_norm,
            'total_variance': total_variance
        })
        
        # Measure variance per dimension
        var_per_dim = np.var(non_zero_embs, axis=0)
        for i, v in enumerate(var_per_dim):
            analysis_results[f'var_dim_{i}'] = v
    else:
        pass
        
    if output_csv_path:
        df_analysis = pd.DataFrame([analysis_results])
        df_analysis.to_csv(output_csv_path, index=False)

    return analysis_results


if __name__ == "__main__":
    # Example usage / Test block
    embs, model, csv_path, bt, et, thresh = load_or_generate_embeddings(
        product_id=907969,
        metric='cid',
        window_size=15,
        step_size=1,
        threshold=None,
        percentile=0.5,
        dimensions=20,
        enable_edges_within_star=False,
        overwrite_embeddings=False,
    )
    
    analyze_embeddings(embs, output_csv_path=csv_path.replace('.csv', '_analysis.csv'))
    
    