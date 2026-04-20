import os
import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from graph2vec import CustomGraph2Vec as Graph2Vec

def get_graph2vec_embeddings(graphs, dimensions=20, workers=1, epochs=100, min_count=1, window=0, seed=42):
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
            
    print(f"Generating Graph2Vec embeddings for {len(valid_graphs)} valid graphs out of {len(graphs)}...")
            
    # Initialize and fit Graph2Vec
    model = Graph2Vec(dimensions=dimensions, workers=workers, epochs=epochs, min_count=min_count, seed=seed)
    model.fit(valid_graphs)
    
    # Get embeddings
    emb = model.get_embedding()
    
    # Reconstruct full array keeping zeros for empty graphs
    graph_embeddings = np.zeros((len(graphs), dimensions))
    for idx, valid_idx in enumerate(valid_indices):
        graph_embeddings[valid_idx] = emb[idx]
        
    
    elapsed_time = time.time() - start_time
    print("Finished generating embeddings!")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    
    return graph_embeddings# Graph2Vec does not produce node-level embeddings


def load_or_generate_embeddings(product_id, metric, window_size, step_size, threshold, percentile, dimensions=20, walk_length=10, num_walks=50, workers=1, epochs=100, min_count=1, window=0, seed=42, use_residuals=False, model_type='ridge', enable_edges_within_star=True, run_id=None, overwrite_embeddings=False):
    """
    Loads cached embeddings if they exist. If not, loads the graphs from the expected directory, 
    generates embeddings using get_node2vec_embeddings, and saves them to a pickle file.
    """
    prefix = "" if enable_edges_within_star else "star_"
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if use_residuals:
        base_dir = os.path.join(curr_dir, '..', 'GraphAnalysis', 'DynamicGraphPkls', f'residuals_{model_type}', metric, str(window_size), str(step_size), str(product_id))
    else:
        base_dir = os.path.join(curr_dir, '..', 'GraphAnalysis', 'DynamicGraphPkls', metric, str(window_size), str(step_size), str(product_id))
        
    pkl_path = ""
    if threshold is not None:
        pkl_path = os.path.join(base_dir, f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_th{threshold}.pkl")
        emb_pkl_path = os.path.join(base_dir, f"{prefix}embeddings_{metric}_Window{window_size}_Step{step_size}_th{threshold}{run_suffix}.pkl")
    if percentile is not None:
        if use_residuals:
            pkl_path = os.path.join(base_dir, f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_top{percentile}pct.pkl")
            emb_pkl_path = os.path.join(base_dir, f"{prefix}embeddings_{metric}_Window{window_size}_Step{step_size}_top{percentile}pct{run_suffix}.pkl")
        else:
            pkl_path = os.path.join(base_dir, f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_pct{percentile}.pkl")
            emb_pkl_path = os.path.join(base_dir, f"{prefix}embeddings_{metric}_Window{window_size}_Step{step_size}_pct{percentile}{run_suffix}.pkl")

    if os.path.exists(emb_pkl_path) and not overwrite_embeddings:
        print(f"Embeddings already exist. Loading from {emb_pkl_path}...")
        with open(emb_pkl_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        graph_embeddings = embeddings_data['graph_embeddings']
        print("Successfully loaded embeddings!")
    else:
        print(f"Loading graphs from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
            
        print(f"Successfully loaded {len(graphs)} graphs.")
        
        use_weights = False
        
        graph_embeddings= get_graph2vec_embeddings(
            graphs, dimensions=dimensions, workers=workers, epochs=epochs, min_count=min_count, window=window, seed=seed
        )
        
        # Save the embeddings to a pickle file
        print(f"Saving embeddings to {emb_pkl_path}...")
        os.makedirs(os.path.dirname(emb_pkl_path), exist_ok=True)
        with open(emb_pkl_path, 'wb') as f:
            pickle.dump({
                'graph_embeddings': graph_embeddings
            }, f)
            
        print("Embeddings saved successfully.")

    csv_path = emb_pkl_path.replace('.pkl', '.csv')
    if not os.path.exists(csv_path):
        import pandas as pd
        print(f"Saving graph embeddings to {csv_path}...")
        df_embs = pd.DataFrame(graph_embeddings)
        df_embs.to_csv(csv_path, index=False)

    print(f"Graph Embeddings Shape: {graph_embeddings.shape}")
    
    
    return graph_embeddings, csv_path


def analyze_embeddings(embeddings, output_csv_path=None):
    """
    Analyzes the generated embeddings to check for repetitions, missing values, and uniqueness.
    Optionally exports the metrics (including variance per dimension) to a CSV file.
    """
    import pandas as pd
    print("\n--- Embedding Analysis ---")
    total_graphs = embeddings.shape[0]
    
    # Check for all-zero embeddings (usually corresponding to empty graphs)
    zero_rows = np.all(embeddings == 0, axis=1)
    num_zeros = np.sum(zero_rows)
    
    print(f"Total embeddings: {total_graphs}")
    print(f"All-zero embeddings (empty graphs): {num_zeros} ({(num_zeros/total_graphs)*100:.2f}%)")
    
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
        
        print(f"Non-zero embeddings: {len(non_zero_embs)}")
        print(f"Unique non-zero embeddings: {num_unique} ({(num_unique/len(non_zero_embs))*100:.2f}%)")
        print(f"Duplicate/Repeated non-zero embeddings: {repeats}")
        
        if repeats > 0:
            print("Note: Repeated embeddings typically indicate that the graph structure remained exactly identical across those time steps.")
            
        mean_norm = np.mean(np.linalg.norm(non_zero_embs, axis=1))
        total_variance = np.var(non_zero_embs)
        
        print(f"Mean embedding norm: {mean_norm:.4f}")
        print(f"Embedding variance (across all dimensions): {total_variance:.4f}")
        
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
        print("All embeddings were zero, skipping uniqueness analysis.")
        
    print("--------------------------\n")
    
    if output_csv_path:
        df_analysis = pd.DataFrame([analysis_results])
        df_analysis.to_csv(output_csv_path, index=False)
        print(f"Analysis saved to {output_csv_path}")

    return analysis_results


if __name__ == "__main__":
    # Example usage / Test block
    embs,csv_path = load_or_generate_embeddings(
        product_id=907969,
        metric='cid',
        window_size=15,
        step_size=1,
        threshold=None,
        percentile=0.5,
        enable_edges_within_star=False,
        overwrite_embeddings=True,
    )
    
    analyze_embeddings(embs, output_csv_path=csv_path.replace('.csv', '_analysis.csv'))
    
    