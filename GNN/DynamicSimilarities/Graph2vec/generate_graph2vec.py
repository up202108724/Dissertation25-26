import os
import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from graph2vec import CustomGraph2Vec as Graph2Vec

def get_graph2vec_embeddings(graphs, dimensions=64, workers=4):
    """
    Generates Graph2Vec embeddings for a sequence of graphs.
    Returns the graph embeddings. Target node embeddings will be zeros as Graph2Vec embeds the whole graph.
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
    model = Graph2Vec(dimensions=dimensions, workers=workers)
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


def load_or_generate_embeddings(product_id, metric, window_size, step_size, threshold, percentile, dimensions=64, walk_length=10, num_walks=50, workers=4, use_residuals=False, model_type='ridge'):
    """
    Loads cached embeddings if they exist. If not, loads the graphs from the expected directory, 
    generates embeddings using get_node2vec_embeddings, and saves them to a pickle file.
    """
    if use_residuals:
        base_dir = f"../GraphAnalysis/DynamicGraphPkls/residuals_{model_type}/{metric}/{product_id}/{window_size}/{step_size}"
    else:
        base_dir = f"../GraphAnalysis/DynamicGraphPkls/{metric}/{product_id}/{window_size}/{step_size}"
    pkl_path= ""
    if threshold is not None:
        pkl_path = f"{base_dir}/dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_th{threshold}.pkl"
        emb_pkl_path = f"{base_dir}/embeddings_{metric}_Window{window_size}_Step{step_size}_th{threshold}.pkl"
    if percentile is not None:
        if use_residuals:
            pkl_path = f"{base_dir}/dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_top{percentile}pct.pkl"
            emb_pkl_path = f"{base_dir}/embeddings_{metric}_Window{window_size}_Step{step_size}_top{percentile}pct.pkl"
        else:
            pkl_path = f"{base_dir}/dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_pct{percentile}.pkl"
            emb_pkl_path = f"{base_dir}/embeddings_{metric}_Window{window_size}_Step{step_size}_pct{percentile}.pkl"

    if os.path.exists(emb_pkl_path):
        print(f"Embeddings already exist. Loading from {emb_pkl_path}...")
        with open(emb_pkl_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        graph_embeddings = embeddings_data['graph_embeddings']
        target_node_embeddings = embeddings_data['target_node_embeddings']
        print("Successfully loaded embeddings!")
    else:
        print(f"Loading graphs from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
            
        print(f"Successfully loaded {len(graphs)} graphs.")
        
        # Use Graph2Vec instead of Node2Vec to generate entire graph embeddings.
        graph_embeddings, target_node_embeddings = get_graph2vec_embeddings(
            graphs, dimensions, workers
        )
        
        # Save the embeddings to a pickle file
        print(f"Saving embeddings to {emb_pkl_path}...")
        os.makedirs(os.path.dirname(emb_pkl_path), exist_ok=True)
        with open(emb_pkl_path, 'wb') as f:
            pickle.dump({
                'graph_embeddings': graph_embeddings,
                'target_node_embeddings': target_node_embeddings
            }, f)
        print("Embeddings saved successfully.")

    print(f"Graph Embeddings Shape: {graph_embeddings.shape}")
    print(f"Target Node Embeddings Shape: {target_node_embeddings.shape}")
    
    return graph_embeddings, target_node_embeddings

if __name__ == "__main__":
    # Example usage / Test block
    load_or_generate_embeddings(
        product_id=907969,
        metric='spearman',
        window_size=15,
        step_size=1,
        threshold=0.8
    )
    
    