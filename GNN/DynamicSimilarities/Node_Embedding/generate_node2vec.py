import os
import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from node2vec import Node2Vec

def get_node2vec_embeddings(graphs, product_id, dimensions=64, walk_length=10, num_walks=50, workers=4):
    """
    Generates Node2Vec embeddings for a sequence of graphs.
    Returns the average graph embeddings and the target node embeddings.
    """
    graph_embeddings = []
    target_node_embeddings = []
    
    print("Generating node2vec embeddings for each graph in the sequence...")
    start_time = time.time()

    # Using tqdm to show a progress bar with ETA
    for G in tqdm(graphs, desc="Node2Vec Progress"):
        # Skip empty graphs or graphs with no edges
        if G.number_of_edges() == 0:
            graph_embeddings.append(np.zeros(dimensions))
            target_node_embeddings.append(np.zeros(dimensions))
            continue
            
        # Initialize Node2Vec
        # quiet=True prevents flooding the console with progress bars for every graph
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
        
        # Fit model (gensim under the hood)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        
        # 1. Graph Embedding: Average of all node embeddings in this temporal graph
        node_vecs = [model.wv[str(n)] for n in G.nodes() if str(n) in model.wv]
        if len(node_vecs) > 0:
            graph_emb = np.mean(node_vecs, axis=0)
        else:
            graph_emb = np.zeros(dimensions)
        graph_embeddings.append(graph_emb)
        
        # 2. Target Node Embedding: Embedding of the specific product ID
        if str(product_id) in model.wv:
            target_emb = model.wv[str(product_id)]
        else:
            target_emb = np.zeros(dimensions)
        target_node_embeddings.append(target_emb)
            
    graph_embeddings = np.array(graph_embeddings)
    target_node_embeddings = np.array(target_node_embeddings)

    elapsed_time = time.time() - start_time

    print("Finished generating embeddings!")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    
    return graph_embeddings, target_node_embeddings

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
        
        graph_embeddings, target_node_embeddings = get_node2vec_embeddings(
            graphs, product_id, dimensions, walk_length, num_walks, workers
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
    
    