import time
import torch
import numpy as np
import torch.nn as nn

    
def pretrain_and_cache_graphsage(graphs_data, target_idx, input_size, out_size=32, num_layers=2, epochs=10, lr=0.001, device='cpu', gcn=False, agg_func='MEAN'):
    """
    Unsupervised pre-training of the GraphSAGE model on a sequence of historical graph snapshots,
    followed by the extraction (caching) of the target node's embeddings to be used as exogenous features for the LSTM.

    Args:
        graphs_data: List of dictionaries for each timestep t in the training set.
                     Format: {'adj_lists': Dict[int, Set[int]], 'raw_features': torch.Tensor(N, W)}
        target_idx: The integer index of the target SKU (i*) in the graph.
        input_size: The dimension of input features (W, window size).
        out_size: The dimension of the output graph embedding (d).
        num_layers: Number of GraphSAGE layers (default: 2).
        epochs: Unsupervised training epochs.
        
    Returns:
        model: The trained GraphSAGE model.
        cached_embeddings: Numpy array of shape (T, out_size) containing the target's embeddings.
    """
    from GNN.DynamicSimilarities.GraphSAGE.GraphSAGE import GraphSAGE, UnsupervisedLoss
    
    if not graphs_data:
        print("No graph data provided.")
        return None, np.array([])
        
    # Initialize the GraphSAGE model using the topology and features of the first snapshot
    first_graph = graphs_data[0]
    model = GraphSAGE(
        num_layers=num_layers,
        input_size=input_size, 
        out_size=out_size,      
        raw_features=first_graph['raw_features'].to(device),
        adj_lists=first_graph['adj_lists'],
        device=device,
        gcn=gcn,
        agg_func=agg_func
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting GraphSAGE unsupervised pre-training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for current_graph in graphs_data:
            model.train()
            # Update the graph structure dynamically for this timestep
            model.adj_lists = current_graph['adj_lists']
            model.raw_features = current_graph['raw_features'].to(device)
            
            all_nodes = list(range(model.raw_features.shape[0]))
            
            # Unsupervised Loss formulation (uses random walks & negative sampling)
            unsup_loss = UnsupervisedLoss(model.adj_lists, all_nodes, device)
            unique_nodes_batch = unsup_loss.extend_nodes(all_nodes, num_neg=5)
            
            optimizer.zero_grad()
            batch_embeddings = model(unique_nodes_batch)
            
            loss = unsup_loss.get_loss_sage(batch_embeddings, unique_nodes_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Unsupervised Loss: {total_loss/len(graphs_data):.4f}")
            
    print("Pre-training complete. Caching embeddings for the target node...")
    model.eval()
    cached_embeddings = []
    
    with torch.no_grad():
        for current_graph in graphs_data:
            model.adj_lists = current_graph['adj_lists']
            model.raw_features = current_graph['raw_features'].to(device)
            
            # Forward pass specifically for the target SKU node
            target_emb = model([target_idx])
            cached_embeddings.append(target_emb.cpu().numpy().flatten())
            
    cached_embeddings = np.array(cached_embeddings)
    print(f"Cached embeddings shape: {cached_embeddings.shape}")
    
    return model, cached_embeddings