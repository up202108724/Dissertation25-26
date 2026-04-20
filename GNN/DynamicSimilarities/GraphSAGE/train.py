import time
import torch
import numpy as np
import torch.nn as nn

def train_model(seed, epochs, model, train_loader, val_loader, exog_cols, criterion, criterion2, optimizer, device, best_model_path, scheduler=None, patience=10):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    
    start_train_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                all_outputs.append(outputs)
                all_targets.append(batch_y)

        all_outputs = torch.cat(all_outputs, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        
        val_loss = criterion2(all_outputs, all_targets).item()
        val_losses.append(val_loss)
        
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter on improvement
            torch.save(model.state_dict(), best_model_path)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
        patience_counter +=1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    train_time = time.time() - start_train_time

    return model, train_losses, val_losses, best_epoch, train_time

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