import sys
import os
import pickle
import torch
import pandas as pd
import numpy as np
import networkx as nx
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

def compute_correlation_matrix_gpu(df_pivot: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Computes correlation matrix on GPU using PyTorch for massive speedups."""
    if not torch.cuda.is_available():
        print(f"CUDA not available. Falling back to pandas/CPU for {method} correlation.")
        return df_pivot.corr(method=method)

    device = torch.device('cuda')
    X = torch.tensor(df_pivot.values, dtype=torch.float32, device=device)
    
    if method == "pearson":
        corr_tensor = torch.corrcoef(X.T)
        
    elif method == "spearman":
        # Fast GPU rank approximation (without tie averaging)
        _, indices = torch.sort(X, dim=0)
        ranks = torch.empty_like(X)
        ranks.scatter_(0, indices, torch.arange(1, X.shape[0]+1, dtype=torch.float32, device=device).unsqueeze(1).expand_as(X))
        corr_tensor = torch.corrcoef(ranks.T)
        
    elif method == "kendall":
        # Kendall Tau-b formula using pairwise signs on GPU
        M, N = X.shape
        if M < 2:
            corr_tensor = torch.eye(N, device=device)
        else:
            # Get all combinations of row indices i < j
            idx1, idx2 = torch.triu_indices(M, M, offset=1, device=device)
            # Find pairwise differences and directions (signs)
            signs = torch.sign(X[idx1] - X[idx2]) # Shape: [num_pairs, num_items]
            
            # S matrix (Concordant - Discordant pairs)
            # Dot product of signs between items calculates matching vs mismatching pairs 
            S = torch.mm(signs.T, signs) 
            
            # Number of non-tied pairs for each item
            non_ties = torch.sum(signs ** 2, dim=0) 
            # Kendall Tau-b denominator
            denom = torch.sqrt(torch.outer(non_ties, non_ties))
            
            corr_tensor = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
            corr_tensor.fill_diagonal_(1.0)
    else:
        return df_pivot.corr(method=method)

    # Convert back to pandas DataFrame matching df_pivot columns
    return pd.DataFrame(corr_tensor.cpu().numpy(), index=df_pivot.columns, columns=df_pivot.columns)


def build_statistical_similarity_graph(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    similarity_method: str = "pearson",   # "pearson", "spearman", "kendall"
    similarity_threshold: float = 0.7,
    k: int = None,
    use_absolute_similarity: bool = False,
):
    # 1) Pivot: rows = dates, cols = items
    df_pivot = (
        df.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc
        )
        .sort_index()
        .fillna(0)
    )

    # 2) Compute item-item similarity matrix
    # Correlate columns (items) across time
    print(f"Computing {similarity_method} similarity matrix...")
    sim_df = compute_correlation_matrix_gpu(df_pivot, method=similarity_method)
    print(f"Finishing computing similarity matrix. Shape: {sim_df.shape}")
    item_ids = sim_df.columns.tolist()
    # 3) Build graph
    G = nx.Graph(name=f"{similarity_method.capitalize()}_Similarity_Graph")
    G.add_nodes_from(item_ids)

    n = len(item_ids)

    if k is not None:
        # k-NN graph: connect each node to top-k most similar neighbors
        for i in range(n):
            sim_row = sim_df.iloc[i].copy()
            sim_row.iloc[i] = np.nan  # remove self-correlation

            if use_absolute_similarity:
                top_k_neighbors = sim_row.abs().nlargest(k).dropna()
            else:
                top_k_neighbors = sim_row.nlargest(k).dropna()

            for neighbor_id, sim_value in top_k_neighbors.items():
                j = sim_df.columns.get_loc(neighbor_id)

                edge_weight = abs(sim_value) if use_absolute_similarity else sim_value

                if use_absolute_similarity or sim_value >= similarity_threshold:
                    G.add_edge(
                        item_ids[i],
                        neighbor_id,
                        weight=float(edge_weight),
                        similarity=float(sim_value)
                    )
                    '''
                    print(
                        f"Added edge between {item_ids[i]} and {item_ids[j]} "
                        f"with {similarity_method} similarity: {sim_value:.4f}"
                    )
                    '''

    else:
        # Threshold graph
        for i in range(n):
            for j in range(i + 1, n):
                sim_value = sim_df.iloc[i, j]

                if pd.isna(sim_value):
                    continue

                edge_weight = abs(sim_value) if use_absolute_similarity else sim_value

                condition = (
                    abs(sim_value) >= similarity_threshold
                    if use_absolute_similarity
                    else sim_value >= similarity_threshold
                )

                if condition:
                    G.add_edge(
                        item_ids[i],
                        item_ids[j],
                        weight=float(edge_weight),
                        similarity=float(sim_value)
                    )
                    '''
                    print(
                        f"Added edge between {item_ids[i]} and {item_ids[j]} "
                        f"with {similarity_method} similarity: {sim_value:.4f}"
                    )
                    '''

    print(f"Number of nodes in the {similarity_method} graph:", G.number_of_nodes())
    print(f"Number of edges in the {similarity_method} graph:", G.number_of_edges())
    
    return G, sim_df, df_pivot

def build_dynamic_similarity_graphs(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    window_size: int,
    step_size: int,
    aggfunc: str = "sum",
    similarity_method: str = "pearson",
    similarity_threshold: float = 0.7,
    k: int = None,
    use_absolute_similarity: bool = False,
):
    """
    Builds a sequence of dynamic similarity graphs using a sliding window 
    over the dates in the dataframe.
    """
    unique_dates = sorted(df[date_col].unique())
    
    num_dates = len(unique_dates)
    graphs = []
    sim_dfs = []
    df_pivots = []
    window_info = []

    for start_idx in range(0, num_dates - window_size + 1, step_size):
        end_idx = start_idx + window_size
        current_dates = unique_dates[start_idx:end_idx]
        
        start_date = current_dates[0]
        end_date = current_dates[-1]
        
        mask = df[date_col].isin(current_dates)
        df_window = df[mask]
        
        print(f"\nBuilding graph for window: {start_date} to {end_date}")
        
        G, sim_df, df_pivot = build_statistical_similarity_graph(
            df=df_window,
            date_col=date_col,
            item_col=item_col,
            target_col=target_col,
            aggfunc=aggfunc,
            similarity_method=similarity_method,
            similarity_threshold=similarity_threshold,
            k=k,
            use_absolute_similarity=use_absolute_similarity
        )
        
        G.graph["start_date"] = start_date
        G.graph["end_date"] = end_date
        
        graphs.append(G)
        sim_dfs.append(sim_df)
        df_pivots.append(df_pivot)
        window_info.append({"start_date": start_date, "end_date": end_date})
        
    return graphs, sim_dfs, df_pivots, window_info

if __name__ == "__main__":
   
    
    # 1. Definir caminhos e hiperparâmetros
    # Como o script é executado a partir de c:/Users/Andre Silva/Desktop/Dissertation25-26/ (devido ao terminal),
    # o caminho relativo para o dataset deve ser dataset/independent_items.feather 
    # ou podemos usar os caminhos absolutos definindo a raiz.
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'data_andre.feather')
    
    
    DATE_COL = 'date'
    TARGET_COL = 'value'
    ITEM_COL = 'item_id'
    
    WINDOW_SIZE = 7
    STEP_SIZE = 1
    SIMILARITY_METHOD = "kendall"
    SIMILARITY_THRESHOLD = 0.8
    NUM_ITEMS = None  # Definir como None para usar todos os produtos
    COMPUTE_NODE_FEATURES = False  # Flag opcional para calcular features
    
    # Create subfolder based on SIMILARITY_METHOD
    output_dir = os.path.join(os.path.dirname(__file__), SIMILARITY_METHOD)
    os.makedirs(output_dir, exist_ok=True)
    
    OUTPUT_PATH = os.path.join(output_dir, f'dynamic_graphs_output_{SIMILARITY_METHOD}_{SIMILARITY_THRESHOLD}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl')
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    # Handle DATE_COL (ensure it is a column and not in the index)
    if DATE_COL not in df.columns:
        if DATE_COL in df.index.names:
            df = df.reset_index(level=DATE_COL)
        else:
            # Fallback: se apenas tiver um index complexo, reset genérico
            df = df.reset_index()
            
            # Se ainda assim não encontrar, é possível que a coluna se chame 'Date' em vez de 'date'
            if DATE_COL not in df.columns and DATE_COL.capitalize() in df.columns:
                df = df.rename(columns={DATE_COL.capitalize(): DATE_COL})
            elif DATE_COL not in df.columns and DATE_COL.upper() in df.columns:
                df = df.rename(columns={DATE_COL.upper(): DATE_COL})

    # Filtro opcional: limitar aos primeiros N produtos, como no notebook
    if NUM_ITEMS is not None:
        top_items = df[ITEM_COL].unique()[:NUM_ITEMS]
        df = df[df[ITEM_COL].isin(top_items)]
        print(f"Using top {NUM_ITEMS} products.")
    else:
        print("NUM_ITEMS is None. Using all available products.")
    
    # Garantir ordenação temporal
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, ITEM_COL]).reset_index(drop=True)
    
    # Filter dataset to exclude the test set (using your forecast horizon of 152)
    TEST_SIZE = 152
    unique_dates = df[DATE_COL].dropna().unique()
    train_val_dates = unique_dates[:-TEST_SIZE]
    df_train_val = df[df[DATE_COL].isin(train_val_dates)].copy()
    
    print("Building dynamic similarity graphs for train and validation...")
    graphs, sim_dfs, df_pivots, window_info = build_dynamic_similarity_graphs(
        df_train_val,
        date_col=DATE_COL,
        item_col=ITEM_COL,
        target_col=TARGET_COL,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        similarity_method=SIMILARITY_METHOD,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    
    # 2. Computar os node features para cada janela do grafo
    def compute_window_node_features(df_pivot: pd.DataFrame):
        num_items = df_pivot.shape[1]
        window_size = df_pivot.shape[0]
        features = []
        
        for j in range(num_items):
            item_ts = df_pivot.iloc[:, j].values
            last_demand = item_ts[-1] if window_size > 0 else 0
            mean7 = np.mean(item_ts[-7:]) if window_size >= 7 else np.mean(item_ts)
            mean28 = np.mean(item_ts[-28:]) if window_size >= 28 else np.mean(item_ts)
            std28 = np.std(item_ts[-28:]) if window_size >= 28 else np.std(item_ts)
            if window_size >= 28:
                zero_ratio28 = np.mean(item_ts[-28:] == 0)
                slope28 = np.polyfit(np.arange(28), item_ts[-28:], 1)[0]
                min_28 = np.min(item_ts[-28:])
                max_28 = np.max(item_ts[-28:])
            elif window_size > 1:
                zero_ratio28 = np.mean(item_ts == 0)
                slope28 = np.polyfit(np.arange(window_size), item_ts, 1)[0]
                min_28 = np.min(item_ts)
                max_28 = np.max(item_ts)
            else:
                zero_ratio28 = np.mean(item_ts == 0)
                slope28 = 0.0
                min_28 = item_ts[0] if window_size > 0 else 0
                max_28 = item_ts[0] if window_size > 0 else 0
                
            features.append([last_demand, mean7, mean28, std28, zero_ratio28, slope28, min_28, max_28])
        return np.array(features)

    if COMPUTE_NODE_FEATURES:
        print("Computing dynamic graph features...")
        dynamic_graph_features = []
        for pivot_table in df_pivots:
            feats = compute_window_node_features(pivot_table)
            dynamic_graph_features.append(torch.tensor(feats, dtype=torch.float32))
    else:
        dynamic_graph_features = None

    # 3. Empacotar tudo num dicionário e guardar
    output_data = {
        'graphs': graphs,
        'sim_dfs': sim_dfs,
        'df_pivots': df_pivots,
        'window_info': window_info,
        'dynamic_graph_features': dynamic_graph_features,
        'similarity_method': SIMILARITY_METHOD,
        'similarity_threshold': SIMILARITY_THRESHOLD,
    }
    
    print(f"Exporting results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(output_data, f)
        
    print("Done! Data exported successfully.")
    
    #from plot import save_dynamic_graph_plots
    #save_dynamic_graph_plots(graphs, SIMILARITY_METHOD, node_categories=None, window_size=WINDOW_SIZE, step_size=STEP_SIZE)