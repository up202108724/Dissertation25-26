import pandas as pd
import numpy as np
import networkx as nx

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
        .ffill()
        .fillna(0)
    )

    # 2) Compute item-item similarity matrix
    # Correlate columns (items) across time
    sim_df = df_pivot.corr(method=similarity_method)

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
                    print(
                        f"Added edge between {item_ids[i]} and {neighbor_id} "
                        f"with {similarity_method} similarity: {sim_value:.4f}"
                    )

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
                    print(
                        f"Added edge between {item_ids[i]} and {item_ids[j]} "
                        f"with {similarity_method} similarity: {sim_value:.4f}"
                    )

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