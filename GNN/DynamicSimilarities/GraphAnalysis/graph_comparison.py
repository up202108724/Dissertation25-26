import os
import pickle
import networkx as nx

def load_graphs(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_graphs(g1, g2):
    diffs = []
    
    # Compare nodes
    nodes1 = set(g1.nodes())
    nodes2 = set(g2.nodes())
    
    if nodes1 != nodes2:
        only_1 = nodes1 - nodes2
        only_2 = nodes2 - nodes1
        if only_1: diffs.append(f"Nodes only in G1: {only_1}")
        if only_2: diffs.append(f"Nodes only in G2: {only_2}")
        
    # Compare edges
    edges1 = set([tuple(sorted((u, v))) for u, v in g1.edges()])
    edges2 = set([tuple(sorted((u, v))) for u, v in g2.edges()])
    
    if edges1 != edges2:
        only_1_e = edges1 - edges2
        only_2_e = edges2 - edges1
        if only_1_e: diffs.append(f"Edges only in G1: {only_1_e}")
        if only_2_e: diffs.append(f"Edges only in G2: {only_2_e}")
        
    return diffs

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Parameters to locate the PKLs
    product_id = 26008
    metric = "spearman"
    window_size = 15
    step_size = 1
    prefix = "" # "star_" or "2nddegree_star_" if you used those
    
    # Pairwise comparing these thresholds
    thresholds = [0.616, 0.618]
    
    print(f"--- Graph Comparison Tool ---")
    print(f"Item: {product_id} | Metric: {metric} | Thresholds: {thresholds}\n")
    
    # Load all into a dictionary
    graphs_dict = {}
    for th in thresholds:
        dir_label = f"th{th}"
        pkl_name = f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl"
        pkl_path = os.path.join(BASE_DIR, "DynamicGraphPkls", str(product_id), metric, str(window_size), str(step_size), dir_label, pkl_name)
        
        graphs = load_graphs(pkl_path)
        if graphs is not None:
            graphs_dict[th] = graphs
            
    # Iterate and compare pairwise (e.g. 0.611 vs 0.612, 0.612 vs 0.613)
    for i in range(len(thresholds) - 1):
        th1 = thresholds[i]
        th2 = thresholds[i+1]
        
        print(f"===========================================================")
        print(f" COMPARING THRESHOLD {th1} (G1) vs THRESHOLD {th2} (G2)")
        print(f"===========================================================")
        
        if th1 not in graphs_dict or th2 not in graphs_dict:
            print("Missing data for comparison. Skipping.\n")
            continue
            
        g_list_1 = graphs_dict[th1]
        g_list_2 = graphs_dict[th2]
        
        if len(g_list_1) != len(g_list_2):
            print(f"Mismatch in total graphs length! {len(g_list_1)} vs {len(g_list_2)}")
            
        diff_count = 0
        
        for idx, (g1, g2) in enumerate(zip(g_list_1, g_list_2)):
            diffs = compare_graphs(g1, g2)
            
            if diffs:
                diff_count += 1
                start_d = g1.graph.get('start_date', 'UnknownDate')
                end_d = g1.graph.get('end_date', 'UnknownDate')
                
                print(f"--- Difference found at Window {idx} ({start_d} to {end_d}) ---")
                for d in diffs:
                    print(f"  > {d}")
                print()
                
        if diff_count == 0:
            print(f"No differences found between {th1} and {th2}! They have identical structures.\n")
        else:
            print(f"Total disparate windows found: {diff_count}\n")
