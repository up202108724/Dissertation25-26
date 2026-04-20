import pickle
import os

def load_graphs(pkl_filepath, num_graphs=7):
    
    if not os.path.exists(pkl_filepath):
        raise FileNotFoundError(f"The specified file does not exist: {pkl_filepath}")
        
    # Load the dynamic sequence of graphs
    with open(pkl_filepath, 'rb') as f:
        all_graphs = pickle.load(f)
        
    print(f"Successfully loaded a sequence of {len(all_graphs)} total graphs.")
    
    # Slice the list to retrieve the requested number of graphs
    sample_graphs = all_graphs[:num_graphs]
    
    print(f"Extracted {len(sample_graphs)} graphs to send via email.")
    return sample_graphs

if __name__ == '__main__':
    # Use absolute path resolving from this script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, 'DynamicGraphPkls', 'cid', '15', '1', '907969', 'star_dynamic_graphs_cid_Window15_Step1_pct0.5.pkl')
    
    graphs = load_graphs(pkl_path, num_graphs=7)
    print(graphs)
