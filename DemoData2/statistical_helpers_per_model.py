import os
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

# CD diagrams are written here, resolved relative to this file so the location
# is independent of the notebook's working directory.
CD_DIAGRAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CD Diagrams")

# Model variants grouped by forecasting head. The statistical tests are run
# separately per family so LSTM and MLP variants are never ranked together.
MODEL_FAMILIES = {
    'lstm': ['lstm_baseline', 'graph2vec_lstm', 'gcn_lstm', 'gat_lstm'],
    'mlp':  ['mlp_baseline', 'graph2vec_mlp', 'gcn_mlp', 'gat_mlp'],
}


def run_demsar_framework_by_family(data_input, metric='rmse', alpha=0.05, mode='total', seed_value=None, compare_ablations=False, model_family='lstm'):
    """
    Executes the Demšar statistical framework (Friedman + Nemenyi).
    compare_ablations: If True, treats ablate_z=True rows as separate models. 
                       If False, filters them out to only evaluate full models.
    model_family:      Which group of models to compare. 'lstm' restricts the test
                       to (lstm_baseline, graph2vec_lstm, gcn_lstm, gat_lstm); 'mlp'
                       restricts it to (mlp_baseline, graph2vec_mlp, gcn_mlp,
                       gat_mlp). Use 'all'/None to compare every variant together.
    """
    if mode == 'per_seed' and seed_value is None:
        raise ValueError("You must provide a 'seed_value' when using mode='per_seed'.")

    # 1. Load Data
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("data_input must be a file path string or a pandas DataFrame")

    # ---------------------------------------------------------
    # Restrict to a single model family (LSTM or MLP) so the
    # statistical comparison only ranks like-for-like heads.
    # ---------------------------------------------------------
    if model_family not in (None, 'all'):
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {list(MODEL_FAMILIES)}, 'all', or None.")
        family_models = MODEL_FAMILIES[model_family]
        df = df[df['model_variant'].isin(family_models)]
        print(f"[*] MODEL FAMILY: '{model_family}' -> {family_models}")

    # ---------------------------------------------------------
    # NEW: Handle Ablations elegantly
    # ---------------------------------------------------------
    if compare_ablations:
        # Identify rows where ablate_z is True
        is_ablated = df['ablate_z'] == True
        # Rename them so the pivot table treats them as a brand new model
        df.loc[is_ablated, 'model_variant'] = df.loc[is_ablated, 'model_variant'] + '_ablated'
        print(f"[*] ABLATION MODE: Treating ablate_z=True as separate models.")
    else:
        # Filter OUT ablated runs. Keep only ablate_z=False OR baselines (which are NaN)
        df = df[(df['ablate_z'] == False) | (df['ablate_z'].isna())]
        print(f"[*] STANDARD MODE: Filtered out ablate_z=True runs.")

    # 2. Handle Modes (Filtering and Block Selection)
    if mode == 'per_seed':
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | SEED: {seed_value} ---")
        print(f"{'='*60}")
        df = df[df['seed'] == seed_value]
        block_cols = ['product_id']
    else:
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | TOTAL SEEDS ---")
        print(f"{'='*60}")
        block_cols = ['product_id', 'seed']

    # 3. Pivot the data to get blocks
    pivot_df = df.pivot_table(
        index=block_cols, 
        columns='model_variant', 
        values=metric, 
        aggfunc='first' 
    )
    
    initial_rows = len(pivot_df)
    pivot_df = pivot_df.dropna()
    print(f"Loaded {initial_rows} blocks.")
    print(f"Retained {len(pivot_df)} complete blocks after dropping missing runs.\n")
    
    if len(pivot_df) < 2:
        print("Not enough complete blocks to run statistical tests.")
        return None, None, None

    models = pivot_df.columns.tolist()
    
    # 4. Calculate Average Ranks
    ranks = pivot_df.rank(axis=1, method='average', ascending=True)
    avg_ranks = ranks.mean().sort_values()
    
    print("--- Average Ranks (Lower is Better) ---")
    for model, rank in avg_ranks.items():
        print(f"{model:<25} : {rank:.3f}")
    print("\n")
    
    # 5. Omnibus Friedman Test
    data_arrays = [pivot_df[model].values for model in models]
    stat, p_value = stats.friedmanchisquare(*data_arrays)
    
    print("--- Friedman Test Results ---")
    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value:        {p_value:.4e}")
    
    if p_value < alpha:
        print(f"\nResult: Reject the null hypothesis (p < {alpha}).")
        print("There is a statistically significant difference between the models.")
        
        # 6. Post-Hoc Nemenyi Test
        print("\n--- Post-Hoc Nemenyi Test (Pairwise p-values) ---")
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        
        nemenyi_results.columns = models
        nemenyi_results.index = models
        
        print(nemenyi_results.round(4))
        
        print("\n--- Significant Pairwise Differences (p < 0.05) ---")
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_a = models[i]
                model_b = models[j]
                p_val = nemenyi_results.loc[model_a, model_b]
                
                if p_val < alpha:
                    better_model = model_a if avg_ranks[model_a] < avg_ranks[model_b] else model_b
                    worse_model = model_b if better_model == model_a else model_a
                    print(f"✅ {better_model} significantly outperforms {worse_model} (p = {p_val:.4f})")
    else:
        print(f"\nResult: Fail to reject the null hypothesis (p >= {alpha}).")
        print("There is NO statistically significant difference between the models.")
        nemenyi_results = None
        
    return pivot_df, avg_ranks, nemenyi_results


def plot_cd_diagram(avg_ranks, nemenyi_results, metric='rmse', graph_type='Spearman', mode='total', seed_value=None, compare_ablations=False, model_family='lstm'):
    """
    Generates, saves, and displays the Critical Difference Diagram.
    """
    if nemenyi_results is None:
        print("\nSkipping CD diagram: No statistically significant differences found.")
        return

    # Add an ablation tag to the file name and title so it doesn't overwrite your standard plots
    ablation_tag_title = " (With Ablations)" if compare_ablations else ""
    ablation_tag_file = "_ablations" if compare_ablations else ""

    # Add a model-family tag so LSTM and MLP diagrams are saved separately
    family_tag_title = f" - {model_family.upper()}" if model_family not in (None, 'all') else ""
    family_tag_file = f"_{model_family.lower()}" if model_family not in (None, 'all') else ""

    if mode == 'per_seed':
        plot_title = f"Critical Difference Diagram ({metric.upper()} - {graph_type.capitalize()}{family_tag_title}{ablation_tag_title} - Seed {seed_value})\n"
        save_plot_path = f"cd_diagram_{metric.lower()}_{graph_type.lower()}{family_tag_file}_seed{seed_value}{ablation_tag_file}.pdf"
        print(f"\nGenerating CD Diagram for {graph_type.capitalize()} (Seed {seed_value})...")
    else:
        plot_title = f"Critical Difference Diagram ({metric.upper()} - {graph_type.capitalize()}{family_tag_title}{ablation_tag_title} - Total Aggregation)\n"
        save_plot_path = f"cd_diagram_{metric.lower()}_{graph_type.lower()}{family_tag_file}_total{ablation_tag_file}.pdf"
        print(f"\nGenerating CD Diagram for {graph_type.capitalize()} (Total Aggregation)...")

    os.makedirs(CD_DIAGRAM_DIR, exist_ok=True)
    save_plot_path = os.path.join(CD_DIAGRAM_DIR, save_plot_path)

    fig = plt.figure(figsize=(10, 4))
    
    sp.critical_difference_diagram(avg_ranks, nemenyi_results)
    
    plt.title(plot_title, fontsize=14, weight='bold')
    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot successfully saved to: {save_plot_path}")
    
    plt.show()