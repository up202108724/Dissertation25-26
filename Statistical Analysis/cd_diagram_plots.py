import os
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt


# CD diagrams are written here, resolved relative to this file so the location
# is independent of the notebook's working directory.
CD_DIAGRAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CD Diagrams")

def plot_cd_diagram(avg_ranks, nemenyi_results, metric='rmse', graph_type='Spearman', mode='total', seed_value=None, compare_ablations=False):
    """
    Generates, saves, and displays the Critical Difference Diagram.
    """
    if nemenyi_results is None:
        print("\nSkipping CD diagram: No statistically significant differences found.")
        return

    # Add an ablation tag to the file name and title so it doesn't overwrite your standard plots
    ablation_tag_title = " (With Ablations)" if compare_ablations else ""
    ablation_tag_file = "_ablations" if compare_ablations else ""

    if mode == 'per_seed':
        plot_title = f"Critical Difference Diagram ({metric.upper()} - {graph_type.capitalize()}{ablation_tag_title} - Seed {seed_value})\n"
        save_plot_path = f"cd_diagram_{metric.lower()}_{graph_type.lower()}_seed{seed_value}{ablation_tag_file}.pdf"
        print(f"\nGenerating CD Diagram for {graph_type.capitalize()} (Seed {seed_value})...")
    else:
        plot_title = f"Critical Difference Diagram ({metric.upper()} - {graph_type.capitalize()}{ablation_tag_title} - Total Aggregation)\n"
        save_plot_path = f"cd_diagram_{metric.lower()}_{graph_type.lower()}_total{ablation_tag_file}.pdf"
        print(f"\nGenerating CD Diagram for {graph_type.capitalize()} (Total Aggregation)...")

    os.makedirs(CD_DIAGRAM_DIR, exist_ok=True)
    save_plot_path = os.path.join(CD_DIAGRAM_DIR, save_plot_path)

    fig = plt.figure(figsize=(10, 4))
    
    sp.critical_difference_diagram(avg_ranks, nemenyi_results)
    
    plt.title(plot_title, fontsize=14, weight='bold')
    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot successfully saved to: {save_plot_path}")
    
    plt.show()
    
def plot_cd_diagram_per_family(avg_ranks, nemenyi_results, metric='rmse', graph_type='Spearman', mode='total', seed_value=None, compare_ablations=False, model_family='lstm'):
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
    elif mode == 'per_product':
        plot_title = f"Critical Difference Diagram ({metric.upper()} - {graph_type.capitalize()}{family_tag_title}{ablation_tag_title} - Per Product)\n"
        save_plot_path = f"cd_diagram_{metric.lower()}_{graph_type.lower()}{family_tag_file}_perproduct{ablation_tag_file}.pdf"
        print(f"\nGenerating CD Diagram for {graph_type.capitalize()} (Per Product, mean over seeds)...")
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