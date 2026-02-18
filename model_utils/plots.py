"""
Plotting utilities for LSTM time series forecasting.

This module contains functions for visualizing training progress, data distributions,
and forecast comparisons for time series models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from IPython.display import Image, display

def create_plot_dir(base_dir: str, store_id: int, item_id: int) -> Path:
    """Create directory for saving plots for a specific item.
    
    Parameters
    ----------
    base_dir : str
        Base directory for all plots
    store_id : int
        Store identifier
    item_id : int
        Item identifier
        
    Returns
    -------
    Path
        Path to the created directory
    """
    plot_dir = Path(base_dir) / f"store_{store_id}_item_{item_id}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_loss_curves(train_losses: List[float], val_losses: List[float],
                    store_id: int, item_id: int, save_dir: str) -> None:
    """Plot and save training and validation loss curves.
    
    Parameters
    ----------
    train_losses : List[float]
        Training losses per epoch
    val_losses : List[float]
        Validation losses per epoch (can be empty)
    store_id : int
        Store identifier for title/filename
    item_id : int
        Item identifier for title/filename
    save_dir : str
        Directory to save the plot
    """
    plot_dir = create_plot_dir(save_dir, store_id, item_id)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'Training History - Store {store_id}, Item {item_id}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_distribution(df: pd.DataFrame, store_id: int, item_id: int,
                          train_end: int, val_end: int, save_dir: str) -> None:
    """Plot data distribution across train/val/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'value' columns
    store_id : int
        Store identifier
    item_id : int
        Item identifier
    train_end : int
        Index where training data ends
    val_end : int
        Index where validation data ends
    save_dir : str
        Directory to save the plot
    """
    plot_dir = create_plot_dir(save_dir, store_id, item_id)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall time series
    ax = axes[0, 0]
    ax.plot(df['date'], df['value'], 'k-', alpha=0.7, linewidth=1)
    ax.axvline(df['date'].iloc[train_end], color='b', linestyle='--', label='Train End', alpha=0.7)
    ax.axvline(df['date'].iloc[val_end], color='r', linestyle='--', label='Val End', alpha=0.7)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Sales Value', fontsize=10)
    ax.set_title('Time Series with Train/Val/Test Splits', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Distribution histogram
    ax = axes[0, 1]
    train_vals = df['value'].iloc[:train_end]
    val_vals = df['value'].iloc[train_end:val_end]
    test_vals = df['value'].iloc[val_end:]
    
    ax.hist(train_vals, bins=30, alpha=0.6, label='Train', color='blue', density=True)
    ax.hist(val_vals, bins=30, alpha=0.6, label='Val', color='orange', density=True)
    ax.hist(test_vals, bins=30, alpha=0.6, label='Test', color='green', density=True)
    ax.set_xlabel('Sales Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Value Distribution by Split', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Box plot by split
    ax = axes[1, 0]
    box_data = [train_vals, val_vals, test_vals]
    bp = ax.boxplot(box_data, labels=['Train', 'Val', 'Test'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_ylabel('Sales Value', fontsize=10)
    ax.set_title('Value Distribution (Box Plot)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Data Statistics:
    
    Train Set (60%):
      • Samples: {len(train_vals)}
      • Mean: {train_vals.mean():.2f}
      • Std: {train_vals.std():.2f}
      • Min: {train_vals.min():.2f}
      • Max: {train_vals.max():.2f}
    
    Validation Set (20%):
      • Samples: {len(val_vals)}
      • Mean: {val_vals.mean():.2f}
      • Std: {val_vals.std():.2f}
    
    Test Set (20%):
      • Samples: {len(test_vals)}
      • Mean: {test_vals.mean():.2f}
      • Std: {test_vals.std():.2f}
    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Data Overview - Store {store_id}, Item {item_id}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(plot_dir / 'data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_forecast_comparison(df: pd.DataFrame, forecast: np.ndarray,
                            store_id: int, item_id: int, val_end: int,
                            save_dir: str) -> None:
    """Plot actual vs predicted values and forecast distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'value' columns
    forecast : np.ndarray
        Forecasted values
    store_id : int
        Store identifier
    item_id : int
        Item identifier
    val_end : int
        Index where validation ends (test starts)
    save_dir : str
        Directory to save the plot
    """
    plot_dir = create_plot_dir(save_dir, store_id, item_id)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series with forecast
    ax = axes[0, 0]
    test_actual = df['value'].iloc[val_end:]
    test_dates = df['date'].iloc[val_end:]
    
    # Plot historical data
    ax.plot(df['date'], df['value'], 'k-', alpha=0.4, linewidth=1, label='Historical')
    # Plot test actual
    ax.plot(test_dates, test_actual, 'g-', linewidth=2, label='Actual (Test)', marker='o', markersize=3)
    # Plot forecast
    ax.plot(test_dates[:len(forecast)], forecast, 'r--', linewidth=2, label='Forecast', marker='x', markersize=4)
    
    ax.axvline(test_dates.iloc[0], color='orange', linestyle='--', alpha=0.5, label='Forecast Start')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Sales Value', fontsize=10)
    ax.set_title('Forecast vs Actual', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Zoom on test period
    ax = axes[0, 1]
    ax.plot(test_dates, test_actual, 'g-', linewidth=2, label='Actual', marker='o', markersize=4)
    ax.plot(test_dates[:len(forecast)], forecast, 'r--', linewidth=2, label='Forecast', marker='x', markersize=5)
    ax.fill_between(test_dates[:len(forecast)], forecast, test_actual[:len(forecast)], 
                    alpha=0.3, color='gray', label='Error')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Sales Value', fontsize=10)
    ax.set_title('Test Period Detail', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Distribution comparison
    ax = axes[1, 0]
    actual_test = test_actual[:len(forecast)]
    
    ax.hist(df['value'], bins=40, alpha=0.5, label='All Data', color='gray', density=True)
    ax.hist(actual_test, bins=20, alpha=0.7, label='Actual (Test)', color='green', density=True)
    ax.hist(forecast, bins=20, alpha=0.7, label='Forecast', color='red', density=True)
    ax.set_xlabel('Sales Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution: Forecast vs Actual vs Overall', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Error metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    mae = np.mean(np.abs(forecast - actual_test))
    rmse = np.sqrt(np.mean((forecast - actual_test) ** 2))
    mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
    
    metrics_text = f"""
    Forecast Metrics:
    
    Error Metrics:
      • MAE:  {mae:.4f}
      • RMSE: {rmse:.4f}
      • MAPE: {mape:.2f}%
    
    Forecast Statistics:
      • Mean:   {forecast.mean():.2f}
      • Std:    {forecast.std():.2f}
      • Min:    {forecast.min():.2f}
      • Max:    {forecast.max():.2f}
    
    Actual Statistics (Test):
      • Mean:   {actual_test.mean():.2f}
      • Std:    {actual_test.std():.2f}
      • Min:    {actual_test.min():.2f}
      • Max:    {actual_test.max():.2f}
    
    Correlation: {np.corrcoef(forecast, actual_test)[0,1]:.4f}
    """
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'Forecast Analysis - Store {store_id}, Item {item_id}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(plot_dir / 'forecast_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def show_item_plots(store_id, item_id, plots_dir='./training_plots'):
    """Display all plots for a specific item."""
    item_dir = Path(plots_dir) / f"store_{store_id}_item_{item_id}"
    
    if not item_dir.exists():
        print(f"No plots found for Store {store_id}, Item {item_id}")
        return
    
    plot_files = ['data_distribution.png', 'loss_curves.png', 'forecast_comparison.png']
    
    for plot_file in plot_files:
        plot_path = item_dir / plot_file
        if plot_path.exists():
            print(f"\n{'='*80}")
            print(f"  {plot_file.replace('_', ' ').title().replace('.png', '')}")
            print('='*80)
            display(Image(filename=str(plot_path)))
        else:
            print(f"Plot not found: {plot_file}")

# View plots for the first trained item