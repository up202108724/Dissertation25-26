import matplotlib.pyplot as plt

# Redefine plot_results to support saving
def plot_results(train, val, test, forecast, 
                train_index, val_index, test_index, 
                train_losses, val_losses, 
                target_col='value', title='Forecast vs Actual', save_path=None):
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot forecast vs actual - ax1
    ax1.plot(train_index, train, label='Train', alpha=0.7)
    ax1.plot(val_index, val, label='Validation', alpha=0.7, color='orange')
    ax1.plot(test_index, test, label='Actual Test', linewidth=2, color='green')
    ax1.plot(test_index, forecast, label='Forecast', linestyle='--', linewidth=2, color='red')
    ax1.legend()
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(target_col)
    ax1.grid(True, alpha=0.3)
    
    # Plot training loss - ax2
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.legend()
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()