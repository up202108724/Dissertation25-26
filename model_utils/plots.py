import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Redefine plot_results to support saving
def plot_results(train, val, test, forecast, 
                train_index, val_index, test_index, 
                train_losses, val_losses, 
                target_col='value', title='Forecast vs Actual', save_path=None,
                rmse=None, mae=None, bias=None, score=None):
    
    # Calculate metrics if not provided
    if rmse is None:
        rmse = np.sqrt(mean_squared_error(test, forecast))
    if mae is None:
        mae = mean_absolute_error(test, forecast)
    if bias is None:
        bias = np.mean(forecast - test)
    if score is None:
        score = r2_score(test, forecast)
        
    # Update title
    title = f"{title}\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:.4f} | Score: {score:.4f}"

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot forecast vs actual - ax1
    ax1.plot(train_index, train, label='Train', alpha=0.7)
    ax1.plot(val_index, val, label='Validation', alpha=0.7, color='orange')
    ax1.plot(test_index, test, label='Actual Test', linewidth=2, color='green')
    ax1.plot(test_index, forecast, label='Forecast', linestyle='--', linewidth=2, color='red')
    
    # Set x-axis major ticks to every 100 days
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=100))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Pinpoint specific dates
    ax1.axvline(pd.to_datetime('2022-09-24'), color='purple', linestyle=':', linewidth=2, label='2022-09-24')
    if len(test_index) > 0:
        # Use .iloc[-1] to access the last element by position, not by label
        if hasattr(test_index, 'iloc'):
            last_date = test_index.iloc[-1]
        else:
            last_date = test_index[-1]
        ax1.axvline(last_date, color='brown', linestyle=':', linewidth=2, label='Last Date')
        
    ax1.legend()
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(target_col)
    ax1.grid(True, alpha=0.3)
    
    # Plot forecast vs actual test only - ax2
    ax2.plot(test_index, test, label='Actual Test', linewidth=2, color='green')
    ax2.plot(test_index, forecast, label='Forecast', linestyle='--', linewidth=2, color='red')
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.legend()
    ax2.set_title('Test vs Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(target_col)
    ax2.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    
    # Plot training loss - ax3
    ax3.plot(train_losses, label='Train Loss')
    ax3.plot(val_losses, label='Validation Loss')
    ax3.legend()
    ax3.set_title('Training and Validation Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()