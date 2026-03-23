import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Redefine plot_results to support saving
def plot_results(train, val, test, forecast, 
                train_index, val_index, test_index, 
                train_losses, val_losses, 
                target_col='value', title='Forecast vs Actual', save_path=None,
                rmse=None, mae=None, bias=None, score=None, pocid=None, df_full=None):
    
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
    full_title = f"{title}<br>RMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:.4f} | Score: {score:.4f} | POCID: {pocid:.4f}"

    # Visualize results
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=(full_title, 'Test vs Forecast', 'Training and Validation Loss'),
                        vertical_spacing=0.1)
    
    # Define a common hovertemplate to include date and value
    hover_temp = 'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}'

    # Plot forecast vs actual - ax1
    fig.add_trace(go.Scatter(x=train_index, y=train, name='Train', opacity=0.7, mode='lines', hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=val_index, y=val, name='Validation', opacity=0.7, mode='lines', line=dict(color='orange'), hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test', mode='lines', line=dict(color='green', width=2), hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=forecast, name='Forecast', mode='lines', line=dict(color='red', width=2, dash='dash'), hovertemplate=hover_temp), row=1, col=1)
    
    # Pinpoint specific dates
    fig.add_vline(x='2022-09-24', line_dash="dot", line_color="purple", line_width=2, row=1, col=1)
    if len(test_index) > 0:
        # Use .iloc[-1] to access the last element by position, not by label
        if hasattr(test_index, 'iloc'):
            last_date = test_index.iloc[-1]
        else:
            last_date = test_index[-1]
        fig.add_vline(x=last_date, line_dash="dot", line_color="brown", line_width=2, row=1, col=1)
        
    # Mark promotion days using scatter points
    if df_full is not None:
        # Detect promo columns dynamically
        promo_type_cols = [c for c in df_full.columns if c.startswith('promo_type_')]
        promo_colors = ['rgba(255, 0, 255, 0.5)', 'rgba(0, 255, 255, 0.5)', 'rgba(255, 255, 0, 0.5)']
        color_idx = 0
        
        for promo_col in promo_type_cols:
            promo_dates = df_full[df_full[promo_col] == 1]['date']
            # Find the target values for these dates across train, val, and test to correctly place markers
            y_vals = []
            valid_dates = []
            
            for d in promo_dates:
                # Combine train, val, test arrays and indices
                all_idx = np.concatenate([train_index, val_index, test_index])
                all_y = np.concatenate([train, val, test])
                
                # Check if date is in indices
                if d in all_idx:
                    idx_pos = np.where(all_idx == d)[0][0]
                    y_vals.append(all_y[idx_pos])
                    valid_dates.append(d)
                    
            if valid_dates:
                 color = promo_colors[color_idx % len(promo_colors)]
                 promo_name = promo_col.replace('promo_type_', '')
                 # Add scatter points where promotions occur
                 fig.add_trace(go.Scatter(x=valid_dates, y=y_vals, mode='markers',
                                          name=f'Promo: {promo_name}',
                                          marker=dict(color=color, size=10, symbol='star')), 
                               row=1, col=1)
                 color_idx += 1
                 
    # Set tick format for x-axis to 3 months     
    fig.update_xaxes(
        title_text='Date', 
        dtick="M3", 
        tickformat="%b\n%Y",
        row=1, col=1
    )
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    
    # Plot forecast vs actual test only - ax2
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test (Zoom)', mode='lines', line=dict(color='green', width=2), showlegend=False, hovertemplate=hover_temp), row=2, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=forecast, name='Forecast (Zoom)', mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=False, hovertemplate=hover_temp), row=2, col=1)
    
    fig.update_xaxes(
        title_text='Date',
        dtick="M3",
        tickformat="%b\n%Y",
        row=2, col=1
    )
    fig.update_yaxes(title_text=target_col, row=2, col=1)

    # Plot training loss - ax3
    epochs = list(range(1, len(train_losses) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name='Train Loss', mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name='Validation Loss', mode='lines'), row=3, col=1)
    
    fig.update_yaxes(title_text='Loss', row=3, col=1)
    fig.update_xaxes(title_text='Epoch', row=3, col=1)
    
    fig.update_layout(height=1200, width=1000, 
                      hovermode="x unified",
                      template="plotly_white")
    
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            # If kaleido is installed, it can save to png/jpg/pdf. 
            # Otherwise we'll fallback to writing html if it fails.
            try:
                fig.write_image(save_path)
            except Exception:
                fig.write_html(save_path + '.html')
    else:
        fig.show()