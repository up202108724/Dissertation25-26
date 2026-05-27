import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_results(train, val, test, forecast, 
                train_index, val_index, test_index, 
                train_losses, val_losses, metric=None, embedding_strategy=None,
                window_size=None, step_size=None, threshold=None, percentile=None,
                enable_edges_within_star=None, seed=None,
                target_col='value', title='Forecast vs Actual', save_path=None,
                rmse=None, mae=None, bias=None, score=None, pocid=None, df_full=None):
    
    # Check if forecast is a dictionary; if not, wrap it in a dictionary for uniformity
    if not isinstance(forecast, dict):
        forecast = {'Forecast': forecast}
        rmse = {'Forecast': rmse}
        mae = {'Forecast': mae}
        bias = {'Forecast': bias}
        score = {'Forecast': score}
        pocid = {'Forecast': pocid}
        train_losses = {'Forecast': train_losses}
        val_losses = {'Forecast': val_losses}
        
    # Calculate metrics if not provided
    for label, fcast in forecast.items():
        if fcast is None:
            rmse[label], mae[label], bias[label], score[label] = 0, 0, 0, 0
            continue
            
        if rmse.get(label) is None:
            fcast_array = np.array(fcast, dtype=float)
            valid_mask = ~np.isnan(fcast_array)
            valid_test = test[valid_mask]
            valid_forecast = fcast_array[valid_mask]
            if len(valid_test) > 0:
                rmse[label] = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                mae[label] = mean_absolute_error(valid_test, valid_forecast)
                bias[label] = np.mean(valid_forecast - valid_test)
                score[label] = r2_score(valid_test, valid_forecast)
                # pocid would need to be calculated here ideally, but logic isn't provided here yet
            else:
                rmse[label], mae[label], bias[label], score[label] = 0, 0, 0, 0
            
    # Remove large individual metrics from title - instead we'll place them in the legend or skip them if multiple
    is_multi = len(forecast) > 1
        
    # Update title conditionally
    meta_parts = []
    if embedding_strategy is not None:
        meta_parts.append(f"Graph Params: {embedding_strategy}")
    if metric is not None:
        meta_parts.append(f"Metric: {metric}")
    if window_size is not None:
        meta_parts.append(f"Window: {window_size}")
    if step_size is not None:
        meta_parts.append(f"Step: {step_size}")
    if threshold is not None:
        meta_parts.append(f"Threshold: {threshold}")
    if percentile is not None:
        meta_parts.append(f"Percentile: {percentile}")
    if enable_edges_within_star is not None:
        meta_parts.append(f"Edges in Star: {enable_edges_within_star}")
    if seed is not None:
        meta_parts.append(f"Seed: {seed}")
    metadata = " | ".join(meta_parts)
    meta_html = f"<span style='font-size:14px;color:gray'>{metadata}</span><br>" if metadata else ""
    
    full_title = f"{title}<br>{meta_html}"
    if not is_multi:
        label = list(forecast.keys())[0]
        pocid_str = f"{pocid[label]:.4f}" if pocid.get(label) is not None else "N/A"
        _cs = (0.5 * rmse[label] + 0.25 * mae[label] + 0.25 * abs(bias[label])
               if None not in (rmse.get(label), mae.get(label), bias.get(label)) else None)
        cs_str = f"{_cs:.4f}" if _cs is not None else "N/A"
        full_title += (f"RMSE: {rmse[label]:.4f} | MAE: {mae[label]:.4f} | "
                       f"BIAS: {bias[label]:.4f} | POCID: {pocid_str} | Score: {cs_str}")
        
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=(full_title, 'Test vs Forecast', 'Training and Validation Loss'),
                        vertical_spacing=0.1)
    
    # Define a common hovertemplate to include date and value
    hover_temp = 'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}'

    # Plot forecast vs actual - ax1
    fig.add_trace(go.Scatter(x=train_index, y=train, name='Train', opacity=0.7, mode='lines', hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=val_index, y=val, name='Validation', opacity=0.7, mode='lines', line=dict(color='orange'), hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test', mode='lines', line=dict(color='green', width=2), hovertemplate=hover_temp), row=1, col=1)
    
    import plotly.express as px
    colors = px.colors.qualitative.Plotly
    
    for idx, (label, fcast) in enumerate(forecast.items()):
        if fcast is None: continue
        color = colors[idx % len(colors)]
        if not is_multi:
            color = 'red'
        
        legend_name = label
        if is_multi:
             pocid_str = f"{pocid[label]:.4f}" if pocid.get(label) is not None else "N/A"
             _cs = (0.5 * rmse[label] + 0.25 * mae[label] + 0.25 * abs(bias[label])
                    if None not in (rmse.get(label), mae.get(label), bias.get(label)) else None)
             cs_str = f"{_cs:.4f}" if _cs is not None else "N/A"
             legend_name = (f"{label} (RMSE: {rmse[label]:.2f} | MAE: {mae[label]:.2f} | "
                            f"BIAS: {bias[label]:.2f} | POCID: {pocid_str} | Score: {cs_str})")
             
        fig.add_trace(go.Scatter(x=test_index, y=fcast, name=legend_name, legendgroup=label, mode='lines', line=dict(color=color, width=2), hovertemplate=hover_temp), row=1, col=1)
    
    
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
    
    for idx, (label, fcast) in enumerate(forecast.items()):
        if fcast is None: continue
        color = colors[idx % len(colors)]
        if not is_multi:
            color = 'red'
        fig.add_trace(go.Scatter(x=test_index, y=fcast, name=label + ' (Zoom)', legendgroup=label, mode='lines', line=dict(color=color, width=2), showlegend=False, hovertemplate=hover_temp), row=2, col=1)
    
    fig.update_xaxes(
        title_text='Date',
        dtick="M3",
        tickformat="%b\n%Y",
        row=2, col=1
    )
    fig.update_yaxes(title_text=target_col, row=2, col=1)

    # Plot training loss - ax3
    if not isinstance(train_losses, dict):
        train_losses = {'Forecast': train_losses}
        val_losses = {'Forecast': val_losses}
        
    for idx, (label, t_loss) in enumerate(train_losses.items()):
        if t_loss is None:
            continue
        v_loss = val_losses.get(label, [])
        color = colors[idx % len(colors)]
        epochs = list(range(1, len(t_loss) + 1))
        
        name_t = 'Train Loss' if not is_multi else f'Train Loss ({label})'
        name_v = 'Validation Loss' if not is_multi else f'Val Loss ({label})'
        
        fig.add_trace(go.Scatter(x=epochs, y=t_loss, name=name_t, legendgroup=label, mode='lines', line=dict(color=color, dash='solid')), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=v_loss, name=name_v, legendgroup=label, mode='lines', line=dict(color=color, dash='dot')), row=3, col=1)
    
    fig.update_yaxes(title_text='Loss', row=3, col=1)
    fig.update_xaxes(title_text='Epoch', row=3, col=1)
    
    fig.update_layout(
        height=1400,
        width=2200,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=10), tracegroupgap=2),
        margin=dict(l=60, r=20, t=80, b=60),
    )
    
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