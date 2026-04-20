import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

class AutoARIMAPipeline:
    """
    A class to encapsulate the AutoARIMA forecasting pipeline,
    including seasonality analysis, candidate generation, and modeling
    (both lookback state and full history approaches).
    """
    def __init__(self, target_col='value', date_col='date'):
        self.target_col = target_col
        self.date_col = date_col

    def _prepare_data(self, df, item_id, store_id):
        """Filters and sorts data for a specific product and store."""
        subset = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
        if self.date_col in subset.columns:
            subset[self.date_col] = pd.to_datetime(subset[self.date_col])
            subset = subset.sort_values(self.date_col).reset_index(drop=True)
        else:
            subset = subset.reset_index(drop=True)
        return subset

    def analyze_seasonality(self, df, item_id, store_id, max_lag=60):
        """Plots ACF and PACF to help determine the optimal 'm' parameter."""
        subset = self._prepare_data(df, item_id, store_id)
        ts = subset[self.target_col].dropna()
        
        if len(ts) < max_lag:
            print(f"Not enough data for item {item_id} store {store_id}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(ts, lags=max_lag, ax=axes[0])
        axes[0].set_title(f"ACF (Autocorrelation) - Item {item_id}, Store {store_id}")
        axes[0].set_xlabel("Lag (Days)")
        
        plot_pacf(ts, lags=max_lag, ax=axes[1])
        axes[1].set_title(f"PACF (Partial Autocorrelation)")
        axes[1].set_xlabel("Lag (Days)")
        
        plt.show()
        print("Interpretation Guide:")
        print("- Spike at lag 7, 14, 21... -> Strong Weekly Seasonality (m=7)")
        print("- Spike at lag 30, 60...    -> Strong Monthly Seasonality (m=30)")
        print("- No clear pattern          -> Consider seasonal=False or m=1")

    def get_seasonal_candidates(self, df, item_id, store_id, max_lag=35):
        """Analyzes ACF to automatically suggest 'm' candidates (e.g., 7 for weekly, 30 for monthly)."""
        subset = self._prepare_data(df, item_id, store_id)
        ts = subset[self.target_col].dropna().values
        n = len(ts)
        if n < max_lag:
            return [1]

        acf_values = acf(ts, nlags=max_lag, fft=True)
        threshold = 1.96 / np.sqrt(n)
        candidates = [1]
        
        potential_periods = [7, 30] 
        for p in potential_periods:
            if p <= max_lag and abs(acf_values[p]) > threshold:
                candidates.append(p)
        
        if not candidates:
            candidates = [1]
            
        return candidates

    def fit_forecast_lookback(self, df, item_id, store_id, train_size=455, val_size=153, 
                              forecast_window=153, lookback_window=30, seasonal=False, m=1, 
                              exog_cols=None, maxiter=200):
        """
        APPROACH 1: "Rolling / Lookback State" (True Recursive State Propagation)
        1) Split data into train, val, and test.
        2) Use auto_arima to select parameters on train set.
        3) Refit on train+val.
        4) Recursively forecast 1 step at a time, appending the prediction to the SARIMAX state.
        """
        dfp = self._prepare_data(df, item_id, store_id)

        n = len(dfp)
        train_end = n - (val_size + forecast_window)
        val_end = n - forecast_window

        y_train = dfp[self.target_col].iloc[train_end - train_size:train_end].astype(float).values
        y_val   = dfp[self.target_col].iloc[train_end:val_end].astype(float).values
        y_train_val = dfp[self.target_col].iloc[train_end - train_size:val_end].astype(float).values
        y_test  = dfp[self.target_col].iloc[val_end:].astype(float).values

        if exog_cols:
            X_train = dfp[exog_cols].iloc[train_end - train_size:train_end].astype(float).values
            X_val   = dfp[exog_cols].iloc[train_end:val_end].astype(float).values
            X_train_val = dfp[exog_cols].iloc[train_end - train_size:val_end].astype(float).values
            X_test  = dfp[exog_cols].iloc[val_end:].astype(float).values
        else:
            X_train, X_val, X_train_val, X_test = None, None, None, None
        
        current_seasonal = True if (seasonal and m > 1) else False

        best_model = auto_arima(
            y_train, X=X_train, seasonal=current_seasonal, m=m if current_seasonal else 1, 
            stepwise=True, suppress_warnings=True, error_action="ignore",
            start_p=0, start_q=2, max_p=5, max_q=5, max_d=2,
            start_P=1, start_Q=1, max_P=2, max_Q=2, max_D=1
        )
        
        order = best_model.order
        seasonal_order = best_model.seasonal_order if hasattr(best_model, 'seasonal_order') else (0,0,0,0)
        trend = 'c' if best_model.with_intercept else 'n'
        print(f"Lookback Selected model: Order={order}, Seasonal={seasonal_order}, Trend={trend} (AIC={best_model.aic():.2f})")
        
        # Refit on Train + Val
        model = SARIMAX(
            y_train_val, exog=X_train_val, order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=maxiter, method='lbfgs')

        state = res
        forecast = []

        for i in range(forecast_window):
            next_exog = X_test[i:i+1] if X_test is not None else None
            pred = state.forecast(steps=1, exog=next_exog)[0]
            forecast.append(pred)

            # recursive future forecast: append the prediction
            state = state.append([pred], exog=next_exog, refit=False)

        forecast = np.asarray(forecast)
        rmse = float(np.sqrt(mean_squared_error(y_test, forecast)))
        mae  = float(mean_absolute_error(y_test, forecast))

        # Return y_train_val to match previous expected shape roughly for plotting
        return rmse, mae, order, seasonal_order, forecast, y_train_val, y_test

    def fit_forecast_full(self, df, item_id, store_id, train_size=455, val_size=153, 
                          forecast_window=153, seasonal=False, m=1, exog_cols=None, maxiter=200):
        """
        APPROACH 2: "Standard / Full History"
        Use all pre-test observations for training.
        """
        dfp = self._prepare_data(df, item_id, store_id)

        y_train = dfp[self.target_col].iloc[:-forecast_window].astype(float).values
        y_test  = dfp[self.target_col].iloc[-forecast_window:].astype(float).values

        if exog_cols:
            X_train = dfp[exog_cols].iloc[:-forecast_window].astype(float).values
            X_test  = dfp[exog_cols].iloc[-forecast_window:].astype(float).values
        else:
            X_train, X_test = None, None

        current_seasonal = True if (seasonal and m > 1) else False
        
        best_model = auto_arima(
            y_train, X=X_train, seasonal=current_seasonal, m=m if current_seasonal else 1,
            stepwise=True, suppress_warnings=True, error_action="ignore",
            start_p=0, start_q=2, max_p=5, max_q=5, max_d=2,
            start_P=1, start_Q=1, max_P=2, max_Q=2, max_D=1
        )
        
        order = best_model.order
        seasonal_order = best_model.seasonal_order if hasattr(best_model, 'seasonal_order') else (0,0,0,0)
        trend = 'c' if best_model.with_intercept else 'n'
        print(f"Full Selected model: Order={order}, Seasonal={seasonal_order}, Trend={trend} (AIC={best_model.aic():.2f})")

        model = SARIMAX(
            y_train, exog=X_train, order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=maxiter, method='lbfgs')
        
        forecast = res.forecast(steps=forecast_window, exog=X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, forecast)))
        mae  = float(mean_absolute_error(y_test, forecast))
        
        return rmse, mae, order, seasonal_order, forecast


