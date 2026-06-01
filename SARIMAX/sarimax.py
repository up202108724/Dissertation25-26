import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

class SARIMAXPipeline:
    """
    A class to encapsulate the SARIMAX forecasting pipeline,
    including seasonality analysis, candidate generation, and modeling
    (standard history approach).
    """
    def __init__(self, target_col='value', date_col='date'):
        self.target_col = target_col
        self.date_col = date_col

    def _prepare_data(self, df, item_id, store_id):
        """Filters and sorts data for a specific product and store."""
        subset = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
        if self.date_col in subset.columns:
            subset[self.date_col] = pd.to_datetime(subset[self.date_col])
            subset = subset.sort_values(self.date_col)
            # Add DatetimeIndex so statsmodels handles inference cleanly
            subset.index = pd.DatetimeIndex(subset[self.date_col])
            try:
                subset.index.freq = pd.infer_freq(subset.index) or 'D'
            except:
                subset.index.freq = 'D'
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
        """Analyzes ACF to automatically suggest 'm' candidates (e.g., 7 for weekly)."""
        subset = self._prepare_data(df, item_id, store_id)
        ts = subset[self.target_col].dropna().values
        n = len(ts)
        if n < max_lag:
            return [1]

        acf_values = acf(ts, nlags=max_lag, fft=True)
        threshold = 1.96 / np.sqrt(n)
        candidates = [1]
        
        potential_periods = [7] 
        for p in potential_periods:
            if p <= max_lag and abs(acf_values[p]) > threshold:
                candidates.append(p)
        
        if not candidates:
            candidates = [1]
            
        return candidates

    def fit_forecast(self, df, item_id, store_id, train_size=455, val_size=153,
                     forecast_window=153, m_candidates=None, exog_cols=None, maxiter=200):
        """
        Standard / Full History Approach:
        1. Tries every m in m_candidates via auto_arima on train-only data;
           picks the candidate whose model achieves the lowest AIC.
        2. Refits a SARIMAX model on train_size + val_size with the winning order.
        3. Forecasts the final forecast_window.
        """
        if m_candidates is None:
            m_candidates = [1]

        dfp = self._prepare_data(df, item_id, store_id)

        train_end = train_size
        val_end   = train_size + val_size

        y_train_only = dfp[self.target_col].iloc[0:train_end].astype(float).values
        y_train_val  = dfp[self.target_col].iloc[0:val_end].astype(float).values
        y_test       = dfp[self.target_col].iloc[val_end:val_end+forecast_window].astype(float).values

        if exog_cols:
            X_train_only = dfp[exog_cols].iloc[0:train_end].astype(float).values
            X_train_val  = dfp[exog_cols].iloc[0:val_end].astype(float).values
            X_test       = dfp[exog_cols].iloc[val_end:val_end+forecast_window].astype(float).values
        else:
            X_train_only, X_train_val, X_test = None, None, None

        # Try every candidate m; keep the model with the lowest AIC
        best_model = None
        best_aic   = np.inf
        best_m_used = 1
        for m in m_candidates:
            use_seasonal = m > 1
            try:
                candidate = auto_arima(
                    y_train_only, X=X_train_only,
                    seasonal=use_seasonal, m=m if use_seasonal else 1,
                    stepwise=True, suppress_warnings=True, error_action="ignore",
                    start_p=0, start_q=0, max_p=7, max_q=7, max_d=2,
                    start_P=0, start_Q=0, max_P=3, max_Q=3, max_D=1,
                    information_criterion='aic',
                )
                if candidate.aic() < best_aic:
                    best_aic   = candidate.aic()
                    best_model = candidate
                    best_m_used = m
            except Exception as e:
                print(f"    auto_arima failed for m={m}: {e}")

        if best_model is None:
            raise RuntimeError(f"auto_arima failed for all m candidates: {m_candidates}")

        order          = best_model.order
        seasonal_order = best_model.seasonal_order if hasattr(best_model, 'seasonal_order') else (0, 0, 0, 0)
        trend          = 'c' if best_model.with_intercept else 'n'
        print(f"  Selected m={best_m_used} (AIC={best_aic:.2f}) → order={order}, seasonal={seasonal_order}, trend={trend}")

        # Refit on train+val
        model = SARIMAX(
            y_train_val, exog=X_train_val, order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=maxiter, method='lbfgs')
        
        forecast = res.forecast(steps=forecast_window, exog=X_test)
        forecast = np.asarray(forecast)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, forecast)))
        mae  = float(mean_absolute_error(y_test, forecast))
        bias = float(np.mean(forecast - y_test))
        composite_score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
        return rmse, mae, bias, composite_score, order, seasonal_order, forecast, y_train_val, y_test