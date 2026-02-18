
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import TimeSeriesDataset
from lstm import LSTMForecaster

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EDA.preprocessing import preprocess_features, inverse_transform_target
from dataset import TimeSeriesDataset, collate_variable_length
# Import LSTM plotting utilities
from model_utils.plots import plot_loss_curves, plot_data_distribution, plot_forecast_comparison


class MultiProductForecaster:
    """Forecaster for multiple products using recursive autoregressive LSTM."""
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2,
                 lookback_days: int = None,  # Ignored in autoregressive mode
                 hidden_size: int = 128, num_layers: int = 2,
                 save_plots: bool = True, plots_base_dir: str = './training_plots',
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_log1p: bool = False):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.models = {}
        self.scalers = {}  # Will store full scalers_dict per product
        self.forecast_horizons = {}
        self.save_plots = save_plots
        self.plots_base_dir = plots_base_dir
        self.use_log1p = use_log1p  # Apply log1p to target for better handling of skewed sales data
        self.device = device  # device for model training and inference
        
    def calculate_splits(self, total_length: int) -> Tuple[int, int, int]:
        """Calculate train/val/test split indices."""
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        test_length = total_length - val_end  # Forecast horizon
        return train_end, val_end, test_length
        
    def prepare_data(self, df: pd.DataFrame, store_id: int, item_id: int,
                    include_features: List[str] = None):
        """Prepare train/val/test datasets for autoregressive LSTM.
        
        Creates one-step-ahead prediction pairs: [t] -> [t+1]
        """
        
        # Filter to store-item
        product_data = df[
            (df['store_id'] == store_id) & (df['item_id'] == item_id)
        ].sort_values('date').reset_index(drop=True)
        
        total_length = len(product_data)
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        test_length = total_length - val_end  # Forecast horizon
        
        if train_end < 10:
            print(f"Skipping item {item_id}: insufficient training data (need >10, have {train_end})")
            return None, None, None, None, None, None
        
        # Plot data distribution before training (using original unscaled data)
        if self.save_plots:
            plot_data_distribution(product_data, store_id, item_id, train_end, val_end, self.plots_base_dir)
        
        # Store forecast horizon for this product
        key = (store_id, item_id)
        self.forecast_horizons[key] = test_length
        
        # ========== PROPER PREPROCESSING WITH TRAIN/VAL/TEST SPLIT ==========
        # Split RAW data first (before any preprocessing)
        train_raw = product_data.iloc[:train_end].copy()
        val_raw = product_data.iloc[:val_end].copy()  # Val includes train data
        test_raw = product_data.copy()  # Test includes all data
        
        # Preprocess: Fit on training data only
        train_df, encoders, scalers, onehot_cats = preprocess_features(
            train_raw,
            fit_encoders=True,
            encoding_type='onehot',  # or 'label' depending on preference
            use_log1p=self.use_log1p
        )
        
        # Transform validation and test data using training preprocessors
        val_df, _, _, _ = preprocess_features(
            val_raw,
            fit_encoders=False,
            encoding_type='onehot',
            encoders=encoders,
            scalers=scalers,
            onehot_categories=onehot_cats,
            use_log1p=self.use_log1p
        )
        
        test_df, _, _, _ = preprocess_features(
            test_raw,
            fit_encoders=False,
            encoding_type='onehot',
            encoders=encoders,
            scalers=scalers,
            onehot_categories=onehot_cats,
            use_log1p=self.use_log1p
        )
        
        # Store full scalers_dict for inverse transform during prediction
        # Contains 'value' scaler, 'use_log1p' flag, encoders, and onehot_categories
        scalers['encoders'] = encoders
        scalers['onehot_categories'] = onehot_cats
        self.scalers[key] = scalers
        
        # Create datasets (autoregressive: no lookback, no forecast_horizon params)
        train_dataset = TimeSeriesDataset(
            train_df, store_id=store_id, item_id=item_id
        )
        
        val_dataset = TimeSeriesDataset(
            val_df, store_id=store_id, item_id=item_id
        )
        
        test_dataset = TimeSeriesDataset(
            test_df, store_id=store_id, item_id=item_id
        )
        
        print(f"  Total: {total_length} | Train: {train_end} | Val: {val_end-train_end} | Test: {test_length}")
        print(f"  One-step pairs - Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, scalers, test_length, product_data
    
    def train_product(self, store_id: int, item_id: int, 
                     train_dataset: TimeSeriesDataset, val_dataset: TimeSeriesDataset,
                     forecast_horizon: int, num_epochs: int = 50, 
                     learning_rate: float = 0.001, batch_size: int = 32,
                     early_stopping_patience: int = 10):
        """Train LSTM model for single-step-ahead prediction."""
        
        if train_dataset is None or len(train_dataset) == 0:
            return None, None
        
        # Create data loaders with custom collate function for variable-length sequences
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_variable_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length) if val_dataset and len(val_dataset) > 0 else None
        
        # Initialize model (univariate autoregressive: input_size=1, outputs single value)
        model = LSTMForecaster(
            input_size=1,  # Single value input
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop with validation
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Plot loss curves
        if self.save_plots:
            plot_loss_curves(train_losses, val_losses, store_id, item_id, self.plots_base_dir)
        
        # Store model
        key = (store_id, item_id)
        self.models[key] = model
        
        return model, {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict(self, store_id: int, item_id: int, 
                recent_data: pd.DataFrame, include_features: List[str] = None) -> np.ndarray:
        """
        Generate recursive forecast using all available history.
        
        For each prediction step, uses ALL history from start up to current day.
        
        Parameters
        ----------
        store_id, item_id : int
            Product identifier
        recent_data : pd.DataFrame
            Full historical data for the product
        include_features : List[str], optional
            Ignored in univariate mode
        
        Returns
        -------
        np.ndarray
            Forecast (unscaled, original units) with length = forecast_horizon
        """
        key = (store_id, item_id)
        
        if key not in self.models:
            print(f"No model found for store {store_id}, item {item_id}")
            return None
        
        model = self.models[key]
        scalers_dict = self.scalers[key]
        value_scaler = scalers_dict['value']
        forecast_horizon = self.forecast_horizons.get(key, 20)
        
        # Filter to product and sort by date
        product_data = recent_data[
            (recent_data['store_id'] == store_id) & 
            (recent_data['item_id'] == item_id)
        ].sort_values('date')
        
        if len(product_data) == 0:
            print(f"No data found for store {store_id}, item {item_id}")
            return None
        
        # Preprocess all data
        preprocessed_data, _, _, _ = preprocess_features(
            product_data,
            fit_encoders=False,
            encoding_type='onehot',
            encoders=scalers_dict.get('encoders', {}),
            scalers=scalers_dict,
            onehot_categories=scalers_dict.get('onehot_categories', {}),
            use_log1p=self.use_log1p
        )
        
        # Extract scaled values - this will be our growing history
        history_scaled = preprocessed_data['value'].values.tolist()
        
        # Recursive forecasting: maintain all history
        forecast_scaled = []
        model.eval()
        
        with torch.no_grad():
            for step in range(forecast_horizon):
                # Create input: all history up to current step
                history_array = np.array(history_scaled, dtype=np.float32)
                x_input = torch.FloatTensor(history_array).unsqueeze(0).unsqueeze(-1)  # Shape: (1, history_len, 1)
                
                # Predict next value
                next_value_scaled = model(x_input).cpu().numpy()[0]
                forecast_scaled.append(next_value_scaled)
                
                # Add prediction to history for next iteration
                history_scaled.append(next_value_scaled)
        
        # Convert to numpy array
        forecast_scaled = np.array(forecast_scaled)
        
        # Inverse transform to original units
        forecast = inverse_transform_target(forecast_scaled, value_scaler, scalers_dict)
        
        return forecast
    
    def train_all_products(self, df: pd.DataFrame, store_id: int = None,
                          num_epochs: int = 50, learning_rate: float = 0.001,
                          include_features: List[str] = None):
        """Train models for all products in the dataset.
        
        Uses recursive autoregressive forecasting (one-step-ahead predictions).
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe with encoded/scaled features
        store_id : int, optional
            If provided, train only for this store
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Adam optimizer learning rate
        include_features : List[str], optional
            Ignored in univariate autoregressive mode
        """
        
        if store_id is not None:
            product_pairs = [(store_id, item_id) for item_id in df[df['store_id'] == store_id]['item_id'].unique()]
        else:
            product_pairs = df.groupby(['store_id', 'item_id']).size().index.tolist()
        
        results = {}
        
        print(f"\n{'='*80}")
        print(f"Training Recursive Autoregressive LSTM for {len(product_pairs)} products")
        print(f"Mode: One-step-ahead predictions (univariate)")
        print(f"Train {self.train_ratio:.0%} | Val {self.val_ratio:.0%} | Test {self.test_ratio:.0%}")
        if self.save_plots:
            print(f"Plots will be saved to: {self.plots_base_dir}")
        print(f"{'='*80}\n")
        
        for i, (s_id, i_id) in enumerate(product_pairs):
            print(f"[{i+1}/{len(product_pairs)}] Training model for Store {s_id}, Item {i_id}")
            
            # Prepare data
            result_tuple = self.prepare_data(df, s_id, i_id, include_features=include_features)
            
            if result_tuple[0] is None:
                continue
            
            train_ds, val_ds, test_ds, scaler, forecast_horizon, product_data = result_tuple
            
            # Train model with validation
            model, losses = self.train_product(
                s_id, i_id, train_ds, val_ds, forecast_horizon,
                num_epochs=num_epochs, 
                learning_rate=learning_rate
            )
            
            # Generate forecast and plot comparison
            if model is not None:
                forecast = self.predict(s_id, i_id, df, include_features=include_features)
                if forecast is not None and self.save_plots:
                    val_end = int(len(product_data) * (self.train_ratio + self.val_ratio))
                    plot_forecast_comparison(product_data, forecast, s_id, i_id, val_end, self.plots_base_dir)
            
            results[(s_id, i_id)] = {
                'model': model,
                'losses': losses,
                'train_size': len(train_ds) if train_ds else 0,
                'val_size': len(val_ds) if val_ds else 0,
                'test_size': len(test_ds) if test_ds else 0,
                'forecast_horizon': forecast_horizon
            }
        
        return results