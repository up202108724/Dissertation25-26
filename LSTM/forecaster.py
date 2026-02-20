
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
from dataset import TimeSeriesDataset
# Import LSTM plotting utilities
from model_utils.plots import plot_loss_curves, plot_data_distribution, plot_forecast_comparison

from loguru import logger

logger.add("training.log", rotation="10 MB", retention="7 days", level="DEBUG")

class MultiProductForecaster:
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, batch_size: int = 32, num_epochs: int = 50, learning_rate: float = 0.001,
                 lookback_days: int = 30,  # Fixed lookback window
                 hidden_size: int = 128, num_layers: int = 2,
                 save_plots: bool = True, plots_base_dir: str = './training_plots',
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_log1p: bool = False,
                 disable_preprocessing: bool = False):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.lookback_days = lookback_days  # Fixed window size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.models = {}
        self.scalers = {}  # Will store full scalers_dict per product
        self.forecast_horizons = {}
        self.save_plots = save_plots
        self.plots_base_dir = plots_base_dir
        self.use_log1p = use_log1p  # Apply log1p to target for better handling of skewed sales data
        self.device = device  # device for model training and inference
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.disable_preprocessing = disable_preprocessing  # Skip preprocessing (use raw data)
        
    def calculate_splits(self, total_length: int) -> Tuple[int, int, int]:
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        test_length = total_length - val_end  # Forecast horizon
        return train_end, val_end, test_length
        
    def prepare_data(self, df: pd.DataFrame, store_id: int, item_id: int,
                    include_features: List[str] = None):
        # Filter to store-item
        product_data = df[
            (df['store_id'] == store_id) & (df['item_id'] == item_id)
        ].sort_values('date').reset_index(drop=True)
        
        total_length = len(product_data)
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        test_length = total_length - val_end  # Forecast horizon
        
        if self.save_plots:
            plot_data_distribution(product_data, store_id, item_id, train_end, val_end, self.plots_base_dir)
        
       
        key = (store_id, item_id)
        self.forecast_horizons[key] = test_length
        
      
        logger.info(f"Store {store_id}, Item {item_id} - Preprocessing DISABLED, using raw values")
        train_df = product_data.iloc[:train_end].copy()
        val_df = product_data.iloc[max(0, train_end - self.lookback_days):val_end].copy()
        test_df = product_data.copy()
            
      
        
        # Create datasets with fixed lookback window
        train_dataset = TimeSeriesDataset(
            train_df, lookback=self.lookback_days, store_id=store_id, item_id=item_id
        )
        
        val_dataset = TimeSeriesDataset(
            val_df, lookback=self.lookback_days, store_id=store_id, item_id=item_id
        )
        
        test_dataset = TimeSeriesDataset(
            test_df, lookback=self.lookback_days, store_id=store_id, item_id=item_id
        )
        
        print(f"  Total: {total_length} | Train: {train_end} | Val: {val_end-train_end} | Test: {test_length}")
        print(f"  One-step pairs - Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, None, test_length, product_data
    
    def train_product(self, store_id: int, item_id: int, 
                     train_dataset: TimeSeriesDataset, val_dataset: TimeSeriesDataset,
                     forecast_horizon: int):
        
        if train_dataset is None or len(train_dataset) == 0:
            logger.warning(f"Store {store_id}, Item {item_id} - No training data available")
            return None, None
        
        logger.info(f"Store {store_id}, Item {item_id} - Starting training")
        logger.debug(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}")
        logger.debug(f"  Forecast horizon: {forecast_horizon} steps")
        
        # Create data loaders 
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset and len(val_dataset) > 0 else None
        
        # Initialize model (univariate autoregressive: input_size=1, outputs single value)
        model = LSTMForecaster(
            input_size=1,  # Single value input
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model initialized: {self.num_layers} layers, {self.hidden_size} hidden units, {total_params:,} parameters")
        
        # Loss and optimizer
        # Using HuberLoss instead of MSELoss to be less sensitive to outliers
        # and reduce the tendency to predict the mean
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        logger.debug(f"  Optimizer: Adam (lr={self.learning_rate}, weight_decay=1e-5), Loss: HuberLoss")
        
        
        # Training loop with validation
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"  Starting training for {self.num_epochs} epochs ")
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            batch_losses = []
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                # Log batch info for first batch
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"  Batch structure:")
                    logger.info(f"    Total sequences in dataset: {len(train_dataset)}")
                    logger.info(f"    Batch size: {self.batch_size}")
                    logger.info(f"    Number of batches: {len(train_loader)}")
                    logger.info(f"    X_batch shape: {X_batch.shape} (batch_size, lookback, features)")
                    logger.info(f"    y_batch shape: {y_batch.shape} (batch_size, 1)")
                    logger.info(f"    ✓ Shape is CORRECT for univariate LSTM!")

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = model(X_batch)
                y_batch = y_batch.squeeze(-1)  # Ensure shape is (batch_size,)
                loss = criterion(outputs, y_batch)
                
                # Formatted logging
                logger.info(f"  Epoch {epoch+1}, Train Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                if batch_idx == 0 and epoch == 0:  # Only log first batch of first epoch in detail
                    x_np = X_batch.detach().cpu().numpy()
                    y_np = y_batch.detach().cpu().numpy()
                    out_np = outputs.detach().cpu().numpy()
                    
                    logger.debug(f"    Batch shapes: X={X_batch.shape}, y={y_batch.shape}, outputs={outputs.shape}")
                    logger.debug(f"    X_batch stats: min={x_np.min():.2f}, max={x_np.max():.2f}, mean={x_np.mean():.2f}, std={x_np.std():.2f}")
                    logger.debug(f"    X_batch[0] (first sequence, all values): {x_np[0, :, 0]}")
                    logger.debug(f"    X_batch[1] (second sequence, all values): {x_np[1, :, 0]}")
                    logger.debug(f"    y_batch (first 10): {y_np[:10]}")
                    logger.debug(f"    outputs (predictions): min={out_np.min():.2f}, max={out_np.max():.2f}, mean={out_np.mean():.2f}")
                    logger.debug(f"    outputs (first 10): {out_np[:10]}")
                    logger.debug(f"    errors (first 10): {np.abs(y_np[:10] - out_np[:10])}")
                
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                batch_losses.append(loss.item())
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Log batch loss statistics for first few epochs
            if epoch < 3:
                logger.debug(f"  Epoch {epoch+1} batch losses: min={min(batch_losses):.6f}, max={max(batch_losses):.6f}, std={np.std(batch_losses):.6f}")
            
            # Validation phase
            if val_loader:
                model.eval()
                total_val_loss = 0
                val_batch_losses = []
                
                with torch.no_grad():
                    for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = model(X_batch)
                        y_batch = y_batch.squeeze(-1)  # Ensure shape matches outputs
                        loss = criterion(outputs, y_batch)
                        
                        logger.info(f"  Epoch {epoch+1}, Val Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.6f}")
                        
                        if batch_idx == 0 and epoch == 0:  # Only log first batch of first epoch in detail
                            x_np = X_batch.cpu().numpy()
                            y_np = y_batch.cpu().numpy()
                            out_np = outputs.cpu().numpy()
                            
                            logger.debug(f"    Val batch shapes: X={X_batch.shape}, y={y_batch.shape}, outputs={outputs.shape}")
                            logger.debug(f"    Val X_batch stats: min={x_np.min():.2f}, max={x_np.max():.2f}, mean={x_np.mean():.2f}")
                            logger.debug(f"    Val y_batch (targets): min={y_np.min():.2f}, max={y_np.max():.2f}, mean={y_np.mean():.2f}")
                            logger.debug(f"    Val outputs (predictions): min={out_np.min():.2f}, max={out_np.max():.2f}, mean={out_np.mean():.2f}")
                            logger.debug(f"    Val errors (first 10): {np.abs(y_np[:10] - out_np[:10])}")

                        val_batch_losses.append(loss.item())
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Log validation batch statistics for first few epochs
                if epoch < 3:
                    logger.debug(f"  Epoch {epoch+1} val batch losses: min={min(val_batch_losses):.6f}, max={max(val_batch_losses):.6f}, std={np.std(val_batch_losses):.6f}")
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.6f}")
                    logger.info(f"  Epoch {epoch+1}/{self.num_epochs} - Train: {avg_train_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"  Loaded best model (val loss: {best_val_loss:.6f})")
        else:
            logger.info(f"  Using final model (no validation set)")
        
        # Training summary
        total_epochs = len(train_losses)
        loss_reduction = (1 - train_losses[-1] / train_losses[0]) * 100 if train_losses[0] > 0 else 0
        logger.info(f"  Training complete: {total_epochs} epochs, loss reduction: {loss_reduction:.1f}%")
        logger.debug(f"  Final train loss: {train_losses[-1]:.6f}, Initial: {train_losses[0]:.6f}")
        if val_losses:
            logger.debug(f"  Final val loss: {val_losses[-1]:.6f}, Best: {best_val_loss:.6f}")
        
        # Plot loss curves
        if self.save_plots:
            plot_loss_curves(train_losses, val_losses, store_id, item_id, self.plots_base_dir)
        
        # Store model
        key = (store_id, item_id)
        self.models[key] = model
        
        return model, {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict(self, store_id: int, item_id: int, 
                recent_data: pd.DataFrame, include_features: List[str] = None) -> np.ndarray:
        key = (store_id, item_id)
        
        if key not in self.models:
            print(f"No model found for store {store_id}, item {item_id}")
            return None
        
        model = self.models[key]
        forecast_horizon = self.forecast_horizons.get(key, 20)
        
        # Filter to product and sort by date
        product_data = recent_data[
            (recent_data['store_id'] == store_id) & 
            (recent_data['item_id'] == item_id)
        ].sort_values('date')
        
        if len(product_data) == 0:
            print(f"No data found for store {store_id}, item {item_id}")
            return None
        
        # Use raw data directly when preprocessing is disabled
        if self.disable_preprocessing:
            preprocessed_data = product_data
            value_std = product_data['value'].std()
        else:
            # Get scalers for preprocessing
            scalers_dict = self.scalers.get(key, {})
            preprocessed_data, _, _, _ = preprocess_features(
                product_data,
                fit_encoders=False,
                encoding_type='onehot',
                encoders=scalers_dict.get('encoders', {}),
                scalers=scalers_dict,
                onehot_categories=scalers_dict.get('onehot_categories', {}),
                use_log1p=self.use_log1p
            )
        
        # Extract values
        values_scaled = preprocessed_data['value'].values.astype(np.float32)

        total_length = len(values_scaled)
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        # Forecast starts at val_end (start of test set)
        forecast_start = val_end
        forecast_end = min(forecast_start + forecast_horizon, total_length)
        actual_test_steps = max(0, forecast_end - forecast_start)

        if forecast_start < self.lookback_days:
            print(f"Not enough history for lookback={self.lookback_days} at forecast start.")
            return None

        # Seed history from the window ending at the test start boundary
        history_scaled = values_scaled[forecast_start - self.lookback_days:forecast_start].tolist()

        logger.info(f"Store {store_id}, Item {item_id} - Starting forecast")
        logger.debug(f"Forecast start index: {forecast_start}, end index: {forecast_end}, steps: {actual_test_steps}")
        logger.debug(f"Initial history_scaled (boundary window): {history_scaled}")
        
        # Recursive forecasting: maintain sliding window of lookback_days
        forecast_scaled = []
        model.eval()
        
        with torch.no_grad():
            for step in range(forecast_horizon):
                # Create input: last lookback_days values
                history_array = np.array(history_scaled[-self.lookback_days:], dtype=np.float32)
                x_input = torch.FloatTensor(history_array).unsqueeze(0).unsqueeze(-1).to(self.device)  # Shape: (1, lookback, 1)
                
                logger.debug(f"Step {step+1}/{forecast_horizon}:")
                logger.debug(f"  Input tensor shape: {x_input.shape}")
                logger.debug(f"  Input tensor: min={x_input.min().item():.6f}, max={x_input.max().item():.6f}, mean={x_input.mean().item():.6f}")
                logger.debug(f"  Input values: {x_input.squeeze().cpu().numpy()}")
                
                # Predict next value
                output_tensor = model(x_input)
                next_value_scaled = output_tensor.item()
                
                # Add small noise during autoregressive prediction to prevent collapse
                # Only if using raw data (unscaled) to avoid adding too much noise to scaled data
                if self.disable_preprocessing and step > 0:
                   noise = np.random.normal(0, value_std * 0.05)  
                   next_value_scaled += noise
                
                logger.debug(f"  Output tensor: {output_tensor.item()}")
                logger.debug(f"  Predicted value (with noise?): {next_value_scaled:.6f}")
                
                forecast_scaled.append(next_value_scaled)
                
                # Add prediction to history (sliding window)
                history_scaled.append(next_value_scaled)
        
        # Convert to numpy array
        forecast_scaled = np.array(forecast_scaled)
        
        # Return raw values or inverse transform
        if self.disable_preprocessing:
            forecast = forecast_scaled
        else:
            scalers_dict = self.scalers.get(key, {})
            value_scaler = scalers_dict.get('value')
            forecast = inverse_transform_target(forecast_scaled, value_scaler, scalers_dict)
        
        return forecast
    
    def train_all_products(self, df: pd.DataFrame, store_id: int = None,
                          include_features: List[str] = None):
        
        if store_id is not None:
            product_pairs = [(store_id, item_id) for item_id in df[df['store_id'] == store_id]['item_id'].unique()]
        else:
            product_pairs = df.groupby(['store_id', 'item_id']).size().index.tolist()
        
        results = {}
        
        print(f"\n{'='*80}")
        print(f"Training LSTM with Fixed Lookback Window for {len(product_pairs)} products")
        print(f"Lookback: {self.lookback_days} days | Input shape: ({self.lookback_days}, 1)")
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
                s_id, i_id, train_ds, val_ds, forecast_horizon
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
                'forecast': forecast
            }
        
        return results