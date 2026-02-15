# preprocessing.py
"""
Missing Value Imputation Methods
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ImputationMethods:
    """Collection of imputation methods for time series."""
    
    @staticmethod
    def zero(series: pd.Series, missing_indices: List[int]) -> pd.Series:
        """
        Imputation with zero.
        Simple but possibly misleading - sales don't naturally become zero 
        unless there's a stock out or very slow mover.
        """
        result = series.copy()
        result.iloc[missing_indices] = 0
        return result
    
    
    @staticmethod
    def previous_value(series: pd.Series, missing_indices: List[int]) -> pd.Series:
        """
        Imputation with the previous value (forward fill).
        Works well inside seasonal cycles, but risky across cycle boundaries.
        E.g., filling Monday with Sunday's peak value is misleading.
        """
        result = series.copy()
        result = result.fillna(method='ffill')
        return result
    
    
    @staticmethod
    def rolling_mean(series: pd.Series, missing_indices: List[int], 
                     window: int = 7) -> pd.Series:
        """
        Imputation with 7-day rolling mean.
        Smoothens strong seasonality effects - may or may not be beneficial 
        depending on the data structure.
        """
        result = series.copy()
        rolling = series.rolling(window=window, center=True, min_periods=1).mean()
        result.iloc[missing_indices] = rolling.iloc[missing_indices]
        return result
    
    
    @staticmethod
    def dbscan(store_items_dict: Dict[int, pd.Series], 
               item_to_impute: int,
               missing_indices: List[int],
               eps: float = 0.3,
               min_samples: int = 2) -> pd.Series:
        """
        Imputation via DBSCAN clustering with season-trend awareness.
        
        Our preferred method. Process:
        1. Perform STL decomposition on all time series in the store
        2. Cluster items by their seasonal and trend patterns (STL features)
        3. For each missing value, impute with the mean of cluster members 
           on that same date
        
        Parameters
        ----------
        store_items_dict : Dict[int, pd.Series]
            Dict mapping item_id to time series
        item_to_impute : int
            Item ID to impute
        missing_indices : List[int]
            Indices of missing values
        eps : float
            DBSCAN epsilon parameter
        min_samples : int
            DBSCAN min_samples parameter
        """
        # STL decomposition for all items
        decompositions = {}
        seasonal_feats = []
        trend_feats = []
        item_ids = []
        
        for item_id, series in store_items_dict.items():
            try:
                if series.sum() > 0:  # Skip if all zeros
                    stl = STL(series, seasonal=13, trend=None)
                    result = stl.fit()
                    decompositions[item_id] = {
                        'seasonal': result.seasonal,
                        'trend': result.trend,
                        'series': series
                    }
                    seasonal_feats.append(result.seasonal.std())
                    trend_feats.append(result.trend.std())
                    item_ids.append(item_id)
            except:
                pass
        
        if len(item_ids) < 2:
            # Fallback to zero if can't decompose
            result = store_items_dict[item_to_impute].copy()
            result.iloc[missing_indices] = 0
            return result
        
        # Create feature matrix (seasonal and trend std as features)
        features = np.column_stack([seasonal_feats, trend_feats])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        labels = clustering.labels_
        
        # Map item to cluster
        item_to_cluster = {item: label for item, label in zip(item_ids, labels)}
        cluster_of_target = item_to_cluster.get(item_to_impute, -1)
        
        # Find cluster members
        cluster_members = [item for item, cluster in item_to_cluster.items() 
                          if cluster == cluster_of_target]
        
        # Impute with cluster mean on same date
        result = store_items_dict[item_to_impute].copy()
        
        # Compute mean across cluster for each date
        cluster_data = pd.concat(
            [decompositions[item]['series'] for item in cluster_members],
            axis=1
        )
        cluster_mean = cluster_data.mean(axis=1)
        
        result.iloc[missing_indices] = cluster_mean.iloc[missing_indices]
        
        return result
    
    
    @staticmethod
    def hierarchical_dbscan(store_items_dict: Dict[int, pd.Series],
                           item_metadata: Dict[int, str],
                           item_to_impute: int,
                           missing_indices: List[int],
                           hierarchy_col: str = 'cat_label',
                           eps: float = 0.3,
                           min_samples: int = 2) -> pd.Series:
        """
        Upgraded DBSCAN: clusters created within each hierarchical level.
        
        Items from different categories/departments are necessarily in 
        different clusters, even if they have similar patterns.
        
        Parameters
        ----------
        store_items_dict : Dict[int, pd.Series]
            Dict mapping item_id to time series
        item_metadata : Dict[int, str]
            Dict mapping item_id to category/hierarchy value
        item_to_impute : int
            Item ID to impute
        missing_indices : List[int]
            Indices of missing values
        hierarchy_col : str
            The hierarchy level (e.g., 'cat_label', 'dep_label')
        eps : float
            DBSCAN epsilon parameter
        min_samples : int
            DBSCAN min_samples parameter
        """
        target_hierarchy = item_metadata.get(item_to_impute)
        
        # Filter to only items in same hierarchy level
        same_hierarchy_items = {
            item_id: series for item_id, series in store_items_dict.items()
            if item_metadata.get(item_id) == target_hierarchy
        }
        
        if len(same_hierarchy_items) < 2:
            # Fallback to zero if insufficient items in category
            result = store_items_dict[item_to_impute].copy()
            result.iloc[missing_indices] = 0
            return result
        
        # STL decomposition only within hierarchy
        decompositions = {}
        seasonal_feats = []
        trend_feats = []
        item_ids = []
        
        for item_id, series in same_hierarchy_items.items():
            try:
                if series.sum() > 0:
                    stl = STL(series, seasonal=13, trend=None)
                    result = stl.fit()
                    decompositions[item_id] = {
                        'seasonal': result.seasonal,
                        'trend': result.trend,
                        'series': series
                    }
                    seasonal_feats.append(result.seasonal.std())
                    trend_feats.append(result.trend.std())
                    item_ids.append(item_id)
            except:
                pass
        
        if len(item_ids) < 2:
            result = store_items_dict[item_to_impute].copy()
            result.iloc[missing_indices] = 0
            return result
        
        # Create feature matrix
        features = np.column_stack([seasonal_feats, trend_feats])
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # DBSCAN clustering within hierarchy
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        labels = clustering.labels_
        
        # Map item to cluster
        item_to_cluster = {item: label for item, label in zip(item_ids, labels)}
        cluster_of_target = item_to_cluster.get(item_to_impute, -1)
        
        # Find cluster members (within same hierarchy)
        cluster_members = [item for item, cluster in item_to_cluster.items() 
                          if cluster == cluster_of_target]
        
        # Impute with cluster mean
        result = store_items_dict[item_to_impute].copy()
        
        cluster_data = pd.concat(
            [decompositions[item]['series'] for item in cluster_members],
            axis=1
        )
        cluster_mean = cluster_data.mean(axis=1)
        
        result.iloc[missing_indices] = cluster_mean.iloc[missing_indices]
        
        return result


def preprocess_features(df: pd.DataFrame, 
                       fit_encoders: bool = True,
                       encoding_type: str = 'label',
                       encoders: Dict = None,
                       scalers: Dict = None,
                       onehot_categories: Dict = None,
                       use_log1p: bool = False) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
   
    df = df.copy()
    
    # Define feature groups
    categorical_cols = ['cat_label', 'sdep_label', 'dep_label', 'dmn_label']
    numerical_cols = ['value'] 
    promotional_cols =  [col for col in df.columns if col.startswith('promo_value_')]
    
    # Filter to columns that exist
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    promotional_cols = [col for col in promotional_cols if col in df.columns]

    if encoding_type not in ['label', 'onehot']:
        raise ValueError("encoding_type must be 'label' or 'onehot'")
    
    # Initialize containers for return values
    encoders_dict = {}
    scalers_dict = {}
    onehot_dict = {}
    
    # ========== CATEGORICAL ENCODING ==========
    if encoding_type == 'label':
        # Label Encoding
        if fit_encoders:
            encoders_dict = {}
            
            # Fit and transform categorical features
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = df[col].fillna('missing')
                df[col] = le.fit_transform(df[col])
                encoders_dict[col] = le
        else:
            if encoders is None:
                raise ValueError("encoders must be provided when fit_encoders=False with encoding_type='label'")
            
            # Transform categorical features
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                # Handle unseen categories
                known_labels = set(encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: encoders[col].transform([x])[0] if x in known_labels else -1
                )
    
    elif encoding_type == 'onehot':
        # One-hot Encoding
        if fit_encoders:
            onehot_dict = {}
            
            # Fit and transform each categorical column
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                onehot_dict[col] = df[col].unique().tolist()
                # Create one-hot encoded columns
                onehot_encoded = pd.get_dummies(df[[col]], prefix=col, drop_first=False)
                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_encoded], axis=1)
        else:
            if onehot_categories is None:
                raise ValueError("onehot_categories must be provided when fit_encoders=False with encoding_type='onehot'")
            
            # Transform with known categories
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                # Create one-hot with only known categories, unknown = all zeros
                onehot_encoded = pd.get_dummies(df[[col]], prefix=col, drop_first=False)
                
                # Ensure all expected columns exist
                expected_cols = [f"{col}_{cat}" for cat in onehot_categories[col]]
                for expected_col in expected_cols:
                    if expected_col not in onehot_encoded.columns:
                        onehot_encoded[expected_col] = 0
                
                # Keep only expected columns in correct order
                onehot_encoded = onehot_encoded[[col for col in expected_cols if col in onehot_encoded.columns]]
                
                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_encoded], axis=1)
    
    # ========== NUMERICAL SCALING ==========
    if fit_encoders:
        scalers_dict = {}
        
        # Store log1p flag for inverse transformation
        scalers_dict['use_log1p'] = use_log1p
        
        # Fit and transform numerical features
        for col in numerical_cols:
            scaler = StandardScaler()
            df[col] = df[col].fillna(0)
            
            # Apply log1p transformation if enabled (for 'value' column)
            if use_log1p and col == 'value':
                df[col] = np.log1p(df[col])
            
            df[col] = scaler.fit_transform(df[[col]]).flatten()
            scalers_dict[col] = scaler
    else:
        if scalers is None:
            raise ValueError("scalers must be provided when fit_encoders=False")
        
        # Get log1p flag from training scalers
        use_log1p_from_train = scalers.get('use_log1p', False)
        
        # Transform numerical features
        for col in numerical_cols:
            df[col] = df[col].fillna(0)
            
            # Apply log1p transformation if it was used in training
            if use_log1p_from_train and col == 'value':
                df[col] = np.log1p(df[col])
            
            df[col] = scalers[col].transform(df[[col]]).flatten()
    
    return df, encoders_dict, scalers_dict, onehot_dict


def inverse_transform_target(values: np.ndarray, scaler: StandardScaler, 
                             scalers_dict: Dict) -> np.ndarray:
   
    # Ensure 2D shape for scaler
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    # Inverse scale
    values_unscaled = scaler.inverse_transform(values).flatten()
    
    # Inverse log1p if it was applied
    if scalers_dict.get('use_log1p', False):
        values_unscaled = np.expm1(values_unscaled)
    
    return values_unscaled



