# preprocessing.py
"""
Missing Value Imputation Methods
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import DBSCAN
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