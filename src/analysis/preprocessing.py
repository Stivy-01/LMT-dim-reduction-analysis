"""
Preprocessing Module
==================

Handles data preprocessing and feature selection for behavioral analysis.
Includes:
- Data cleaning and normalization
- Feature selection
- Correlation filtering
- Standardization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit

class BehaviorPreprocessor:
    """
    Preprocesses behavioral data for analysis.
    Handles data cleaning, normalization, and feature selection.
    """
    
    def __init__(self, correlation_threshold=0.95, variance_threshold=0.1):
        """
        Initialize preprocessor.
        
        Args:
            correlation_threshold: Threshold for removing highly correlated features
            variance_threshold: Threshold for removing low variance features
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.qt = QuantileTransformer(output_distribution='normal')
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        self.feature_mask = None
        self.dropped_features = None
        self.feature_names = None
        
    def _handle_datetime(self, df):
        """Handle datetime columns and ensure proper interval structure."""
        if 'interval_start' in df.columns:
            df['interval_start'] = pd.to_datetime(df['interval_start'])
            df['interval_id'] = df['interval_start'].dt.strftime('%Y%m%d_%H')
            df['timestamp'] = df['interval_start'].astype('int64') // 10**9
        return df
    
    def _normalize_by_interval(self, df, feature_cols):
        """
        Normalize features within each interval.
        
        Args:
            df: DataFrame with interval_id column
            feature_cols: List of behavioral feature columns
            
        Returns:
            DataFrame with normalized features
        """
        normalized_df = df.copy()
        
        # Group by interval and normalize each feature
        for interval_id in df['interval_id'].unique():
            interval_mask = df['interval_id'] == interval_id
            interval_data = df.loc[interval_mask, feature_cols]
            
            # Quantile normalization per interval
            normalized = self.qt.fit_transform(interval_data)
            normalized_df.loc[interval_mask, feature_cols] = normalized
            
        return normalized_df
    
    def _clean_data(self, df):
        """Clean and prepare data."""
        # Get behavioral feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['mouse_id', 'date', 'interval_start', 'interval_id', 'timestamp']]
        
        # Convert to numeric
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Normalize within each interval
        df_normalized = self._normalize_by_interval(df, feature_cols)
        
        # Extract features and IDs
        X = df_normalized[feature_cols].copy()
        
        # Fill any remaining NaN with interval-specific means
        for interval_id in df['interval_id'].unique():
            interval_mask = df['interval_id'] == interval_id
            X.loc[interval_mask] = X.loc[interval_mask].fillna(
                X.loc[interval_mask].mean()
            )
        
        return X, df['mouse_id'], df['interval_id']
    
    def _remove_correlated_features(self, X):
        """Remove highly correlated features."""
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        self.dropped_features = to_drop
        X_filtered = np.delete(X, [list(upper.columns).index(c) for c in to_drop], axis=1)
        return X_filtered
    
    def _create_cross_validation_splits(self, interval_ids, n_splits=5):
        """
        Create time-series aware cross-validation splits.
        
        Args:
            interval_ids: Series of interval identifiers
            n_splits: Number of cross-validation splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        unique_intervals = np.unique(interval_ids)
        
        splits = []
        for train_idx, test_idx in tscv.split(unique_intervals):
            train_intervals = unique_intervals[train_idx]
            test_intervals = unique_intervals[test_idx]
            
            train_mask = interval_ids.isin(train_intervals)
            test_mask = interval_ids.isin(test_intervals)
            
            splits.append((
                np.where(train_mask)[0],
                np.where(test_mask)[0]
            ))
        
        return splits
    
    def fit_transform(self, df):
        """
        Preprocess data for analysis.
        
        Args:
            df: DataFrame containing behavioral data
            
        Returns:
            tuple: (preprocessed_features, mouse_ids, cv_splits)
        """
        # Handle datetime and intervals
        df = self._handle_datetime(df)
        
        # Clean and normalize data
        X, mouse_ids, interval_ids = self._clean_data(df)
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy array
        X = X.values
        
        # Remove low variance features
        X_var = self.variance_selector.fit_transform(X)
        self.feature_mask = self.variance_selector.get_support()
        
        # Remove highly correlated features
        X_final = self._remove_correlated_features(X_var)
        
        # Create cross-validation splits
        cv_splits = self._create_cross_validation_splits(interval_ids)
        
        print(f"Preprocessing summary:")
        print(f"- Original features: {X.shape[1]}")
        print(f"- After variance threshold: {X_var.shape[1]}")
        print(f"- After correlation filtering: {X_final.shape[1]}")
        print(f"- Number of intervals: {len(np.unique(interval_ids))}")
        print(f"- Cross-validation splits: {len(cv_splits)}")
        
        return X_final, mouse_ids, cv_splits
    
    def get_feature_names(self):
        """Get names of selected features."""
        if self.feature_names is None:
            return None
        
        selected_features = np.array(self.feature_names)[self.feature_mask]
        final_features = [f for f in selected_features if f not in self.dropped_features]
        
        return final_features

def preprocess_data(df, correlation_threshold=0.95, variance_threshold=0.1):
    """
    Convenience function for preprocessing behavioral data.
    
    Args:
        df: DataFrame containing behavioral data
        correlation_threshold: Threshold for removing highly correlated features
        variance_threshold: Threshold for removing low variance features
        
    Returns:
        tuple: (preprocessed_features, mouse_ids, cv_splits, feature_names)
    """
    preprocessor = BehaviorPreprocessor(
        correlation_threshold=correlation_threshold,
        variance_threshold=variance_threshold
    )
    
    X, mouse_ids, cv_splits = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()
    
    return X, mouse_ids, cv_splits, feature_names 