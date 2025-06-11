"""
Feature processor for deep learning models.

This module handles the preparation and categorization of features for 
embedding-based neural networks, matching the sophisticated approach
from full_data_prediction.ipynb.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """
    Advanced feature processor for embedding-based deep learning models.
    
    Handles feature categorization, encoding, and preparation for multi-input
    neural networks with different embedding strategies.
    """
    
    def __init__(self):
        """Initialize the feature processor."""
        self.encoders = {}
        self.scalers = {}
        self.feature_categories = {}
        self.data_shapes = {}
        self.is_fitted = False
        logger.info("FeatureProcessor initialized")
    
    def categorize_features_for_embeddings(self, df: pd.DataFrame, features: List[str]) -> Dict[str, List[str]]:
        """
        Analyze and categorize features for embedding strategies.
        
        Args:
            df: DataFrame with features
            features: List of feature names to categorize
            
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        logger.info("Categorizing features for embedding strategies...")
        
        feature_categories = {
            'temporal_categorical': [],
            'temporal_continuous': [],
            'cyclical': [],
            'promotional_binary': [],
            'promotional_continuous': [],
            'lag_continuous': [],
            'rolling_continuous': [],
            'customer_behavior_continuous': [],
            'store_categorical': [],
            'platform_categorical': [],
            'binary_features': [],
            'count_features': [],
            'ratio_features': [],
            'score_features': [],
            'interaction_features': []
        }
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # Temporal categorical (for embeddings)
            if feature in ['month', 'quarter', 'year']:
                feature_categories['temporal_categorical'].append(feature)
            
            # Temporal continuous (direct input)
            elif any(x in feature for x in ['days_since', '_progress', 'tenure_months', 'day_of_year', 'week_of_year']):
                feature_categories['temporal_continuous'].append(feature)
            
            # Cyclical features (sin/cos - direct input, no scaling needed)
            elif any(x in feature for x in ['_sin', '_cos']):
                feature_categories['cyclical'].append(feature)
            
            # Promotional binary features
            elif any(x in feature for x in ['is_promotional', 'is_singles_day', 'is_chinese_new_year', 'is_618', 'is_holiday']):
                feature_categories['promotional_binary'].append(feature)
            
            # Promotional continuous features
            elif any(x in feature for x in ['promotional_intensity', 'days_to_next_promo', 'days_from_last_promo']):
                feature_categories['promotional_continuous'].append(feature)
            
            # Lag features (continuous)
            elif '_lag_' in feature:
                feature_categories['lag_continuous'].append(feature)
            
            # Rolling window features (continuous)
            elif 'rolling_' in feature:
                feature_categories['rolling_continuous'].append(feature)
            
            # Customer behavior continuous
            elif any(x in feature for x in ['store_sales_', 'brand_', 'market_share', 'diversity', 'relationship_']):
                feature_categories['customer_behavior_continuous'].append(feature)
            
            # Store categorical (for embeddings)
            elif feature.startswith('store_type_') or any(x in feature for x in ['store_quality_', 'bm_']):
                feature_categories['store_categorical'].append(feature)
            
            # Platform categorical 
            elif any(x in feature for x in ['platform_', 'cross_platform', 'multi_platform']):
                feature_categories['platform_categorical'].append(feature)
            
            # Binary features (0/1)
            elif (df[feature].dtype in ['bool'] or 
                  (df[feature].dtype in ['int32', 'int64'] and df[feature].max() <= 1 and df[feature].min() >= 0)):
                feature_categories['binary_features'].append(feature)
            
            # Count features
            elif any(x in feature for x in ['_count', '_nunique', '_rank']):
                feature_categories['count_features'].append(feature)
            
            # Ratio/percentage features
            elif any(x in feature for x in ['_ratio', '_share', '_cv', '_pct', '_percentage']):
                feature_categories['ratio_features'].append(feature)
            
            # Score features
            elif any(x in feature for x in ['_score', '_index']):
                feature_categories['score_features'].append(feature)
            
            # Interaction features
            elif 'interaction' in feature:
                feature_categories['interaction_features'].append(feature)
        
        # Filter out empty categories
        self.feature_categories = {k: v for k, v in feature_categories.items() if v}
        
        # Log categorization results
        total_categorized = sum(len(features) for features in self.feature_categories.values())
        logger.info(f"Feature categorization completed:")
        for category, features in self.feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")
        logger.info(f"Total categorized: {total_categorized}")
        
        return self.feature_categories
    
    def prepare_data_for_training(self, df: pd.DataFrame, 
                                features: List[str], 
                                is_training: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for multi-input neural network training.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            is_training: Whether this is training data (for fitting scalers/encoders)
            
        Returns:
            Tuple of (prepared_data_dict, target_array)
        """
        logger.info(f"Preparing data for {'training' if is_training else 'inference'}...")
        
        df_work = df.copy()
        prepared_data = {}
        
        # Categorize features if not done yet
        if not self.feature_categories:
            self.categorize_features_for_embeddings(df_work, features)
        
        # Process each feature category
        
        # 1. Temporal categorical features (for embeddings)
        if 'temporal_categorical' in self.feature_categories:
            temporal_categorical_data = []
            for feature in self.feature_categories['temporal_categorical']:
                if feature in df_work.columns:
                    if feature == 'month':
                        # Months 1-12 -> 0-11 for embedding
                        values = df_work[feature].fillna(1).astype(int)
                        values = np.clip(values, 1, 12) - 1
                    elif feature == 'quarter':
                        # Quarters 1-4 -> 0-3 for embedding
                        values = df_work[feature].fillna(1).astype(int)
                        values = np.clip(values, 1, 4) - 1
                    elif feature == 'year':
                        # Years -> 0-based indexing
                        if is_training:
                            unique_years = sorted(df_work[feature].dropna().unique())
                            self.encoders[f'{feature}_mapping'] = {year: idx for idx, year in enumerate(unique_years)}
                        
                        year_mapping = self.encoders.get(f'{feature}_mapping', {})
                        values = df_work[feature].map(year_mapping).fillna(0).astype(int)
                    else:
                        values = df_work[feature].fillna(0).astype(int)
                    
                    temporal_categorical_data.append(values)
            
            if temporal_categorical_data:
                prepared_data['temporal_categorical'] = np.column_stack(temporal_categorical_data)
                self.data_shapes['temporal_categorical'] = len(temporal_categorical_data)
        
        # 2. Temporal continuous features (standardized)
        temporal_continuous_features = (
            self.feature_categories.get('temporal_continuous', []) +
            self.feature_categories.get('promotional_continuous', [])
        )
        
        if temporal_continuous_features:
            existing_features = [f for f in temporal_continuous_features if f in df_work.columns]
            if existing_features:
                temporal_continuous_data = df_work[existing_features].values.astype(np.float32)
                temporal_continuous_data = np.nan_to_num(temporal_continuous_data, nan=0.0, posinf=1e6, neginf=-1e6)
                
                if is_training:
                    self.scalers['temporal_continuous'] = StandardScaler()
                    temporal_continuous_data = self.scalers['temporal_continuous'].fit_transform(temporal_continuous_data)
                else:
                    temporal_continuous_data = self.scalers['temporal_continuous'].transform(temporal_continuous_data)
                
                prepared_data['temporal_continuous'] = temporal_continuous_data
                self.data_shapes['temporal_continuous'] = len(existing_features)
        
        # 3. Cyclical features (no scaling needed)
        if 'cyclical' in self.feature_categories:
            existing_features = [f for f in self.feature_categories['cyclical'] if f in df_work.columns]
            if existing_features:
                cyclical_data = df_work[existing_features].values.astype(np.float32)
                cyclical_data = np.nan_to_num(cyclical_data, nan=0.0)
                
                prepared_data['cyclical'] = cyclical_data
                self.data_shapes['cyclical'] = len(existing_features)
        
        # 4. Binary features (no scaling needed)
        binary_features = (
            self.feature_categories.get('promotional_binary', []) +
            self.feature_categories.get('binary_features', []) +
            self.feature_categories.get('store_categorical', []) +
            self.feature_categories.get('platform_categorical', [])
        )
        
        if binary_features:
            existing_features = [f for f in binary_features if f in df_work.columns]
            if existing_features:
                binary_data = df_work[existing_features].values.astype(np.float32)
                binary_data = np.nan_to_num(binary_data, nan=0.0)
                
                prepared_data['binary'] = binary_data
                self.data_shapes['binary'] = len(existing_features)
        
        # 5. Continuous features (robust scaling)
        continuous_features = (
            self.feature_categories.get('lag_continuous', []) +
            self.feature_categories.get('rolling_continuous', []) +
            self.feature_categories.get('customer_behavior_continuous', []) +
            self.feature_categories.get('count_features', []) +
            self.feature_categories.get('ratio_features', []) +
            self.feature_categories.get('score_features', []) +
            self.feature_categories.get('interaction_features', [])
        )
        
        if continuous_features:
            existing_features = [f for f in continuous_features if f in df_work.columns]
            if existing_features:
                continuous_data = df_work[existing_features].values.astype(np.float32)
                continuous_data = np.nan_to_num(continuous_data, nan=0.0, posinf=1e6, neginf=-1e6)
                
                if is_training:
                    self.scalers['continuous'] = RobustScaler()
                    continuous_data = self.scalers['continuous'].fit_transform(continuous_data)
                else:
                    continuous_data = self.scalers['continuous'].transform(continuous_data)
                
                prepared_data['continuous'] = continuous_data
                self.data_shapes['continuous'] = len(existing_features)
        
        # 6. Target variable
        target = None
        if 'sales_quantity_log' in df_work.columns:
            target = df_work['sales_quantity_log'].values.astype(np.float32)
            target = np.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-1.0)
        
        if is_training:
            self.is_fitted = True
        
        logger.info(f"Data preparation completed:")
        for input_name, data in prepared_data.items():
            logger.info(f"  {input_name}: {data.shape}")
        logger.info(f"  target: {target.shape if target is not None else 'None'}")
        
        return prepared_data, target
    
    def get_embedding_configs(self) -> Dict[str, Dict]:
        """
        Get embedding configurations for categorical features.
        
        Returns:
            Dictionary with embedding configurations
        """
        embedding_configs = {}
        
        if 'temporal_categorical' in self.feature_categories:
            temporal_features = self.feature_categories['temporal_categorical']
            
            for i, feature in enumerate(temporal_features):
                if feature == 'month':
                    embedding_configs[f'temporal_categorical_{i}'] = {
                        'vocab_size': 12,
                        'embedding_dim': 8,
                        'feature_name': feature
                    }
                elif feature == 'quarter':
                    embedding_configs[f'temporal_categorical_{i}'] = {
                        'vocab_size': 4,
                        'embedding_dim': 4,
                        'feature_name': feature
                    }
                elif feature == 'year':
                    year_mapping = self.encoders.get(f'{feature}_mapping', {})
                    vocab_size = len(year_mapping) if year_mapping else 10
                    embedding_configs[f'temporal_categorical_{i}'] = {
                        'vocab_size': vocab_size,
                        'embedding_dim': min(8, vocab_size),
                        'feature_name': feature
                    }
                else:
                    embedding_configs[f'temporal_categorical_{i}'] = {
                        'vocab_size': 50,  # Default for other categorical
                        'embedding_dim': 8,
                        'feature_name': feature
                    }
        
        return embedding_configs
    
    def get_input_shapes(self) -> Dict[str, int]:
        """Get input shapes for model architecture."""
        return self.data_shapes.copy()
    
    def inverse_transform_target(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert log-transformed predictions back to original scale.
        
        Args:
            predictions: Log-transformed predictions
            
        Returns:
            Predictions in original scale
        """
        return np.expm1(predictions)
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        importance_groups = {
            'Temporal': (
                self.feature_categories.get('temporal_categorical', []) +
                self.feature_categories.get('temporal_continuous', []) +
                self.feature_categories.get('cyclical', [])
            ),
            'Promotional': (
                self.feature_categories.get('promotional_binary', []) +
                self.feature_categories.get('promotional_continuous', [])
            ),
            'Historical': (
                self.feature_categories.get('lag_continuous', []) +
                self.feature_categories.get('rolling_continuous', [])
            ),
            'Customer_Behavior': self.feature_categories.get('customer_behavior_continuous', []),
            'Store_Platform': (
                self.feature_categories.get('store_categorical', []) +
                self.feature_categories.get('platform_categorical', [])
            ),
            'Interactions': self.feature_categories.get('interaction_features', [])
        }
        
        return {k: v for k, v in importance_groups.items() if v}
    
    def validate_prepared_data(self, prepared_data: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Validate prepared data for common issues.
        
        Args:
            prepared_data: Dictionary of prepared data arrays
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'has_nan': {},
            'has_inf': {},
            'data_ranges': {},
            'shape_consistency': True
        }
        
        for input_name, data in prepared_data.items():
            # Check for NaN values
            has_nan = np.isnan(data).any()
            validation_results['has_nan'][input_name] = has_nan
            
            # Check for infinite values
            has_inf = np.isinf(data).any()
            validation_results['has_inf'][input_name] = has_inf
            
            # Data ranges
            validation_results['data_ranges'][input_name] = {
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std())
            }
        
        # Check shape consistency
        shapes = [data.shape[0] for data in prepared_data.values()]
        validation_results['shape_consistency'] = len(set(shapes)) <= 1
        
        # Log validation results
        issues_found = []
        for input_name in prepared_data.keys():
            if validation_results['has_nan'][input_name]:
                issues_found.append(f"{input_name}: NaN values")
            if validation_results['has_inf'][input_name]:
                issues_found.append(f"{input_name}: Infinite values")
        
        if not validation_results['shape_consistency']:
            issues_found.append("Inconsistent shapes across inputs")
        
        if issues_found:
            logger.warning(f"Data validation issues found: {issues_found}")
        else:
            logger.info("✓ Data validation passed")
        
        validation_results['validation_passed'] = len(issues_found) == 0
        
        return validation_results