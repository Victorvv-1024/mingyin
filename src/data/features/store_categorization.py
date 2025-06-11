"""
Store categorization feature engineering for sales forecasting.

This module focuses on Chinese e-commerce store categorization including:
- Store type classification based on naming patterns
- Store performance indices
- One-hot encoding for store types
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from ..utils import setup_logging

logger = setup_logging()

class StoreCategorization:
    """Handles store categorization and related feature engineering."""
    
    def __init__(self):
        pass
    
    def categorize_store(self, store_name: str) -> str:
        """
        Categorize stores based on Chinese e-commerce store naming conventions.
        
        Args:
            store_name: Name of the store
            
        Returns:
            Store category string
        """
        if pd.isna(store_name):
            return 'Unknown'
        
        store_name = str(store_name)
        
        if '官方旗舰店' in store_name:
            return 'Official Flagship'
        elif '卖场旗舰店' in store_name:
            return 'Mall Flagship'
        elif '旗舰店' in store_name:
            return 'Flagship'
        elif '专卖店' in store_name or '专营店' in store_name:
            return 'Specialty Store'
        elif '卖场店' in store_name:
            return 'Mall Store'
        elif '自营' in store_name:
            return 'Platform Direct'
        elif '超市' in store_name:
            return 'Supermarket'
        else:
            return 'Other'
    
    def add_store_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add store type categorization based on Chinese e-commerce naming patterns."""
        logger.info("Adding store categorization...")
        
        df = df.copy()
        
        # Apply categorization
        df['store_type'] = df['store_name'].apply(self.categorize_store)
        
        # Analyze store type distribution
        store_type_analysis = df.groupby(['store_type', 'primary_platform']).agg({
            'sales_quantity': ['count', 'sum', 'mean'],
            'sales_amount': ['sum', 'mean'],
            'unit_price': 'mean',
            'store_name': 'nunique'
        }).round(2)
        
        store_type_analysis.columns = ['_'.join(col) for col in store_type_analysis.columns]
        logger.info("Store Type Analysis by Platform completed")
        
        # Create store type performance features
        store_type_performance = df.groupby('store_type').agg({
            'sales_quantity': 'mean',
            'unit_price': 'mean'
        })
        
        # Store type premium index (compared to average)
        avg_sales = df['sales_quantity'].mean()
        avg_price = df['unit_price'].mean()
        
        store_type_performance['sales_performance_index'] = store_type_performance['sales_quantity'] / avg_sales
        store_type_performance['price_premium_index'] = store_type_performance['unit_price'] / avg_price
        
        # Map back to main dataset
        df['store_type_sales_index'] = df['store_type'].map(store_type_performance['sales_performance_index']).fillna(1)
        df['store_type_price_index'] = df['store_type'].map(store_type_performance['price_premium_index']).fillna(1)
        
        # One-hot encode store types for modeling
        store_type_dummies = pd.get_dummies(df['store_type'], prefix='store_type')
        df = pd.concat([df, store_type_dummies], axis=1)
        
        logger.info(f"✓ Store categorization complete. Created {len(store_type_dummies.columns)} store type features")
        
        return df
    
    def prepare_modeling_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the dataset for advanced modeling by handling missing values and data quality."""
        logger.info("Preparing dataset for modeling...")
        
        df = df.copy()
        
        # Handle any remaining missing values in lag features (expected for early records)
        lag_columns = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        
        for col in lag_columns:
            if df[col].isnull().any():
                # Fill NaN lag values with appropriate defaults
                if 'lag_' in col:
                    # For lag features, use forward fill within groups, then 0
                    df[col] = df.groupby(['primary_platform', 'store_name', 'brand_name'])[col].ffill().fillna(0)
                elif 'rolling_mean' in col:
                    # For rolling means, use the current value
                    df[col] = df[col].fillna(df['sales_quantity'])
                elif 'rolling_std' in col:
                    # For rolling std, use 0 (no volatility)
                    df[col] = df[col].fillna(0)
                elif 'rolling_min' in col or 'rolling_max' in col:
                    # For rolling min/max, use current value
                    df[col] = df[col].fillna(df['sales_quantity'])
        
        # Handle any infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)
        
        logger.info("✓ Dataset prepared for modeling")
        return df
    
    def engineer_all_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all store categorization feature engineering."""
        logger.info("=== STORE CATEGORIZATION FEATURE ENGINEERING ===")
        
        original_cols = len(df.columns)
        
        # Apply store categorization and preparation
        df = self.add_store_categorization(df)
        df = self.prepare_modeling_dataset(df)
        
        new_cols = len(df.columns)
        logger.info(f"✓ Store categorization completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df 