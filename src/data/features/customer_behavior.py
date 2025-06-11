"""
Customer behavior and store analytics feature engineering.

This module implements sophisticated customer behavior analysis including:
- Store consistency and volatility metrics
- Brand and product diversity calculations
- Market share and competitive analysis
- Customer loyalty and retention indicators
- Store performance categorization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class CustomerBehaviorFeatureEngineer:
    """Comprehensive customer behavior and store analytics feature engineering."""
    
    def __init__(self):
        """Initialize the customer behavior feature engineer."""
        self.label_encoders = {}
        logger.info("CustomerBehaviorFeatureEngineer initialized")
    
    def engineer_all_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all customer behavior features in the correct sequence.
        
        Args:
            df: DataFrame with sales data
            
        Returns:
            DataFrame with comprehensive customer behavior features
        """
        logger.info("Starting comprehensive customer behavior feature engineering...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Ensure required columns exist
        if 'unit_price' not in df.columns and 'sales_amount' in df.columns and 'sales_quantity' in df.columns:
            df['unit_price'] = df['sales_amount'] / df['sales_quantity']
            df['unit_price'] = df['unit_price'].replace([np.inf, -np.inf], np.nan)
            df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        
        # Step 1: Store behavior analysis
        df = self._create_store_behavior_features(df)
        
        # Step 2: Brand analysis features
        df = self._create_brand_analysis_features(df)
        
        # Step 3: Product diversity features
        df = self._create_product_diversity_features(df)
        
        # Step 4: Market share and competitive features
        df = self._create_market_share_features(df)
        
        # Step 5: Customer loyalty indicators
        df = self._create_loyalty_features(df)
        
        # Step 6: Performance categorization
        df = self._create_performance_categories(df)
        
        new_cols = len(df.columns)
        logger.info(f"Customer behavior feature engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df
    
    def _create_store_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store behavior consistency and volatility metrics."""
        logger.info("Creating store behavior features...")
        
        # Sort for proper time series calculations
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        # Store-level aggregations
        store_stats = df.groupby(['primary_platform', 'store_name']).agg({
            'sales_quantity': ['mean', 'std', 'min', 'max', 'count'],
            'sales_amount': ['mean', 'std'],
            'unit_price': ['mean', 'std'],
            'brand_name': 'nunique',
            'product_code': 'nunique'
        }).reset_index()
        
        # Flatten column names
        store_stats.columns = [
            'primary_platform', 'store_name',
            'store_sales_mean', 'store_sales_std', 'store_sales_min', 'store_sales_max', 'store_record_count',
            'store_amount_mean', 'store_amount_std',
            'store_price_mean', 'store_price_std',
            'store_brand_count', 'store_product_count'
        ]
        
        # Calculate derived metrics
        store_stats['store_sales_cv'] = store_stats['store_sales_std'] / (store_stats['store_sales_mean'] + 1)
        store_stats['store_sales_range'] = store_stats['store_sales_max'] - store_stats['store_sales_min']
        store_stats['store_price_cv'] = store_stats['store_price_std'] / (store_stats['store_price_mean'] + 1)
        
        # Store consistency categories
        store_stats['store_consistency_high'] = (store_stats['store_sales_cv'] < 0.5).astype(int)
        store_stats['store_consistency_medium'] = ((store_stats['store_sales_cv'] >= 0.5) & (store_stats['store_sales_cv'] < 1.0)).astype(int)
        store_stats['store_consistency_low'] = (store_stats['store_sales_cv'] >= 1.0).astype(int)
        
        # Store size categories based on average sales
        sales_quantiles = store_stats['store_sales_mean'].quantile([0.33, 0.67])
        store_stats['store_size_small'] = (store_stats['store_sales_mean'] <= sales_quantiles.iloc[0]).astype(int)
        store_stats['store_size_medium'] = ((store_stats['store_sales_mean'] > sales_quantiles.iloc[0]) & 
                                          (store_stats['store_sales_mean'] <= sales_quantiles.iloc[1])).astype(int)
        store_stats['store_size_large'] = (store_stats['store_sales_mean'] > sales_quantiles.iloc[1]).astype(int)
        
        # Merge back to main dataframe
        df = df.merge(store_stats, on=['primary_platform', 'store_name'], how='left')
        
        logger.info("✓ Store behavior features created")
        return df
    
    def _create_brand_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create brand-level analysis features."""
        logger.info("Creating brand analysis features...")
        
        # Brand-level aggregations across all platforms and stores
        brand_stats = df.groupby('brand_name').agg({
            'sales_quantity': ['mean', 'std', 'sum'],
            'sales_amount': ['mean', 'sum'],
            'unit_price': ['mean', 'std'],
            'store_name': 'nunique',
            'primary_platform': 'nunique',
            'product_code': 'nunique'
        }).reset_index()
        
        # Flatten column names
        brand_stats.columns = [
            'brand_name',
            'brand_sales_mean', 'brand_sales_std', 'brand_sales_total',
            'brand_amount_mean', 'brand_amount_total',
            'brand_price_mean', 'brand_price_std',
            'brand_store_count', 'brand_platform_count', 'brand_product_count'
        ]
        
        # Calculate total market for market share calculations
        total_market_sales = df['sales_quantity'].sum()
        total_market_amount = df['sales_amount'].sum()
        
        # Brand market share
        brand_stats['brand_market_share_quantity'] = brand_stats['brand_sales_total'] / total_market_sales
        brand_stats['brand_market_share_amount'] = brand_stats['brand_amount_total'] / total_market_amount
        
        # Brand diversity metrics
        brand_stats['brand_diversity_stores'] = brand_stats['brand_store_count'] / df['store_name'].nunique()
        brand_stats['brand_diversity_platforms'] = brand_stats['brand_platform_count'] / df['primary_platform'].nunique()
        
        # Brand pricing strategy
        overall_mean_price = df['unit_price'].mean()
        brand_stats['brand_premium_indicator'] = (brand_stats['brand_price_mean'] > overall_mean_price * 1.2).astype(int)
        brand_stats['brand_budget_indicator'] = (brand_stats['brand_price_mean'] < overall_mean_price * 0.8).astype(int)
        brand_stats['brand_mid_range_indicator'] = ((brand_stats['brand_price_mean'] >= overall_mean_price * 0.8) & 
                                                   (brand_stats['brand_price_mean'] <= overall_mean_price * 1.2)).astype(int)
        
        # Brand performance categories
        sales_quantiles = brand_stats['brand_sales_mean'].quantile([0.25, 0.75])
        brand_stats['brand_performance_low'] = (brand_stats['brand_sales_mean'] <= sales_quantiles.iloc[0]).astype(int)
        brand_stats['brand_performance_medium'] = ((brand_stats['brand_sales_mean'] > sales_quantiles.iloc[0]) & 
                                                  (brand_stats['brand_sales_mean'] <= sales_quantiles.iloc[1])).astype(int)
        brand_stats['brand_performance_high'] = (brand_stats['brand_sales_mean'] > sales_quantiles.iloc[1]).astype(int)
        
        # Merge back to main dataframe
        df = df.merge(brand_stats, on='brand_name', how='left')
        
        logger.info("✓ Brand analysis features created")
        return df
    
    def _create_product_diversity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product diversity and catalog features."""
        logger.info("Creating product diversity features...")
        
        # Store-brand level product diversity
        store_brand_diversity = df.groupby(['primary_platform', 'store_name', 'brand_name']).agg({
            'product_code': 'nunique',
            'sales_quantity': 'sum',
            'sales_amount': 'sum'
        }).reset_index()
        
        store_brand_diversity.columns = [
            'primary_platform', 'store_name', 'brand_name',
            'product_diversity_count', 'sb_total_quantity', 'sb_total_amount'
        ]
        
        # Product diversity categories
        diversity_quantiles = store_brand_diversity['product_diversity_count'].quantile([0.33, 0.67])
        store_brand_diversity['product_diversity_low'] = (store_brand_diversity['product_diversity_count'] <= diversity_quantiles.iloc[0]).astype(int)
        store_brand_diversity['product_diversity_medium'] = ((store_brand_diversity['product_diversity_count'] > diversity_quantiles.iloc[0]) & 
                                                            (store_brand_diversity['product_diversity_count'] <= diversity_quantiles.iloc[1])).astype(int)
        store_brand_diversity['product_diversity_high'] = (store_brand_diversity['product_diversity_count'] > diversity_quantiles.iloc[1]).astype(int)
        
        # Average sales per product
        store_brand_diversity['avg_sales_per_product'] = store_brand_diversity['sb_total_quantity'] / store_brand_diversity['product_diversity_count']
        
        # Merge back to main dataframe
        df = df.merge(store_brand_diversity, on=['primary_platform', 'store_name', 'brand_name'], how='left')
        
        # Product-level features
        product_stats = df.groupby('product_code').agg({
            'sales_quantity': ['mean', 'std', 'sum'],
            'unit_price': 'mean',
            'store_name': 'nunique',
            'primary_platform': 'nunique'
        }).reset_index()
        
        product_stats.columns = [
            'product_code',
            'product_sales_mean', 'product_sales_std', 'product_sales_total',
            'product_price_mean', 'product_store_count', 'product_platform_count'
        ]
        
        # Product popularity indicators
        product_stats['product_multi_store'] = (product_stats['product_store_count'] > 1).astype(int)
        product_stats['product_multi_platform'] = (product_stats['product_platform_count'] > 1).astype(int)
        
        # Merge product stats
        df = df.merge(product_stats, on='product_code', how='left')
        
        logger.info("✓ Product diversity features created")
        return df
    
    def _create_market_share_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market share and competitive analysis features."""
        logger.info("Creating market share features...")
        
        # Platform-level market share
        platform_totals = df.groupby('primary_platform')['sales_quantity'].sum()
        total_market = df['sales_quantity'].sum()
        
        platform_market_share = (platform_totals / total_market).to_dict()
        df['platform_market_share'] = df['primary_platform'].map(platform_market_share)
        
        # Store market share within platform
        store_platform_totals = df.groupby(['primary_platform', 'store_name'])['sales_quantity'].sum().reset_index()
        store_platform_totals.columns = ['primary_platform', 'store_name', 'store_platform_total']
        
        platform_totals_df = df.groupby('primary_platform')['sales_quantity'].sum().reset_index()
        platform_totals_df.columns = ['primary_platform', 'platform_total']
        
        store_platform_totals = store_platform_totals.merge(platform_totals_df, on='primary_platform')
        store_platform_totals['store_platform_market_share'] = store_platform_totals['store_platform_total'] / store_platform_totals['platform_total']
        
        df = df.merge(store_platform_totals[['primary_platform', 'store_name', 'store_platform_market_share']], 
                     on=['primary_platform', 'store_name'], how='left')
        
        # Brand market share within platform
        brand_platform_totals = df.groupby(['primary_platform', 'brand_name'])['sales_quantity'].sum().reset_index()
        brand_platform_totals.columns = ['primary_platform', 'brand_name', 'brand_platform_total']
        
        brand_platform_totals = brand_platform_totals.merge(platform_totals_df, on='primary_platform')
        brand_platform_totals['brand_platform_market_share'] = brand_platform_totals['brand_platform_total'] / brand_platform_totals['platform_total']
        
        df = df.merge(brand_platform_totals[['primary_platform', 'brand_name', 'brand_platform_market_share']], 
                     on=['primary_platform', 'brand_name'], how='left')
        
        # Competitive position indicators
        df['dominant_store'] = (df['store_platform_market_share'] > 0.1).astype(int)
        df['dominant_brand'] = (df['brand_platform_market_share'] > 0.1).astype(int)
        
        # Market concentration indicators
        df['highly_concentrated_store'] = (df['store_platform_market_share'] > 0.3).astype(int)
        df['highly_concentrated_brand'] = (df['brand_platform_market_share'] > 0.3).astype(int)
        
        logger.info("✓ Market share features created")
        return df
    
    def _create_loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer loyalty and retention indicators."""
        logger.info("Creating loyalty features...")
        
        # Sort for time series analysis
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        # Store-brand relationship duration
        store_brand_duration = df.groupby(['primary_platform', 'store_name', 'brand_name']).agg({
            'sales_month': ['min', 'max', 'count'],
            'sales_quantity': 'sum'
        }).reset_index()
        
        store_brand_duration.columns = [
            'primary_platform', 'store_name', 'brand_name',
            'relationship_start', 'relationship_end', 'relationship_months', 'relationship_total_sales'
        ]
        
        # Calculate relationship duration in months
        store_brand_duration['relationship_duration_months'] = (
            (store_brand_duration['relationship_end'] - store_brand_duration['relationship_start']).dt.days / 30.44
        ).round().astype(int)
        
        # Loyalty indicators
        store_brand_duration['long_term_relationship'] = (store_brand_duration['relationship_duration_months'] >= 12).astype(int)
        store_brand_duration['medium_term_relationship'] = ((store_brand_duration['relationship_duration_months'] >= 6) & 
                                                           (store_brand_duration['relationship_duration_months'] < 12)).astype(int)
        store_brand_duration['short_term_relationship'] = (store_brand_duration['relationship_duration_months'] < 6).astype(int)
        
        # Relationship consistency (months with sales / total duration)
        store_brand_duration['relationship_consistency'] = store_brand_duration['relationship_months'] / (store_brand_duration['relationship_duration_months'] + 1)
        
        # High consistency relationships
        store_brand_duration['high_consistency_relationship'] = (store_brand_duration['relationship_consistency'] > 0.8).astype(int)
        
        # Merge back
        df = df.merge(store_brand_duration[[
            'primary_platform', 'store_name', 'brand_name',
            'relationship_duration_months', 'relationship_consistency',
            'long_term_relationship', 'medium_term_relationship', 'short_term_relationship',
            'high_consistency_relationship'
        ]], on=['primary_platform', 'store_name', 'brand_name'], how='left')
        
        logger.info("✓ Loyalty features created")
        return df
    
    def _create_performance_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance categorization features."""
        logger.info("Creating performance categories...")
        
        # Create entity-level performance summaries
        entity_performance = df.groupby(['primary_platform', 'store_name', 'brand_name']).agg({
            'sales_quantity': ['mean', 'std', 'sum'],
            'sales_amount': ['mean', 'sum'],
            'unit_price': 'mean',
            'sales_month': 'count'
        }).reset_index()
        
        entity_performance.columns = [
            'primary_platform', 'store_name', 'brand_name',
            'entity_sales_mean', 'entity_sales_std', 'entity_sales_total',
            'entity_amount_mean', 'entity_amount_total',
            'entity_price_mean', 'entity_record_count'
        ]
        
        # Performance scoring
        entity_performance['entity_sales_cv'] = entity_performance['entity_sales_std'] / (entity_performance['entity_sales_mean'] + 1)
        
        # Multi-dimensional performance categories
        sales_quantiles = entity_performance['entity_sales_mean'].quantile([0.33, 0.67])
        consistency_threshold = entity_performance['entity_sales_cv'].median()
        
        # High performers: High sales + High consistency
        entity_performance['high_performer'] = ((entity_performance['entity_sales_mean'] > sales_quantiles.iloc[1]) & 
                                               (entity_performance['entity_sales_cv'] < consistency_threshold)).astype(int)
        
        # Steady performers: Medium sales + High consistency  
        entity_performance['steady_performer'] = ((entity_performance['entity_sales_mean'] > sales_quantiles.iloc[0]) & 
                                                 (entity_performance['entity_sales_mean'] <= sales_quantiles.iloc[1]) &
                                                 (entity_performance['entity_sales_cv'] < consistency_threshold)).astype(int)
        
        # Volatile performers: High sales + Low consistency
        entity_performance['volatile_performer'] = ((entity_performance['entity_sales_mean'] > sales_quantiles.iloc[1]) & 
                                                   (entity_performance['entity_sales_cv'] >= consistency_threshold)).astype(int)
        
        # Low performers: Low sales
        entity_performance['low_performer'] = (entity_performance['entity_sales_mean'] <= sales_quantiles.iloc[0]).astype(int)
        
        # Merge back
        df = df.merge(entity_performance[[
            'primary_platform', 'store_name', 'brand_name',
            'entity_sales_cv', 'high_performer', 'steady_performer', 
            'volatile_performer', 'low_performer'
        ]], on=['primary_platform', 'store_name', 'brand_name'], how='left')
        
        logger.info("✓ Performance categories created")
        return df
    
    def get_customer_behavior_feature_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize customer behavior features for deep learning model input preparation.
        
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        behavior_categories = {
            'store_metrics': [
                col for col in df.columns if any(x in col for x in [
                    'store_sales_', 'store_amount_', 'store_price_', 'store_brand_count',
                    'store_product_count', 'store_consistency_', 'store_size_'
                ])
            ],
            'brand_metrics': [
                col for col in df.columns if any(x in col for x in [
                    'brand_sales_', 'brand_amount_', 'brand_price_', 'brand_market_share_',
                    'brand_diversity_', 'brand_premium_', 'brand_budget_', 'brand_mid_range_',
                    'brand_performance_'
                ])
            ],
            'product_diversity': [
                col for col in df.columns if any(x in col for x in [
                    'product_diversity_', 'avg_sales_per_product', 'product_sales_',
                    'product_price_', 'product_multi_'
                ])
            ],
            'market_share': [
                col for col in df.columns if any(x in col for x in [
                    'platform_market_share', 'store_platform_market_share', 
                    'brand_platform_market_share', 'dominant_', 'highly_concentrated_'
                ])
            ],
            'loyalty': [
                col for col in df.columns if any(x in col for x in [
                    'relationship_', 'long_term_', 'medium_term_', 'short_term_',
                    'high_consistency_'
                ])
            ],
            'performance': [
                col for col in df.columns if any(x in col for x in [
                    'entity_sales_cv', 'high_performer', 'steady_performer',
                    'volatile_performer', 'low_performer'
                ])
            ]
        }
        
        # Filter to only include columns that exist in the dataframe
        filtered_categories = {}
        for category, features in behavior_categories.items():
            existing_features = [f for f in features if f in df.columns]
            if existing_features:
                filtered_categories[category] = existing_features
        
        return filtered_categories