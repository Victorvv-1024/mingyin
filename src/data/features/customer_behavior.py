"""
Customer behavior feature engineering for sales forecasting.

This module focuses on store and brand behavior patterns including:
- Store performance metrics
- Brand market positioning
- Customer purchasing patterns
- Behavioral consistency indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from ..utils import setup_logging

logger = setup_logging()

class CustomerBehaviorFeatureEngineer:
    """Handles all customer behavior feature engineering."""
    
    def __init__(self):
        pass
    
    def create_store_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-level behavior analysis features."""
        logger.info("Creating store behavior features...")
        
        df = df.copy()
        
        # Store-level behavior analysis
        store_behavior = df.groupby(['store_name', 'primary_platform']).agg({
            'sales_quantity': ['mean', 'std', 'min', 'max', 'count'],
            'sales_amount': ['mean', 'sum'],
            'unit_price': ['mean', 'std'],
            'brand_name': 'nunique',
            'product_code': 'nunique'
        })
        
        # Flatten column names
        store_behavior.columns = ['_'.join(col) for col in store_behavior.columns]
        store_behavior = store_behavior.reset_index()
        
        # Create derived behavior metrics
        store_behavior['store_sales_cv'] = (
            store_behavior['sales_quantity_std'] / store_behavior['sales_quantity_mean']
        ).fillna(0)  # Coefficient of variation - measures consistency
        
        store_behavior['store_sales_range'] = (
            store_behavior['sales_quantity_max'] - store_behavior['sales_quantity_min']
        )  # Sales volatility
        
        store_behavior['avg_revenue_per_transaction'] = (
            store_behavior['sales_amount_mean'] / store_behavior['sales_quantity_mean']
        ).fillna(0)
        
        store_behavior['brand_diversity'] = store_behavior['brand_name_nunique']
        store_behavior['product_diversity'] = store_behavior['product_code_nunique']
        
        # Price positioning
        store_behavior['price_premium_index'] = (
            store_behavior['unit_price_mean'] / store_behavior['unit_price_mean'].mean()
        ).fillna(1)
        
        # Store size categorization based on sales volume
        store_behavior['total_historical_sales'] = store_behavior['sales_quantity_count'] * store_behavior['sales_quantity_mean']
        store_behavior['store_size_category'] = pd.qcut(
            store_behavior['total_historical_sales'], 
            q=5, 
            labels=['Micro', 'Small', 'Medium', 'Large', 'Mega'],
            duplicates='drop'
        )
        
        # Merge back to main dataset
        behavior_cols = ['store_name', 'primary_platform', 'store_sales_cv', 'store_sales_range',
                        'avg_revenue_per_transaction', 'brand_diversity', 'product_diversity',
                        'price_premium_index', 'store_size_category']
        
        df = df.merge(store_behavior[behavior_cols], on=['store_name', 'primary_platform'], how='left')
        
        logger.info("✓ Store behavior features created")
        return df
    
    def create_brand_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create brand market positioning features."""
        logger.info("Creating brand market positioning features...")
        
        df = df.copy()
        
        # Brand market positioning features
        brand_positioning = df.groupby(['brand_name', 'primary_platform']).agg({
            'sales_quantity': ['mean', 'sum', 'count'],
            'unit_price': 'mean',
            'store_name': 'nunique'
        })
        
        brand_positioning.columns = ['_'.join(col) for col in brand_positioning.columns]
        brand_positioning = brand_positioning.reset_index()
        
        # Brand market share on each platform
        brand_market_share = df.groupby(['brand_name', 'primary_platform'])['sales_quantity'].sum().reset_index()
        platform_totals_dict = df.groupby('primary_platform')['sales_quantity'].sum().to_dict()
        
        brand_market_share['brand_market_share'] = brand_market_share.apply(
            lambda x: x['sales_quantity'] / platform_totals_dict.get(x['primary_platform'], 1), axis=1
        )
        
        df = df.merge(
            brand_market_share[['brand_name', 'primary_platform', 'brand_market_share']], 
            on=['brand_name', 'primary_platform'], 
            how='left'
        )
        
        logger.info("✓ Brand market positioning features created")
        return df
    
    def create_cross_platform_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture competitive dynamics and cross-platform effects."""
        logger.info("Creating cross-platform dynamics features...")
        
        df = df.copy()
        
        # Monthly platform competition intensity
        monthly_platform_stats = df.groupby(['sales_month', 'primary_platform']).agg({
            'brand_name': 'nunique',
            'store_name': 'nunique', 
            'sales_quantity': ['sum', 'mean', 'count']
        })
        
        monthly_platform_stats.columns = ['_'.join(col) for col in monthly_platform_stats.columns]
        monthly_platform_stats = monthly_platform_stats.reset_index()
        
        # Merge competition metrics
        df = df.merge(
            monthly_platform_stats.rename(columns={
                'brand_name_nunique': 'monthly_competing_brands',
                'store_name_nunique': 'monthly_competing_stores',
                'sales_quantity_sum': 'monthly_platform_total_sales',
                'sales_quantity_mean': 'monthly_platform_avg_sales',
                'sales_quantity_count': 'monthly_platform_transactions'
            }),
            on=['sales_month', 'primary_platform'],
            how='left'
        )
        
        # Brand performance across platforms (for brands present on multiple platforms)
        brand_platform_presence = df.groupby('brand_name')['primary_platform'].nunique()
        multi_platform_brands = brand_platform_presence[brand_platform_presence > 1].index
        
        df['is_multi_platform_brand'] = df['brand_name'].isin(multi_platform_brands).astype(int)
        
        # For multi-platform brands, calculate relative performance
        brand_platform_performance = df.groupby(['brand_name', 'primary_platform'])['sales_quantity'].mean()
        brand_overall_performance = df.groupby('brand_name')['sales_quantity'].mean()
        
        def calculate_platform_preference_score(row):
            if row['is_multi_platform_brand'] == 0:
                return 1.0  # Single platform brands get neutral score
            
            brand = row['brand_name']
            platform = row['primary_platform']
            
            platform_performance = brand_platform_performance.get((brand, platform), 0)
            overall_performance = brand_overall_performance.get(brand, 1)
            
            return platform_performance / overall_performance if overall_performance > 0 else 1.0
        
        df['brand_platform_preference_score'] = df.apply(calculate_platform_preference_score, axis=1)
        
        logger.info("✓ Cross-platform competitive dynamics created")
        return df
    
    def create_seasonal_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced seasonal and promotional interaction features."""
        logger.info("Creating seasonal interaction features...")
        
        df = df.copy()
        
        # Brand-seasonal performance patterns
        brand_seasonal_performance = df.groupby(['brand_name', 'month'])['sales_quantity'].mean()
        brand_annual_performance = df.groupby('brand_name')['sales_quantity'].mean()
        
        def get_brand_seasonal_index(row):
            brand = row['brand_name']
            month = row['month']
            
            seasonal_perf = brand_seasonal_performance.get((brand, month), 0)
            annual_perf = brand_annual_performance.get(brand, 1)
            
            return seasonal_perf / annual_perf if annual_perf > 0 else 1.0
        
        df['brand_seasonal_index'] = df.apply(get_brand_seasonal_index, axis=1)
        
        # Promotional effectiveness by brand
        promo_performance = df[df['is_promotional'] == 1].groupby('brand_name')['sales_quantity'].mean()
        non_promo_performance = df[df['is_promotional'] == 0].groupby('brand_name')['sales_quantity'].mean()
        
        promotional_lift = (promo_performance / non_promo_performance).fillna(1.0)
        df['brand_promotional_effectiveness'] = df['brand_name'].map(promotional_lift).fillna(1.0)
        
        # Complex interactions
        df['brand_seasonal_promo_interaction'] = (
            df['brand_seasonal_index'] * 
            df['is_promotional'] * 
            df['brand_promotional_effectiveness']
        )
        
        df['platform_brand_seasonal_interaction'] = (
            df['platform_seasonal_index'] * 
            df['brand_seasonal_index']
        )
        
        # Trend-season interaction (are trends stronger in certain seasons?)
        df['trend_seasonal_interaction'] = (
            df['sales_pct_change_3'] * df['monthly_intensity_learned']
        )
        
        logger.info("✓ Seasonal interaction features created")
        return df
    
    def engineer_all_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all customer behavior feature engineering."""
        logger.info("=== CUSTOMER BEHAVIOR FEATURE ENGINEERING ===")
        
        original_cols = len(df.columns)
        
        # Apply all customer behavior feature engineering
        df = self.create_store_behavior_features(df)
        df = self.create_brand_market_features(df)
        df = self.create_cross_platform_dynamics(df)
        df = self.create_seasonal_interaction_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"✓ Customer behavior engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df 