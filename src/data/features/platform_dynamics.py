"""
Platform dynamics and cross-platform competitive features.

This module implements sophisticated cross-platform analysis including:
- Cross-platform competitive dynamics
- Platform transfer patterns and loyalty
- Multi-platform seller behavior
- Platform-specific seasonal effects
- Competitive intelligence features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PlatformDynamicsEngineer:
    """Cross-platform competitive dynamics and behavior analysis."""
    
    def __init__(self):
        """Initialize platform dynamics engineer."""
        logger.info("PlatformDynamicsEngineer initialized")
    
    def engineer_all_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all platform dynamics features.
        
        Args:
            df: DataFrame with sales data
            
        Returns:
            DataFrame with comprehensive platform dynamics features
        """
        logger.info("Starting comprehensive platform dynamics feature engineering...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Step 1: Cross-platform competitive analysis
        df = self._create_cross_platform_competitive_features(df)
        
        # Step 2: Multi-platform seller behavior
        df = self._create_multi_platform_seller_features(df)
        
        # Step 3: Platform transfer and loyalty patterns
        df = self._create_platform_loyalty_features(df)
        
        # Step 4: Platform-specific seasonal effects
        df = self._create_platform_seasonal_features(df)
        
        # Step 5: Competitive intelligence features
        df = self._create_competitive_intelligence_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"Platform dynamics feature engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df
    
    def _create_cross_platform_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-platform competitive analysis features."""
        logger.info("Creating cross-platform competitive features...")
        
        # Brand presence across platforms
        brand_platform_presence = df.groupby('brand_name').agg({
            'primary_platform': 'nunique',
            'store_name': 'nunique',
            'sales_quantity': 'sum',
            'sales_amount': 'sum'
        }).reset_index()
        
        brand_platform_presence.columns = [
            'brand_name', 'brand_platform_count', 'brand_total_stores',
            'brand_cross_platform_sales', 'brand_cross_platform_amount'
        ]
        
        # Brand platform strategy indicators
        brand_platform_presence['brand_multi_platform'] = (brand_platform_presence['brand_platform_count'] > 1).astype(int)
        brand_platform_presence['brand_platform_exclusive'] = (brand_platform_presence['brand_platform_count'] == 1).astype(int)
        brand_platform_presence['brand_omnichannel'] = (brand_platform_presence['brand_platform_count'] >= 3).astype(int)
        
        # Merge back
        df = df.merge(brand_platform_presence, on='brand_name', how='left')
        
        # Store presence across platforms
        store_platform_presence = df.groupby('store_name').agg({
            'primary_platform': 'nunique',
            'brand_name': 'nunique',
            'sales_quantity': 'sum'
        }).reset_index()
        
        store_platform_presence.columns = [
            'store_name', 'store_platform_count', 'store_brand_count_cross',
            'store_cross_platform_sales'
        ]
        
        store_platform_presence['store_multi_platform_seller'] = (store_platform_presence['store_platform_count'] > 1).astype(int)
        
        df = df.merge(store_platform_presence, on='store_name', how='left')
        
        # Platform competitive intensity for brand-store combinations
        platform_competition = df.groupby(['brand_name', 'store_name']).agg({
            'primary_platform': 'nunique'
        }).reset_index()
        platform_competition.columns = ['brand_name', 'store_name', 'competing_platforms']
        
        df = df.merge(platform_competition, on=['brand_name', 'store_name'], how='left')
        df['cross_platform_competition'] = (df['competing_platforms'] > 1).astype(int)
        
        logger.info("✓ Cross-platform competitive features created")
        return df
    
    def _create_multi_platform_seller_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-platform seller behavior features."""
        logger.info("Creating multi-platform seller features...")
        
        # Identify multi-platform sellers
        multi_platform_stores = df[df['store_multi_platform_seller'] == 1]['store_name'].unique()
        
        if len(multi_platform_stores) > 0:
            # For multi-platform stores, analyze platform performance differences
            mp_store_performance = df[df['store_name'].isin(multi_platform_stores)].groupby(
                ['store_name', 'primary_platform']
            ).agg({
                'sales_quantity': ['mean', 'sum'],
                'sales_amount': 'sum',
                'unit_price': 'mean'
            }).reset_index()
            
            mp_store_performance.columns = [
                'store_name', 'primary_platform',
                'mp_sales_mean', 'mp_sales_total', 'mp_amount_total', 'mp_price_mean'
            ]
            
            # Calculate platform preferences for multi-platform stores
            store_platform_preferences = mp_store_performance.groupby('store_name').apply(
                lambda x: x.loc[x['mp_sales_total'].idxmax()]
            ).reset_index(drop=True)
            
            store_platform_preferences = store_platform_preferences[['store_name', 'primary_platform']].rename(
                columns={'primary_platform': 'preferred_platform'}
            )
            
            df = df.merge(store_platform_preferences, on='store_name', how='left')
            
            # Platform preference alignment
            df['on_preferred_platform'] = (df['primary_platform'] == df['preferred_platform']).astype(int)
            df['on_preferred_platform'] = df['on_preferred_platform'].fillna(0)
            
            # Multi-platform performance ratios
            mp_store_performance['store_platform_key'] = mp_store_performance['store_name'] + '_' + mp_store_performance['primary_platform']
            
            # Get max performance for each store across platforms
            store_max_performance = mp_store_performance.groupby('store_name')['mp_sales_total'].max().reset_index()
            store_max_performance.columns = ['store_name', 'max_platform_sales']
            
            mp_store_performance = mp_store_performance.merge(store_max_performance, on='store_name')
            mp_store_performance['platform_performance_ratio'] = mp_store_performance['mp_sales_total'] / mp_store_performance['max_platform_sales']
            
            # Merge performance ratios
            df['store_platform_key'] = df['store_name'] + '_' + df['primary_platform']
            df = df.merge(
                mp_store_performance[['store_platform_key', 'platform_performance_ratio']], 
                on='store_platform_key', how='left'
            )
            df['platform_performance_ratio'] = df['platform_performance_ratio'].fillna(1.0)
            
            # Platform specialization indicators
            df['platform_dominant_performance'] = (df['platform_performance_ratio'] > 0.8).astype(int)
            df['platform_balanced_performance'] = ((df['platform_performance_ratio'] >= 0.4) & 
                                                  (df['platform_performance_ratio'] <= 0.8)).astype(int)
            df['platform_weak_performance'] = (df['platform_performance_ratio'] < 0.4).astype(int)
            
            # Clean up temporary column
            df = df.drop(['store_platform_key'], axis=1)
        
        else:
            # If no multi-platform stores, create default columns
            df['preferred_platform'] = df['primary_platform']
            df['on_preferred_platform'] = 1
            df['platform_performance_ratio'] = 1.0
            df['platform_dominant_performance'] = 1
            df['platform_balanced_performance'] = 0
            df['platform_weak_performance'] = 0
        
        logger.info("✓ Multi-platform seller features created")
        return df
    
    def _create_platform_loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create platform loyalty and transfer pattern features."""
        logger.info("Creating platform loyalty features...")
        
        # Sort for time series analysis
        df = df.sort_values(['store_name', 'brand_name', 'sales_month'])
        
        # Platform consistency over time for store-brand combinations
        store_brand_platform_history = df.groupby(['store_name', 'brand_name']).agg({
            'primary_platform': ['nunique', 'first', 'last'],
            'sales_month': ['min', 'max', 'count']
        }).reset_index()
        
        store_brand_platform_history.columns = [
            'store_name', 'brand_name',
            'platform_switches', 'first_platform', 'last_platform',
            'first_month', 'last_month', 'total_months'
        ]
        
        # Platform loyalty indicators
        store_brand_platform_history['platform_loyal'] = (store_brand_platform_history['platform_switches'] == 1).astype(int)
        store_brand_platform_history['platform_switcher'] = (store_brand_platform_history['platform_switches'] > 1).astype(int)
        store_brand_platform_history['platform_stable'] = (store_brand_platform_history['first_platform'] == 
                                                           store_brand_platform_history['last_platform']).astype(int)
        
        # Merge back
        df = df.merge(store_brand_platform_history[[
            'store_name', 'brand_name', 'platform_loyal', 'platform_switcher', 'platform_stable'
        ]], on=['store_name', 'brand_name'], how='left')
        
        # Platform tenure (months since first appearance on platform)
        df['entity_key'] = df['store_name'] + '_' + df['brand_name'] + '_' + df['primary_platform']
        entity_first_month = df.groupby('entity_key')['sales_month'].min().reset_index()
        entity_first_month.columns = ['entity_key', 'platform_entry_month']
        
        df = df.merge(entity_first_month, on='entity_key', how='left')
        df['platform_tenure_months'] = ((df['sales_month'] - df['platform_entry_month']).dt.days / 30.44).round().astype(int)
        
        # Platform experience categories
        df['platform_newcomer'] = (df['platform_tenure_months'] <= 3).astype(int)
        df['platform_experienced'] = ((df['platform_tenure_months'] > 3) & (df['platform_tenure_months'] <= 12)).astype(int)
        df['platform_veteran'] = (df['platform_tenure_months'] > 12).astype(int)
        
        # Clean up temporary columns
        df = df.drop(['entity_key', 'platform_entry_month'], axis=1)
        
        logger.info("✓ Platform loyalty features created")
        return df
    
    def _create_platform_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create platform-specific seasonal effect features."""
        logger.info("Creating platform seasonal features...")
        
        # Ensure month column exists
        if 'month' not in df.columns:
            df['month'] = df['sales_month'].dt.month
        
        # Platform-month performance patterns
        platform_month_performance = df.groupby(['primary_platform', 'month']).agg({
            'sales_quantity': ['mean', 'sum'],
            'sales_amount': 'sum'
        }).reset_index()
        
        platform_month_performance.columns = [
            'primary_platform', 'month',
            'platform_month_avg_sales', 'platform_month_total_sales', 'platform_month_total_amount'
        ]
        
        # Calculate platform seasonal indices
        platform_annual_avg = df.groupby('primary_platform')['sales_quantity'].mean().reset_index()
        platform_annual_avg.columns = ['primary_platform', 'platform_annual_avg']
        
        platform_month_performance = platform_month_performance.merge(platform_annual_avg, on='primary_platform')
        platform_month_performance['platform_seasonal_index'] = (
            platform_month_performance['platform_month_avg_sales'] / platform_month_performance['platform_annual_avg']
        )
        
        # Merge seasonal indices
        df = df.merge(platform_month_performance[[
            'primary_platform', 'month', 'platform_seasonal_index'
        ]], on=['primary_platform', 'month'], how='left')
        
        # Platform seasonal categories
        df['platform_peak_season'] = (df['platform_seasonal_index'] > 1.2).astype(int)
        df['platform_normal_season'] = ((df['platform_seasonal_index'] >= 0.8) & 
                                       (df['platform_seasonal_index'] <= 1.2)).astype(int)
        df['platform_low_season'] = (df['platform_seasonal_index'] < 0.8).astype(int)
        
        # Platform-specific promotional effectiveness
        if 'is_promotional_period' in df.columns:
            platform_promo_effectiveness = df.groupby(['primary_platform', 'is_promotional_period']).agg({
                'sales_quantity': 'mean'
            }).reset_index()
            
            # Calculate promotional lift by platform
            promo_baseline = platform_promo_effectiveness[
                platform_promo_effectiveness['is_promotional_period'] == 0
            ][['primary_platform', 'sales_quantity']].rename(columns={'sales_quantity': 'baseline_sales'})
            
            promo_peak = platform_promo_effectiveness[
                platform_promo_effectiveness['is_promotional_period'] == 1
            ][['primary_platform', 'sales_quantity']].rename(columns={'sales_quantity': 'promo_sales'})
            
            platform_promo_lift = promo_baseline.merge(promo_peak, on='primary_platform', how='inner')
            platform_promo_lift['platform_promo_lift'] = (
                platform_promo_lift['promo_sales'] / platform_promo_lift['baseline_sales']
            ) - 1
            
            df = df.merge(platform_promo_lift[['primary_platform', 'platform_promo_lift']], 
                         on='primary_platform', how='left')
            df['platform_promo_lift'] = df['platform_promo_lift'].fillna(0)
            
            # Platform promotional responsiveness categories
            df['platform_promo_highly_responsive'] = (df['platform_promo_lift'] > 0.5).astype(int)
            df['platform_promo_moderately_responsive'] = ((df['platform_promo_lift'] > 0.2) & 
                                                         (df['platform_promo_lift'] <= 0.5)).astype(int)
            df['platform_promo_low_responsive'] = (df['platform_promo_lift'] <= 0.2).astype(int)
        
        logger.info("✓ Platform seasonal features created")
        return df
    
    def _create_competitive_intelligence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competitive intelligence and market positioning features."""
        logger.info("Creating competitive intelligence features...")
        
        # Brand-platform competitive landscape
        brand_platform_competitors = df.groupby(['brand_name', 'primary_platform']).agg({
            'store_name': 'nunique',
            'sales_quantity': 'sum',
            'sales_amount': 'sum'
        }).reset_index()
        
        brand_platform_competitors.columns = [
            'brand_name', 'primary_platform',
            'brand_platform_competitor_count', 'brand_platform_total_sales', 'brand_platform_total_amount'
        ]
        
        # Competitive intensity levels
        brand_platform_competitors['brand_platform_monopoly'] = (brand_platform_competitors['brand_platform_competitor_count'] == 1).astype(int)
        brand_platform_competitors['brand_platform_oligopoly'] = ((brand_platform_competitors['brand_platform_competitor_count'] >= 2) & 
                                                                 (brand_platform_competitors['brand_platform_competitor_count'] <= 5)).astype(int)
        brand_platform_competitors['brand_platform_competitive'] = (brand_platform_competitors['brand_platform_competitor_count'] > 5).astype(int)
        
        df = df.merge(brand_platform_competitors, on=['brand_name', 'primary_platform'], how='left')
        
        # Store competitive position within brand-platform
        store_brand_platform_sales = df.groupby(['store_name', 'brand_name', 'primary_platform']).agg({
            'sales_quantity': 'sum'
        }).reset_index()
        store_brand_platform_sales.columns = ['store_name', 'brand_name', 'primary_platform', 'store_bp_sales']
        
        df = df.merge(store_brand_platform_sales, on=['store_name', 'brand_name', 'primary_platform'], how='left')
        
        # Calculate competitive rank and share
        df['store_bp_market_share'] = df['store_bp_sales'] / df['brand_platform_total_sales']
        
        # Rank stores within brand-platform combinations
        df['store_bp_rank'] = df.groupby(['brand_name', 'primary_platform'])['store_bp_sales'].rank(
            method='dense', ascending=False
        )
        
        # Competitive position categories
        df['store_bp_leader'] = (df['store_bp_rank'] == 1).astype(int)
        df['store_bp_top_3'] = (df['store_bp_rank'] <= 3).astype(int)
        df['store_bp_top_10'] = (df['store_bp_rank'] <= 10).astype(int)
        
        # Market share categories
        df['store_bp_dominant'] = (df['store_bp_market_share'] > 0.5).astype(int)
        df['store_bp_major'] = ((df['store_bp_market_share'] > 0.2) & (df['store_bp_market_share'] <= 0.5)).astype(int)
        df['store_bp_minor'] = ((df['store_bp_market_share'] > 0.05) & (df['store_bp_market_share'] <= 0.2)).astype(int)
        df['store_bp_niche'] = (df['store_bp_market_share'] <= 0.05).astype(int)
        
        # Cross-platform competitive advantage
        if len(df['primary_platform'].unique()) > 1:
            # Calculate store performance across different platforms
            store_cross_platform_performance = df.groupby(['store_name', 'primary_platform']).agg({
                'sales_quantity': 'mean',
                'unit_price': 'mean'
            }).reset_index()
            
            # Find each store's best and worst performing platforms
            store_best_platform = store_cross_platform_performance.loc[
                store_cross_platform_performance.groupby('store_name')['sales_quantity'].idxmax()
            ][['store_name', 'primary_platform']].rename(columns={'primary_platform': 'best_platform'})
            
            df = df.merge(store_best_platform, on='store_name', how='left')
            df['on_best_platform'] = (df['primary_platform'] == df['best_platform']).astype(int)
            df['on_best_platform'] = df['on_best_platform'].fillna(0)
        else:
            df['best_platform'] = df['primary_platform']
            df['on_best_platform'] = 1
        
        # Platform specialization vs diversification strategy
        store_platform_specialization = df.groupby('store_name').agg({
            'primary_platform': 'nunique',
            'sales_quantity': ['sum', 'std']
        }).reset_index()
        
        store_platform_specialization.columns = [
            'store_name', 'store_platform_diversity', 'store_total_sales', 'store_sales_std'
        ]
        
        store_platform_specialization['store_specialization_strategy'] = (store_platform_specialization['store_platform_diversity'] == 1).astype(int)
        store_platform_specialization['store_diversification_strategy'] = (store_platform_specialization['store_platform_diversity'] > 2).astype(int)
        
        df = df.merge(store_platform_specialization[[
            'store_name', 'store_specialization_strategy', 'store_diversification_strategy'
        ]], on='store_name', how='left')
        
        # Platform entry timing advantages
        if 'platform_tenure_months' in df.columns:
            # Early mover advantage indicators
            platform_entry_quantiles = df.groupby('primary_platform')['platform_tenure_months'].quantile([0.25, 0.75]).reset_index()
            platform_entry_quantiles_pivot = platform_entry_quantiles.pivot(
                index='primary_platform', columns='level_1', values='platform_tenure_months'
            ).reset_index()
            platform_entry_quantiles_pivot.columns = ['primary_platform', 'tenure_q25', 'tenure_q75']
            
            df = df.merge(platform_entry_quantiles_pivot, on='primary_platform', how='left')
            
            df['platform_early_mover'] = (df['platform_tenure_months'] >= df['tenure_q75']).astype(int)
            df['platform_late_entrant'] = (df['platform_tenure_months'] <= df['tenure_q25']).astype(int)
            df['platform_mid_entrant'] = ((df['platform_tenure_months'] > df['tenure_q25']) & 
                                         (df['platform_tenure_months'] < df['tenure_q75'])).astype(int)
            
            # Clean up temporary columns
            df = df.drop(['tenure_q25', 'tenure_q75'], axis=1)
        
        logger.info("✓ Competitive intelligence features created")
        return df
    
    def get_platform_dynamics_feature_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize platform dynamics features for deep learning model input preparation.
        
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        platform_categories = {
            'cross_platform_competitive': [
                col for col in df.columns if any(x in col for x in [
                    'brand_platform_count', 'brand_multi_platform', 'brand_platform_exclusive',
                    'brand_omnichannel', 'store_multi_platform_seller', 'cross_platform_competition'
                ])
            ],
            'multi_platform_behavior': [
                col for col in df.columns if any(x in col for x in [
                    'preferred_platform', 'on_preferred_platform', 'platform_performance_ratio',
                    'platform_dominant_performance', 'platform_balanced_performance', 'platform_weak_performance'
                ])
            ],
            'platform_loyalty': [
                col for col in df.columns if any(x in col for x in [
                    'platform_loyal', 'platform_switcher', 'platform_stable',
                    'platform_tenure_months', 'platform_newcomer', 'platform_experienced', 'platform_veteran'
                ])
            ],
            'platform_seasonal': [
                col for col in df.columns if any(x in col for x in [
                    'platform_seasonal_index', 'platform_peak_season', 'platform_normal_season', 'platform_low_season',
                    'platform_promo_lift', 'platform_promo_highly_responsive', 'platform_promo_moderately_responsive'
                ])
            ],
            'competitive_intelligence': [
                col for col in df.columns if any(x in col for x in [
                    'brand_platform_competitor_count', 'brand_platform_monopoly', 'brand_platform_oligopoly',
                    'store_bp_market_share', 'store_bp_rank', 'store_bp_leader', 'store_bp_top_3',
                    'store_bp_dominant', 'store_bp_major', 'store_bp_minor', 'store_bp_niche',
                    'on_best_platform', 'store_specialization_strategy', 'store_diversification_strategy',
                    'platform_early_mover', 'platform_late_entrant'
                ])
            ]
        }
        
        # Filter to only include columns that exist in the dataframe
        filtered_categories = {}
        for category, features in platform_categories.items():
            existing_features = [f for f in features if f in df.columns]
            if existing_features:
                filtered_categories[category] = existing_features
        
        return filtered_categories
    
    def analyze_platform_competitive_landscape(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze the competitive landscape across platforms.
        
        Returns:
            Dictionary with competitive landscape analysis
        """
        analysis = {}
        
        # Platform market share
        platform_market_share = df.groupby('primary_platform')['sales_quantity'].sum()
        total_market = platform_market_share.sum()
        platform_market_share_pct = (platform_market_share / total_market * 100).round(2)
        analysis['platform_market_share'] = platform_market_share_pct.to_dict()
        
        # Cross-platform brand presence
        brand_platform_matrix = df.groupby(['brand_name', 'primary_platform'])['sales_quantity'].sum().unstack(fill_value=0)
        cross_platform_brands = (brand_platform_matrix > 0).sum(axis=1)
        analysis['cross_platform_brands'] = {
            'single_platform': (cross_platform_brands == 1).sum(),
            'dual_platform': (cross_platform_brands == 2).sum(),
            'multi_platform': (cross_platform_brands >= 3).sum()
        }
        
        # Platform competitive intensity
        platform_competition_stats = df.groupby('primary_platform').agg({
            'store_name': 'nunique',
            'brand_name': 'nunique',
            'sales_quantity': 'sum'
        })
        analysis['platform_competition_stats'] = platform_competition_stats.to_dict()
        
        return analysis