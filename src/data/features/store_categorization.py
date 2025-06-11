"""
Store categorization and classification features for Chinese e-commerce.

This module implements sophisticated store type classification based on:
- Chinese naming patterns and conventions
- Store behavior and performance metrics
- Platform-specific store characteristics
- Business model classification
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StoreCategorization:
    """Advanced store categorization for Chinese e-commerce platforms."""
    
    def __init__(self):
        """Initialize store categorization with Chinese naming patterns."""
        self.store_patterns = self._initialize_store_patterns()
        self.platform_patterns = self._initialize_platform_patterns()
        logger.info("StoreCategorization initialized")
    
    def _initialize_store_patterns(self) -> Dict[str, Dict]:
        """Initialize Chinese store naming patterns for classification."""
        
        patterns = {
            "flagship": {
                "keywords": ["旗舰店", "官方旗舰店", "品牌旗舰店", "flagship"],
                "weight": 1.0,
                "description": "Official brand flagship stores"
            },
            "official": {
                "keywords": ["官方", "官网", "official", "正品", "直营"],
                "weight": 0.9,
                "description": "Official brand stores"
            },
            "supermarket": {
                "keywords": ["超市", "大卖场", "商城", "mall", "supermarket", "天猫超市"],
                "weight": 0.8,
                "description": "Supermarket and mall stores"
            },
            "specialty": {
                "keywords": ["专营店", "专卖店", "特产店", "specialty", "专业"],
                "weight": 0.7,
                "description": "Specialty and category-focused stores"
            },
            "franchise": {
                "keywords": ["加盟店", "连锁", "franchise", "分店"],
                "weight": 0.6,
                "description": "Franchise and chain stores"
            },
            "distributor": {
                "keywords": ["经销商", "代理商", "distributor", "批发"],
                "weight": 0.5,
                "description": "Distributors and wholesalers"
            },
            "third_party": {
                "keywords": ["店铺", "商店", "小店", "店", "shop"],
                "weight": 0.3,
                "description": "Third-party sellers"
            },
            "cross_border": {
                "keywords": ["海外", "进口", "跨境", "global", "international", "海淘"],
                "weight": 0.4,
                "description": "Cross-border and international stores"
            }
        }
        
        return patterns
    
    def _initialize_platform_patterns(self) -> Dict[str, Dict]:
        """Initialize platform-specific store patterns."""
        
        patterns = {
            "Tmall": {
                "flagship_indicators": ["天猫旗舰店", "tmall", "天猫超市"],
                "official_indicators": ["官方", "正品保证"],
                "quality_threshold": 0.8
            },
            "JD": {
                "flagship_indicators": ["京东旗舰店", "jd", "京东自营", "京东超市"],
                "official_indicators": ["自营", "官方", "直营"],
                "quality_threshold": 0.8
            },
            "Douyin": {
                "flagship_indicators": ["官方", "旗舰", "品牌"],
                "official_indicators": ["认证", "官方认证"],
                "quality_threshold": 0.6
            }
        }
        
        return patterns
    
    def engineer_all_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all store categorization features.
        
        Args:
            df: DataFrame with store data
            
        Returns:
            DataFrame with comprehensive store categorization features
        """
        logger.info("Starting comprehensive store categorization...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Step 1: Basic store type classification
        df = self._classify_store_types(df)
        
        # Step 2: Platform-specific store features
        df = self._create_platform_store_features(df)
        
        # Step 3: Store quality and trust indicators
        df = self._create_store_quality_features(df)
        
        # Step 4: Store business model classification
        df = self._create_business_model_features(df)
        
        # Step 5: Store competitive positioning
        df = self._create_competitive_positioning_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"Store categorization completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df
    
    def _classify_store_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify stores based on Chinese naming patterns."""
        logger.info("Classifying store types...")
        
        # Initialize store type columns
        for store_type in self.store_patterns.keys():
            df[f'store_type_{store_type}'] = 0
        
        # Classify each store
        for idx, row in df.iterrows():
            store_name = str(row['store_name']).lower()
            store_scores = {}
            
            # Calculate scores for each store type
            for store_type, pattern_info in self.store_patterns.items():
                score = 0
                for keyword in pattern_info['keywords']:
                    if keyword.lower() in store_name:
                        score += pattern_info['weight']
                store_scores[store_type] = score
            
            # Assign primary store type (highest score)
            if store_scores and max(store_scores.values()) > 0:
                primary_type = max(store_scores, key=store_scores.get)
                df.loc[idx, f'store_type_{primary_type}'] = 1
            else:
                # Default to third_party if no pattern matches
                df.loc[idx, 'store_type_third_party'] = 1
        
        # Create store type summary
        df['store_type_score'] = 0
        for store_type, pattern_info in self.store_patterns.items():
            df['store_type_score'] += df[f'store_type_{store_type}'] * pattern_info['weight']
        
        logger.info("✓ Store type classification completed")
        return df
    
    def _create_platform_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create platform-specific store features."""
        logger.info("Creating platform-specific store features...")
        
        # Platform-store type combinations
        for platform in df['primary_platform'].unique():
            platform_mask = df['primary_platform'] == platform
            
            if platform in self.platform_patterns:
                pattern_info = self.platform_patterns[platform]
                
                # Platform flagship indicators
                flagship_indicators = pattern_info['flagship_indicators']
                df[f'{platform.lower()}_flagship_store'] = 0
                for indicator in flagship_indicators:
                    mask = platform_mask & df['store_name'].str.contains(indicator, case=False, na=False)
                    df.loc[mask, f'{platform.lower()}_flagship_store'] = 1
                
                # Platform official indicators
                official_indicators = pattern_info['official_indicators']
                df[f'{platform.lower()}_official_store'] = 0
                for indicator in official_indicators:
                    mask = platform_mask & df['store_name'].str.contains(indicator, case=False, na=False)
                    df.loc[mask, f'{platform.lower()}_official_store'] = 1
        
        # Cross-platform store presence
        store_platform_count = df.groupby('store_name')['primary_platform'].nunique().to_dict()
        df['store_multi_platform'] = df['store_name'].map(store_platform_count)
        df['store_multi_platform_binary'] = (df['store_multi_platform'] > 1).astype(int)
        
        # Platform preference (dominant platform for multi-platform stores)
        store_platform_sales = df.groupby(['store_name', 'primary_platform'])['sales_quantity'].sum().reset_index()
        store_dominant_platform = store_platform_sales.loc[
            store_platform_sales.groupby('store_name')['sales_quantity'].idxmax()
        ][['store_name', 'primary_platform']].rename(columns={'primary_platform': 'dominant_platform'})
        
        df = df.merge(store_dominant_platform, on='store_name', how='left')
        df['is_dominant_platform'] = (df['primary_platform'] == df['dominant_platform']).astype(int)
        
        logger.info("✓ Platform-specific store features created")
        return df
    
    def _create_store_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store quality and trust indicators."""
        logger.info("Creating store quality features...")
        
        # Store performance metrics for quality assessment
        store_quality_metrics = df.groupby(['primary_platform', 'store_name']).agg({
            'sales_quantity': ['mean', 'std', 'sum'],
            'sales_amount': ['mean', 'sum'],
            'unit_price': ['mean', 'std'],
            'brand_name': 'nunique',
            'product_code': 'nunique',
            'sales_month': 'nunique'
        }).reset_index()
        
        # Flatten columns
        store_quality_metrics.columns = [
            'primary_platform', 'store_name',
            'quality_sales_mean', 'quality_sales_std', 'quality_sales_total',
            'quality_amount_mean', 'quality_amount_total',
            'quality_price_mean', 'quality_price_std',
            'quality_brand_count', 'quality_product_count', 'quality_active_months'
        ]
        
        # Quality indicators
        store_quality_metrics['quality_consistency'] = store_quality_metrics['quality_sales_std'] / (store_quality_metrics['quality_sales_mean'] + 1)
        store_quality_metrics['quality_diversity'] = store_quality_metrics['quality_brand_count'] * store_quality_metrics['quality_product_count']
        store_quality_metrics['quality_longevity'] = store_quality_metrics['quality_active_months']
        store_quality_metrics['quality_volume'] = store_quality_metrics['quality_sales_total']
        
        # Normalize quality metrics (0-1 scale)
        for metric in ['quality_consistency', 'quality_diversity', 'quality_longevity', 'quality_volume']:
            if metric == 'quality_consistency':
                # Lower consistency score is better (less volatility)
                store_quality_metrics[f'{metric}_score'] = 1 - (
                    (store_quality_metrics[metric] - store_quality_metrics[metric].min()) / 
                    (store_quality_metrics[metric].max() - store_quality_metrics[metric].min() + 1e-6)
                )
            else:
                # Higher scores are better
                store_quality_metrics[f'{metric}_score'] = (
                    (store_quality_metrics[metric] - store_quality_metrics[metric].min()) / 
                    (store_quality_metrics[metric].max() - store_quality_metrics[metric].min() + 1e-6)
                )
        
        # Overall quality score
        store_quality_metrics['store_quality_score'] = (
            store_quality_metrics['quality_consistency_score'] * 0.3 +
            store_quality_metrics['quality_diversity_score'] * 0.2 +
            store_quality_metrics['quality_longevity_score'] * 0.2 +
            store_quality_metrics['quality_volume_score'] * 0.3
        )
        
        # Quality categories
        quality_quantiles = store_quality_metrics['store_quality_score'].quantile([0.33, 0.67])
        store_quality_metrics['store_quality_high'] = (store_quality_metrics['store_quality_score'] > quality_quantiles.iloc[1]).astype(int)
        store_quality_metrics['store_quality_medium'] = ((store_quality_metrics['store_quality_score'] > quality_quantiles.iloc[0]) & 
                                                        (store_quality_metrics['store_quality_score'] <= quality_quantiles.iloc[1])).astype(int)
        store_quality_metrics['store_quality_low'] = (store_quality_metrics['store_quality_score'] <= quality_quantiles.iloc[0]).astype(int)
        
        # Trust indicators based on store type and quality
        store_quality_metrics['store_trust_high'] = ((store_quality_metrics['store_quality_score'] > 0.7)).astype(int)
        store_quality_metrics['store_trust_verified'] = 0  # Will be set based on store type
        
        # Merge back
        df = df.merge(store_quality_metrics[[
            'primary_platform', 'store_name', 'store_quality_score',
            'store_quality_high', 'store_quality_medium', 'store_quality_low',
            'store_trust_high'
        ]], on=['primary_platform', 'store_name'], how='left')
        
        # Set verified trust based on store type
        df['store_trust_verified'] = (
            df['store_type_flagship'] | 
            df['store_type_official'] | 
            df['store_type_supermarket']
        ).astype(int)
        
        logger.info("✓ Store quality features created")
        return df
    
    def _create_business_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store business model classification features."""
        logger.info("Creating business model features...")
        
        # Business model indicators based on behavior patterns
        store_business_metrics = df.groupby(['primary_platform', 'store_name']).agg({
            'brand_name': 'nunique',
            'product_code': 'nunique',
            'unit_price': ['mean', 'std'],
            'sales_quantity': ['mean', 'sum']
        }).reset_index()
        
        store_business_metrics.columns = [
            'primary_platform', 'store_name',
            'bm_brand_count', 'bm_product_count',
            'bm_price_mean', 'bm_price_std',
            'bm_sales_mean', 'bm_sales_total'
        ]
        
        # Business model classification
        # Multi-brand retailer
        store_business_metrics['bm_multi_brand_retailer'] = (store_business_metrics['bm_brand_count'] >= 5).astype(int)
        
        # Single brand specialist
        store_business_metrics['bm_single_brand_specialist'] = (store_business_metrics['bm_brand_count'] == 1).astype(int)
        
        # Limited brand specialist
        store_business_metrics['bm_limited_brand_specialist'] = ((store_business_metrics['bm_brand_count'] >= 2) & 
                                                               (store_business_metrics['bm_brand_count'] <= 4)).astype(int)
        
        # High volume retailer
        volume_threshold = store_business_metrics['bm_sales_total'].quantile(0.8)
        store_business_metrics['bm_high_volume_retailer'] = (store_business_metrics['bm_sales_total'] > volume_threshold).astype(int)
        
        # Premium retailer (high average prices)
        price_threshold = store_business_metrics['bm_price_mean'].quantile(0.8)
        store_business_metrics['bm_premium_retailer'] = (store_business_metrics['bm_price_mean'] > price_threshold).astype(int)
        
        # Budget retailer (low average prices)
        budget_threshold = store_business_metrics['bm_price_mean'].quantile(0.2)
        store_business_metrics['bm_budget_retailer'] = (store_business_metrics['bm_price_mean'] < budget_threshold).astype(int)
        
        # Price consistency (low price variation)
        store_business_metrics['bm_price_cv'] = store_business_metrics['bm_price_std'] / (store_business_metrics['bm_price_mean'] + 1)
        store_business_metrics['bm_consistent_pricing'] = (store_business_metrics['bm_price_cv'] < 0.3).astype(int)
        
        # Diverse pricing strategy
        store_business_metrics['bm_diverse_pricing'] = (store_business_metrics['bm_price_cv'] > 0.7).astype(int)
        
        # Merge back
        df = df.merge(store_business_metrics[[
            'primary_platform', 'store_name',
            'bm_multi_brand_retailer', 'bm_single_brand_specialist', 'bm_limited_brand_specialist',
            'bm_high_volume_retailer', 'bm_premium_retailer', 'bm_budget_retailer',
            'bm_consistent_pricing', 'bm_diverse_pricing'
        ]], on=['primary_platform', 'store_name'], how='left')
        
        logger.info("✓ Business model features created")
        return df
    
    def _create_competitive_positioning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competitive positioning features."""
        logger.info("Creating competitive positioning features...")
        
        # Competitive analysis within platform-brand combinations
        platform_brand_competition = df.groupby(['primary_platform', 'brand_name']).agg({
            'store_name': 'nunique',
            'sales_quantity': 'sum'
        }).reset_index()
        platform_brand_competition.columns = ['primary_platform', 'brand_name', 'competing_stores', 'total_brand_sales']
        
        # Merge to get competition level
        df = df.merge(platform_brand_competition, on=['primary_platform', 'brand_name'], how='left')
        
        # Competition intensity categories
        df['competition_low'] = (df['competing_stores'] <= 2).astype(int)
        df['competition_medium'] = ((df['competing_stores'] > 2) & (df['competing_stores'] <= 5)).astype(int)
        df['competition_high'] = (df['competing_stores'] > 5).astype(int)
        
        # Store's share within brand-platform combination
        store_brand_platform_sales = df.groupby(['primary_platform', 'brand_name', 'store_name'])['sales_quantity'].sum().reset_index()
        store_brand_platform_sales.columns = ['primary_platform', 'brand_name', 'store_name', 'store_brand_sales']
        
        df = df.merge(store_brand_platform_sales, on=['primary_platform', 'brand_name', 'store_name'], how='left')
        df['store_brand_share'] = df['store_brand_sales'] / df['total_brand_sales']
        
        # Competitive position categories
        df['competitive_leader'] = (df['store_brand_share'] > 0.5).astype(int)
        df['competitive_major'] = ((df['store_brand_share'] > 0.2) & (df['store_brand_share'] <= 0.5)).astype(int)
        df['competitive_minor'] = ((df['store_brand_share'] > 0.05) & (df['store_brand_share'] <= 0.2)).astype(int)
        df['competitive_niche'] = (df['store_brand_share'] <= 0.05).astype(int)
        
        # Platform positioning
        store_platform_performance = df.groupby(['primary_platform', 'store_name']).agg({
            'sales_quantity': ['mean', 'sum'],
            'brand_name': 'nunique'
        }).reset_index()
        store_platform_performance.columns = [
            'primary_platform', 'store_name',
            'platform_sales_mean', 'platform_sales_total', 'platform_brand_count'
        ]
        
        # Platform rankings
        for platform in df['primary_platform'].unique():
            platform_stores = store_platform_performance[
                store_platform_performance['primary_platform'] == platform
            ].copy()
            
            platform_stores[f'{platform.lower()}_sales_rank'] = platform_stores['platform_sales_total'].rank(
                method='dense', ascending=False
            )
            platform_stores[f'{platform.lower()}_top_10_store'] = (platform_stores[f'{platform.lower()}_sales_rank'] <= 10).astype(int)
            platform_stores[f'{platform.lower()}_top_50_store'] = (platform_stores[f'{platform.lower()}_sales_rank'] <= 50).astype(int)
            
            df = df.merge(platform_stores[[
                'primary_platform', 'store_name',
                f'{platform.lower()}_sales_rank', f'{platform.lower()}_top_10_store', f'{platform.lower()}_top_50_store'
            ]], on=['primary_platform', 'store_name'], how='left')
        
        logger.info("✓ Competitive positioning features created")
        return df
    
    def get_store_categorization_feature_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize store features for deep learning model input preparation.
        
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        store_categories = {
            'store_type': [
                col for col in df.columns if col.startswith('store_type_')
            ],
            'platform_specific': [
                col for col in df.columns if any(x in col for x in [
                    '_flagship_store', '_official_store', 'store_multi_platform',
                    'is_dominant_platform', 'dominant_platform'
                ])
            ],
            'quality_trust': [
                col for col in df.columns if any(x in col for x in [
                    'store_quality_', 'store_trust_'
                ])
            ],
            'business_model': [
                col for col in df.columns if col.startswith('bm_')
            ],
            'competitive_position': [
                col for col in df.columns if any(x in col for x in [
                    'competition_', 'competitive_', 'store_brand_share',
                    '_sales_rank', '_top_10_store', '_top_50_store'
                ])
            ]
        }
        
        # Filter to only include columns that exist in the dataframe
        filtered_categories = {}
        for category, features in store_categories.items():
            existing_features = [f for f in features if f in df.columns]
            if existing_features:
                filtered_categories[category] = existing_features
        
        return filtered_categories
    
    def analyze_store_distribution(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze store type distribution across platforms.
        
        Returns:
            Dictionary with distribution analysis
        """
        analysis = {}
        
        # Store type distribution
        store_type_cols = [col for col in df.columns if col.startswith('store_type_')]
        store_type_dist = df[store_type_cols].sum().to_dict()
        analysis['store_type_distribution'] = store_type_dist
        
        # Platform-specific store type distribution
        platform_store_dist = {}
        for platform in df['primary_platform'].unique():
            platform_data = df[df['primary_platform'] == platform]
            platform_store_dist[platform] = platform_data[store_type_cols].sum().to_dict()
        analysis['platform_store_type_distribution'] = platform_store_dist
        
        # Quality distribution
        quality_cols = ['store_quality_high', 'store_quality_medium', 'store_quality_low']
        quality_dist = df[quality_cols].sum().to_dict()
        analysis['quality_distribution'] = quality_dist
        
        # Business model distribution
        bm_cols = [col for col in df.columns if col.startswith('bm_') and col.endswith('_retailer')]
        bm_dist = df[bm_cols].sum().to_dict()
        analysis['business_model_distribution'] = bm_dist
        
        return analysis