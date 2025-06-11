"""
Master feature engineering pipeline for sales forecasting.

This module orchestrates the complete feature engineering process,
combining all specialized feature modules into a cohesive pipeline that
matches the exact sequence from full_data_prediction.ipynb.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging

from .utils import setup_logging, get_modeling_features
from .features.temporal import TemporalFeatureEngineer
from .features.customer_behavior import CustomerBehaviorFeatureEngineer
from .features.store_categorization import StoreCategorization
from .features.platform_dynamics import PlatformDynamicsEngineer
from .preprocessing import SalesDataProcessor

logger = setup_logging()

class SalesFeaturePipeline:
    """
    Master pipeline for comprehensive sales forecasting feature engineering.
    Implements the exact 6-step sequence from full_data_prediction.ipynb.
    """
    
    def __init__(self, output_dir: str = "data/engineered"):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            output_dir: Directory to save engineered datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all feature engineers
        self.temporal_engineer = TemporalFeatureEngineer()
        self.customer_behavior_engineer = CustomerBehaviorFeatureEngineer()
        self.store_categorization = StoreCategorization()
        self.platform_dynamics_engineer = PlatformDynamicsEngineer()
        
        logger.info("SalesFeaturePipeline initialized with all feature engineers")
        
    def run_complete_pipeline(self, 
                            raw_data_dir: str,
                            years: List[int] = [2021, 2022, 2023]) -> Tuple[str, List[str], List[Tuple], Dict]:
        """
        Run the complete feature engineering pipeline from raw data to engineered dataset.
        
        Args:
            raw_data_dir: Directory containing raw Excel files
            years: Years to process
            
        Returns:
            Tuple of (engineered_data_path, modeling_features, rolling_splits, metadata)
        """
        logger.info("=" * 80)
        logger.info("RUNNING COMPLETE SALES FORECASTING FEATURE PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        processor = SalesDataProcessor(raw_data_dir, "data/processed")
        processed_data_path, quality_metrics = processor.process(years)
        
        # Load processed data
        df = pd.read_pickle(processed_data_path)
        logger.info(f"Loaded processed data: {df.shape}")
        
        # Step 2: Comprehensive Feature Engineering (6-step sequence)
        logger.info("Step 2: Comprehensive Feature Engineering")
        df_engineered = self.engineer_all_features(df)
        
        # Step 3: Prepare for modeling
        logger.info("Step 3: Preparing Modeling Dataset")
        modeling_features = self.get_modeling_features(df_engineered)
        
        # Step 4: Create rolling splits
        logger.info("Step 4: Creating Rolling Time Series Splits")
        rolling_splits = self.create_enhanced_rolling_splits(df_engineered)
        
        # Step 5: Feature validation and quality checks
        logger.info("Step 5: Feature Validation and Quality Checks")
        validation_results = self.validate_engineered_features(df_engineered, modeling_features)
        
        # Step 6: Save engineered dataset
        logger.info("Step 6: Saving Engineered Dataset")
        engineered_data_path = self.save_engineered_dataset(
            df_engineered, modeling_features, rolling_splits, quality_metrics, validation_results
        )
        
        logger.info("=" * 80)
        logger.info("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final dataset: {df_engineered.shape}")
        logger.info(f"Modeling features: {len(modeling_features)}")
        logger.info(f"Rolling splits: {len(rolling_splits)}")
        logger.info(f"Saved to: {engineered_data_path}")
        
        return engineered_data_path, modeling_features, rolling_splits, {**quality_metrics, **validation_results}
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering following the exact sequence from the notebook.
        
        This implements the 6-step process:
        1. Basic setup and data preparation
        2. Temporal features (advanced temporal patterns)
        3. Customer behavior features (store and brand analytics)
        4. Store categorization and platform dynamics
        5. Advanced interactions and derived features
        6. Final feature consolidation and cleanup
        
        Args:
            df: Processed DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 60)
        logger.info("STARTING 6-STEP FEATURE ENGINEERING SEQUENCE")
        logger.info("=" * 60)
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Step 1: Basic Setup and Data Preparation
        logger.info("STEP 1: Basic Setup and Data Preparation")
        df = self._step1_basic_setup(df)
        step1_cols = len(df.columns)
        logger.info(f"✓ Step 1 completed: {original_cols} → {step1_cols} (+{step1_cols - original_cols} features)")
        
        # Step 2: Temporal Features (Advanced Temporal Patterns)
        logger.info("STEP 2: Temporal Features (Advanced Temporal Patterns)")
        df = self._step2_temporal_features(df)
        step2_cols = len(df.columns)
        logger.info(f"✓ Step 2 completed: {step1_cols} → {step2_cols} (+{step2_cols - step1_cols} features)")
        
        # Step 3: Customer Behavior Features (Store and Brand Analytics)
        logger.info("STEP 3: Customer Behavior Features (Store and Brand Analytics)")
        df = self._step3_customer_behavior_features(df)
        step3_cols = len(df.columns)
        logger.info(f"✓ Step 3 completed: {step2_cols} → {step3_cols} (+{step3_cols - step2_cols} features)")
        
        # Step 4: Store Categorization and Platform Dynamics
        logger.info("STEP 4: Store Categorization and Platform Dynamics")
        df = self._step4_store_and_platform_features(df)
        step4_cols = len(df.columns)
        logger.info(f"✓ Step 4 completed: {step3_cols} → {step4_cols} (+{step4_cols - step3_cols} features)")
        
        # Step 5: Advanced Interactions and Derived Features
        logger.info("STEP 5: Advanced Interactions and Derived Features")
        df = self._step5_advanced_interactions(df)
        step5_cols = len(df.columns)
        logger.info(f"✓ Step 5 completed: {step4_cols} → {step5_cols} (+{step5_cols - step4_cols} features)")
        
        # Step 6: Final Feature Consolidation and Cleanup
        logger.info("STEP 6: Final Feature Consolidation and Cleanup")
        df = self._step6_final_consolidation(df)
        final_cols = len(df.columns)
        logger.info(f"✓ Step 6 completed: {step5_cols} → {final_cols} (+{final_cols - step5_cols} features)")
        
        # Print comprehensive feature summary
        self._print_comprehensive_feature_summary(df, original_cols)
        
        logger.info("=" * 60)
        logger.info("6-STEP FEATURE ENGINEERING SEQUENCE COMPLETED")
        logger.info("=" * 60)
        
        return df
    
    def _step1_basic_setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Basic setup and data preparation."""
        logger.info("  Setting up basic features and data types...")
        
        # Ensure datetime conversion
        df['sales_month'] = pd.to_datetime(df['sales_month'])
        
        # Add unit price calculation (critical for many features)
        if 'unit_price' not in df.columns:
            df['unit_price'] = df['sales_amount'] / df['sales_quantity']
            df['unit_price'] = df['unit_price'].replace([np.inf, -np.inf], np.nan)
            df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        
        # Log transform target variable for modeling
        df['sales_quantity_log'] = np.log1p(df['sales_quantity'])
        
        # Basic data quality indicators
        df['has_missing_price'] = df['unit_price'].isna().astype(int)
        df['has_zero_sales'] = (df['sales_quantity'] == 0).astype(int)
        df['has_high_price'] = (df['unit_price'] > df['unit_price'].quantile(0.95)).astype(int)
        
        # Sort data for time series operations
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        logger.info("  ✓ Basic setup completed")
        return df
    
    def _step2_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Apply comprehensive temporal feature engineering."""
        logger.info("  Applying temporal feature engineering...")
        
        # Use the comprehensive temporal engineer
        df = self.temporal_engineer.engineer_all_temporal_features(df)
        
        # Add seasonal interaction features specific to the notebook
        df = self._create_seasonal_interactions(df)
        
        logger.info("  ✓ Temporal features completed")
        return df
    
    def _step3_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Apply customer behavior and analytics features."""
        logger.info("  Applying customer behavior analytics...")
        
        # Use the comprehensive customer behavior engineer
        df = self.customer_behavior_engineer.engineer_all_customer_behavior_features(df)
        
        logger.info("  ✓ Customer behavior features completed")
        return df
    
    def _step4_store_and_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Apply store categorization and platform dynamics."""
        logger.info("  Applying store categorization and platform dynamics...")
        
        # Store categorization
        df = self.store_categorization.engineer_all_store_features(df)
        
        # Platform dynamics
        df = self.platform_dynamics_engineer.engineer_all_platform_features(df)
        
        logger.info("  ✓ Store and platform features completed")
        return df
    
    def _step5_advanced_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Create advanced interaction and derived features."""
        logger.info("  Creating advanced interactions and derived features...")
        
        # Seasonal-Brand interactions
        df = self._create_seasonal_brand_interactions(df)
        
        # Platform-Brand interactions
        df = self._create_platform_brand_interactions(df)
        
        # Store-Brand performance interactions
        df = self._create_store_brand_interactions(df)
        
        # Spike detection features
        df = self._create_spike_detection_features(df)
        
        # Advanced momentum features
        df = self._create_advanced_momentum_features(df)
        
        logger.info("  ✓ Advanced interactions completed")
        return df
    
    def _step6_final_consolidation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6: Final feature consolidation and cleanup."""
        logger.info("  Performing final consolidation and cleanup...")
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with appropriate defaults
        df = self._fill_missing_values(df)
        
        # Remove temporary columns used for calculations
        temp_cols = [col for col in df.columns if any(x in col for x in ['_temp_', '_intermediate_'])]
        if temp_cols:
            df = df.drop(columns=temp_cols)
            logger.info(f"  Removed {len(temp_cols)} temporary columns")
        
        # Ensure proper data types for deep learning
        df = self._ensure_proper_dtypes(df)
        
        logger.info("  ✓ Final consolidation completed")
        return df
    
    def _create_seasonal_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal interaction features."""
        # Month-Platform interactions
        if 'month' in df.columns:
            for platform in df['primary_platform'].unique():
                platform_safe = platform.lower().replace(' ', '_')
                df[f'month_{platform_safe}_interaction'] = (
                    df['month'] * (df['primary_platform'] == platform).astype(int)
                )
        
        # Promotional-Platform interactions
        if 'is_promotional_period' in df.columns:
            for platform in df['primary_platform'].unique():
                platform_safe = platform.lower().replace(' ', '_')
                df[f'promo_{platform_safe}_interaction'] = (
                    df['is_promotional_period'] * (df['primary_platform'] == platform).astype(int)
                )
        
        return df
    
    def _create_seasonal_brand_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal-brand interaction features."""
        # Get top brands for interaction features
        top_brands = df['brand_name'].value_counts().head(10).index.tolist()
        
        if 'month' in df.columns:
            for brand in top_brands:
                brand_safe = brand.replace(' ', '_').replace('(', '').replace(')', '').lower()
                df[f'seasonal_brand_{brand_safe}'] = (
                    df['month'] * (df['brand_name'] == brand).astype(int)
                )
        
        return df
    
    def _create_platform_brand_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create platform-brand interaction features."""
        # Platform-brand combinations
        platform_brand_combos = df.groupby(['primary_platform', 'brand_name']).size().reset_index(name='combo_frequency')
        top_combos = platform_brand_combos.nlargest(15, 'combo_frequency')
        
        for _, row in top_combos.iterrows():
            platform = row['primary_platform']
            brand = row['brand_name']
            platform_safe = platform.lower().replace(' ', '_')
            brand_safe = brand.replace(' ', '_').replace('(', '').replace(')', '').lower()
            
            df[f'platform_brand_{platform_safe}_{brand_safe}'] = (
                (df['primary_platform'] == platform) & (df['brand_name'] == brand)
            ).astype(int)
        
        return df
    
    def _create_store_brand_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-brand performance interaction features."""
        # Store-brand relationship strength
        store_brand_strength = df.groupby(['store_name', 'brand_name']).agg({
            'sales_quantity': ['count', 'mean', 'sum'],
            'sales_month': 'nunique'
        }).reset_index()
        
        store_brand_strength.columns = [
            'store_name', 'brand_name', 'sb_record_count', 'sb_avg_sales', 'sb_total_sales', 'sb_active_months'
        ]
        
        # Relationship strength indicators
        store_brand_strength['sb_relationship_strength'] = (
            store_brand_strength['sb_active_months'] * store_brand_strength['sb_avg_sales']
        )
        
        # Normalize relationship strength
        store_brand_strength['sb_relationship_score'] = (
            store_brand_strength['sb_relationship_strength'] / 
            (store_brand_strength['sb_relationship_strength'].max() + 1e-6)
        )
        
        df = df.merge(store_brand_strength[[
            'store_name', 'brand_name', 'sb_relationship_score'
        ]], on=['store_name', 'brand_name'], how='left')
        
        return df
    
    def _create_spike_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spike detection and anomaly features."""
        # Sort for proper calculations
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        # Create entity grouping
        entity_groups = df.groupby(['primary_platform', 'store_name', 'brand_name'])
        
        # Z-score based spike detection
        df['sales_z_score'] = entity_groups['sales_quantity'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        
        # Spike indicators
        df['sales_spike_major'] = (df['sales_z_score'] > 2.5).astype(int)
        df['sales_spike_moderate'] = ((df['sales_z_score'] > 1.5) & (df['sales_z_score'] <= 2.5)).astype(int)
        df['sales_drop_major'] = (df['sales_z_score'] < -2.5).astype(int)
        df['sales_drop_moderate'] = ((df['sales_z_score'] < -1.5) & (df['sales_z_score'] >= -2.5)).astype(int)
        
        # Moving deviation from trend
        df['sales_ma_3'] = entity_groups['sales_quantity'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['sales_deviation_from_ma'] = (df['sales_quantity'] - df['sales_ma_3']) / (df['sales_ma_3'] + 1)
        
        # Spike propensity (tendency to have spikes)
        spike_propensity = entity_groups.agg({
            'sales_spike_major': 'mean',
            'sales_spike_moderate': 'mean'
        }).reset_index()
        spike_propensity['spike_propensity_score'] = (
            spike_propensity['sales_spike_major'] * 1.0 + spike_propensity['sales_spike_moderate'] * 0.5
        )
        
        df = df.merge(spike_propensity[['primary_platform', 'store_name', 'brand_name', 'spike_propensity_score']], 
                     on=['primary_platform', 'store_name', 'brand_name'], how='left')
        
        return df
    
    def _create_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced momentum and trend features."""
        # Sort for time series calculations
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        # Create entity grouping
        entity_groups = df.groupby(['primary_platform', 'store_name', 'brand_name'])
        
        # Multi-period momentum
        for periods in [2, 3, 6]:
            df[f'momentum_{periods}m'] = entity_groups['sales_quantity'].transform(
                lambda x: x.pct_change(periods).fillna(0)
            )
        
        # Trend consistency (how consistent is the trend direction)
        df['trend_consistency_3m'] = entity_groups['momentum_2m'].transform(
            lambda x: (x.rolling(3).apply(lambda y: (y > 0).sum()).fillna(0) / 3)
        )
        
        # Momentum acceleration (second derivative)
        df['momentum_acceleration'] = entity_groups['sales_mom_change'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Categorical/binary features → 0
        binary_cols = [col for col in df.columns if df[col].dtype == 'int64' and df[col].max() <= 1]
        df[binary_cols] = df[binary_cols].fillna(0)
        
        # Ratio/percentage features → 0
        ratio_cols = [col for col in df.columns if any(x in col for x in ['_ratio', '_share', '_cv', '_pct'])]
        df[ratio_cols] = df[ratio_cols].fillna(0)
        
        # Score features → median
        score_cols = [col for col in df.columns if any(x in col for x in ['_score', '_index'])]
        for col in score_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Remaining numeric columns → 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _ensure_proper_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for deep learning models."""
        # Binary features → int32
        binary_cols = [col for col in df.columns if df[col].dtype in ['int64', 'bool'] and df[col].max() <= 1]
        df[binary_cols] = df[binary_cols].astype('int32')
        
        # Count features → int32
        count_cols = [col for col in df.columns if any(x in col for x in ['_count', '_nunique', '_rank'])]
        for col in count_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].astype('int32')
        
        # Float features → float32
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        return df
    
    def get_modeling_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get comprehensive list of modeling features, categorized for deep learning.
        
        Args:
            df: Engineered DataFrame
            
        Returns:
            List of feature names suitable for modeling
        """
        # Exclude non-feature columns
        exclude_patterns = [
            'sales_month', 'store_name', 'brand_name', 'product_code', 
            'primary_platform_original', 'secondary_platform',
            'sales_amount', 'sales_quantity',  # Keep only log-transformed target
            '_temp_', '_intermediate_', 'entity_key', 'store_platform_key'
        ]
        
        modeling_features = []
        for col in df.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                if df[col].dtype in ['int32', 'int64', 'float32', 'float64', 'bool']:
                    modeling_features.append(col)
        
        # Categorize features for deep learning preparation
        feature_categories = self._categorize_features_for_modeling(df, modeling_features)
        
        logger.info(f"Identified {len(modeling_features)} modeling features")
        for category, features in feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")
        
        return modeling_features
    
    def _categorize_features_for_modeling(self, df: pd.DataFrame, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features for deep learning model preparation."""
        categories = {
            'temporal_basic': [],
            'temporal_cyclical': [],
            'promotional': [],
            'lag_features': [],
            'rolling_features': [],
            'momentum_features': [],
            'customer_behavior': [],
            'store_categorization': [],
            'platform_dynamics': [],
            'interactions': [],
            'spike_detection': [],
            'target': []
        }
        
        for feature in features:
            if any(x in feature for x in ['month', 'quarter', 'year', 'day_of_year']):
                categories['temporal_basic'].append(feature)
            elif any(x in feature for x in ['_sin', '_cos']):
                categories['temporal_cyclical'].append(feature)
            elif any(x in feature for x in ['promotional', 'promo', 'holiday']):
                categories['promotional'].append(feature)
            elif '_lag_' in feature:
                categories['lag_features'].append(feature)
            elif 'rolling_' in feature:
                categories['rolling_features'].append(feature)
            elif any(x in feature for x in ['momentum', 'trend_', 'acceleration', 'volatility']):
                categories['momentum_features'].append(feature)
            elif any(x in feature for x in ['store_', 'brand_', 'market_share', 'diversity']):
                categories['customer_behavior'].append(feature)
            elif any(x in feature for x in ['store_type_', 'quality_', 'business_model', 'competitive_']):
                categories['store_categorization'].append(feature)
            elif any(x in feature for x in ['platform_', 'cross_platform']):
                categories['platform_dynamics'].append(feature)
            elif 'interaction' in feature:
                categories['interactions'].append(feature)
            elif any(x in feature for x in ['spike_', 'z_score', 'deviation']):
                categories['spike_detection'].append(feature)
            elif feature == 'sales_quantity_log':
                categories['target'].append(feature)
        
        return categories
    
    def create_enhanced_rolling_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """Create enhanced rolling time series splits with proper temporal validation."""
        logger.info("Creating enhanced rolling time series splits...")
        
        # Sort by date
        df_sorted = df.sort_values('sales_month').copy()
        
        rolling_splits = []
        
        # Split 1: 2021 → 2022 Q1
        train_1 = df_sorted[df_sorted['sales_month'].dt.year == 2021].copy()
        val_1 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([1, 2, 3]))
        ].copy()
        
        if len(train_1) > 0 and len(val_1) > 0:
            rolling_splits.append((train_1, val_1, "2021_full → 2022_Q1"))
            logger.info(f"Split 1: Train {len(train_1):,} samples, Val {len(val_1):,} samples")
        
        # Split 2: 2021 + 2022 Q1 → 2022 Q2
        train_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & (df_sorted['sales_month'].dt.month.isin([1, 2, 3])))
        ].copy()
        val_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([4, 5, 6]))
        ].copy()
        
        if len(train_2) > 0 and len(val_2) > 0:
            rolling_splits.append((train_2, val_2, "2021_full+2022_Q1 → 2022_Q2"))
            logger.info(f"Split 2: Train {len(train_2):,} samples, Val {len(val_2):,} samples")
        
        # Split 3: 2021 + 2022 Q1-Q2 → 2022 Q3
        train_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & (df_sorted['sales_month'].dt.month.isin([1, 2, 3, 4, 5, 6])))
        ].copy()
        val_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([7, 8, 9]))
        ].copy()
        
        if len(train_3) > 0 and len(val_3) > 0:
            rolling_splits.append((train_3, val_3, "2021_full+2022_Q1Q2 → 2022_Q3"))
            logger.info(f"Split 3: Train {len(train_3):,} samples, Val {len(val_3):,} samples")
        
        # Split 4: 2021 + 2022 Q1-Q3 → 2022 Q4
        train_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & (df_sorted['sales_month'].dt.month <= 9))
        ].copy()
        val_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([10, 11, 12]))
        ].copy()
        
        if len(train_4) > 0 and len(val_4) > 0:
            rolling_splits.append((train_4, val_4, "2021_full+2022_Q1Q2Q3 → 2022_Q4"))
            logger.info(f"Split 4: Train {len(train_4):,} samples, Val {len(val_4):,} samples")
        
        # Optional Split 5: Include 2023 data if available
        available_2023 = df_sorted[df_sorted['sales_month'].dt.year == 2023]
        if len(available_2023) > 0:
            train_5 = df_sorted[
                (df_sorted['sales_month'].dt.year == 2021) |
                (df_sorted['sales_month'].dt.year == 2022)
            ].copy()
            val_5 = df_sorted[
                (df_sorted['sales_month'].dt.year == 2023) & 
                (df_sorted['sales_month'].dt.month.isin([1, 2, 3]))
            ].copy()
            
            if len(val_5) > 0:
                rolling_splits.append((train_5, val_5, "2021_full+2022_full → 2023_Q1"))
                logger.info(f"Split 5: Train {len(train_5):,} samples, Val {len(val_5):,} samples")
        
        # Validate splits for temporal integrity
        for i, (train_data, val_data, description) in enumerate(rolling_splits):
            train_max_date = train_data['sales_month'].max()
            val_min_date = val_data['sales_month'].min()
            
            if train_max_date >= val_min_date:
                logger.warning(f"Split {i+1} has temporal overlap! Train max: {train_max_date}, Val min: {val_min_date}")
            else:
                logger.info(f"Split {i+1} temporal integrity validated ✓")
        
        logger.info(f"Created {len(rolling_splits)} enhanced rolling time series splits")
        return rolling_splits
    
    def validate_engineered_features(self, df: pd.DataFrame, modeling_features: List[str]) -> Dict[str, any]:
        """
        Validate engineered features for quality and completeness.
        
        Args:
            df: Engineered DataFrame
            modeling_features: List of modeling features
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating engineered features...")
        
        validation_results = {}
        
        # Feature completeness checks
        missing_ratios = df[modeling_features].isnull().mean()
        high_missing_features = missing_ratios[missing_ratios > 0.1].index.tolist()
        validation_results['high_missing_features'] = high_missing_features
        validation_results['max_missing_ratio'] = missing_ratios.max()
        
        # Infinite value checks
        inf_features = []
        for feature in modeling_features:
            if np.isinf(df[feature]).any():
                inf_features.append(feature)
        validation_results['infinite_value_features'] = inf_features
        
        # Zero variance features
        zero_var_features = []
        for feature in modeling_features:
            if df[feature].var() == 0:
                zero_var_features.append(feature)
        validation_results['zero_variance_features'] = zero_var_features
        
        # Feature distribution checks
        skewed_features = []
        for feature in modeling_features:
            if df[feature].dtype in ['float32', 'float64', 'int32', 'int64']:
                skewness = df[feature].skew()
                if abs(skewness) > 3:
                    skewed_features.append((feature, skewness))
        validation_results['highly_skewed_features'] = skewed_features
        
        # Feature correlation checks (potential multicollinearity)
        high_corr_pairs = []
        if len(modeling_features) > 1:
            corr_matrix = df[modeling_features].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        validation_results['high_correlation_pairs'] = high_corr_pairs
        
        # Feature importance estimation (basic)
        if 'sales_quantity_log' in df.columns:
            feature_target_corr = df[modeling_features + ['sales_quantity_log']].corr()['sales_quantity_log'].abs()
            low_correlation_features = feature_target_corr[feature_target_corr < 0.01].index.tolist()
            validation_results['low_target_correlation_features'] = low_correlation_features
        
        # Data quality summary
        validation_results['total_features'] = len(modeling_features)
        validation_results['total_samples'] = len(df)
        validation_results['feature_density'] = 1 - missing_ratios.mean()
        validation_results['validation_passed'] = (
            len(high_missing_features) == 0 and 
            len(inf_features) == 0 and 
            len(zero_var_features) == 0
        )
        
        # Log validation summary
        logger.info(f"Validation Results:")
        logger.info(f"  Total features: {validation_results['total_features']}")
        logger.info(f"  High missing features: {len(high_missing_features)}")
        logger.info(f"  Infinite value features: {len(inf_features)}")
        logger.info(f"  Zero variance features: {len(zero_var_features)}")
        logger.info(f"  Highly skewed features: {len(skewed_features)}")
        logger.info(f"  High correlation pairs: {len(high_corr_pairs)}")
        logger.info(f"  Feature density: {validation_results['feature_density']:.3f}")
        logger.info(f"  Validation passed: {validation_results['validation_passed']}")
        
        return validation_results
    
    def save_engineered_dataset(self, 
                               df: pd.DataFrame, 
                               modeling_features: List[str],
                               rolling_splits: List[Tuple],
                               quality_metrics: Dict,
                               validation_results: Dict) -> str:
        """Save the complete engineered dataset with comprehensive metadata."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive dataset package
        dataset_package = {
            'df_final': df,
            'modeling_features': modeling_features,
            'rolling_splits': rolling_splits,
            'feature_categories': self._categorize_features_for_modeling(df, modeling_features),
            'metadata': {
                'creation_timestamp': timestamp,
                'total_records': len(df),
                'total_features': len(modeling_features),
                'date_range': {
                    'start': df['sales_month'].min().strftime('%Y-%m-%d'),
                    'end': df['sales_month'].max().strftime('%Y-%m-%d')
                },
                'platforms': df['primary_platform'].unique().tolist(),
                'stores': df['store_name'].nunique(),
                'brands': df['brand_name'].nunique(),
                'products': df['product_code'].nunique(),
                'quality_metrics': quality_metrics,
                'validation_results': validation_results,
                'pipeline_version': '2.0',
                'feature_engineering_steps': [
                    'Basic Setup and Data Preparation',
                    'Temporal Features (Advanced Temporal Patterns)',
                    'Customer Behavior Features (Store and Brand Analytics)',
                    'Store Categorization and Platform Dynamics',
                    'Advanced Interactions and Derived Features',
                    'Final Feature Consolidation and Cleanup'
                ]
            }
        }
        
        # Save main pickle file
        main_filename = f"sales_forecast_engineered_dataset_{timestamp}.pkl"
        main_filepath = self.output_dir / main_filename
        
        with open(main_filepath, 'wb') as f:
            pickle.dump(dataset_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save CSV for external analysis
        csv_filename = f"sales_forecast_engineered_data_{timestamp}.csv"
        csv_filepath = self.output_dir / csv_filename
        df.to_csv(csv_filepath, index=False)
        
        # Save feature list
        features_filename = f"modeling_features_{timestamp}.txt"
        features_filepath = self.output_dir / features_filename
        
        with open(features_filepath, 'w') as f:
            f.write("MODELING FEATURES\n")
            f.write("="*50 + "\n\n")
            
            feature_categories = self._categorize_features_for_modeling(df, modeling_features)
            for category, features in feature_categories.items():
                if features:
                    f.write(f"{category.upper()} ({len(features)} features):\n")
                    for feature in features:
                        f.write(f"  - {feature}\n")
                    f.write("\n")
            
            f.write(f"TOTAL FEATURES: {len(modeling_features)}\n")
        
        # Save metadata summary
        metadata_filename = f"dataset_metadata_{timestamp}.txt"
        metadata_filepath = self.output_dir / metadata_filename
        
        with open(metadata_filepath, 'w') as f:
            f.write("SALES FORECASTING DATASET METADATA\n")
            f.write("="*50 + "\n\n")
            
            metadata = dataset_package['metadata']
            f.write(f"Creation Time: {metadata['creation_timestamp']}\n")
            f.write(f"Pipeline Version: {metadata['pipeline_version']}\n")
            f.write(f"Total Records: {metadata['total_records']:,}\n")
            f.write(f"Total Features: {metadata['total_features']}\n")
            f.write(f"Date Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}\n")
            f.write(f"Platforms: {', '.join(metadata['platforms'])}\n")
            f.write(f"Unique Stores: {metadata['stores']:,}\n")
            f.write(f"Unique Brands: {metadata['brands']:,}\n")
            f.write(f"Unique Products: {metadata['products']:,}\n\n")
            
            f.write("FEATURE ENGINEERING STEPS:\n")
            for i, step in enumerate(metadata['feature_engineering_steps'], 1):
                f.write(f"{i}. {step}\n")
            f.write("\n")
            
            f.write("VALIDATION RESULTS:\n")
            val_results = metadata['validation_results']
            f.write(f"Validation Passed: {val_results['validation_passed']}\n")
            f.write(f"Feature Density: {val_results['feature_density']:.3f}\n")
            f.write(f"High Missing Features: {len(val_results['high_missing_features'])}\n")
            f.write(f"Zero Variance Features: {len(val_results['zero_variance_features'])}\n")
            
            f.write(f"\nROLLING SPLITS: {len(rolling_splits)}\n")
            for i, (train, val, description) in enumerate(rolling_splits):
                f.write(f"Split {i+1}: {len(train):,} train, {len(val):,} val ({description})\n")
        
        logger.info(f"Engineered dataset saved successfully:")
        logger.info(f"  Main file: {main_filepath}")
        logger.info(f"  CSV file: {csv_filepath}")
        logger.info(f"  Features file: {features_filepath}")
        logger.info(f"  Metadata file: {metadata_filepath}")
        
        return str(main_filepath)
    
    def _print_comprehensive_feature_summary(self, df: pd.DataFrame, original_cols: int) -> None:
        """Print a comprehensive summary of all engineered features."""
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 80)
        
        # Count features by type (matching notebook categories)
        feature_categories = {
            'Temporal Basic': [col for col in df.columns if any(x in col for x in ['month', 'quarter', 'year', 'days_since'])],
            'Cyclical': [col for col in df.columns if any(x in col for x in ['sin', 'cos'])],
            'Promotional': [col for col in df.columns if any(x in col for x in ['promotional', 'distance_to_promo', 'promo'])],
            'Seasonal': [col for col in df.columns if 'seasonal' in col and 'interaction' not in col],
            'Year-over-Year': [col for col in df.columns if 'yoy_' in col],
            'Lag Features': [col for col in df.columns if 'lag_' in col],
            'Rolling Windows': [col for col in df.columns if 'rolling_' in col],
            'Momentum': [col for col in df.columns if any(x in col for x in ['momentum', 'acceleration', 'pct_change', 'volatility'])],
            'Store Behavior': [col for col in df.columns if any(x in col for x in ['store_sales_cv', 'store_sales_range', 'store_consistency'])],
            'Brand Analytics': [col for col in df.columns if any(x in col for x in ['brand_market_share', 'brand_diversity', 'brand_performance'])],
            'Store Types': [col for col in df.columns if 'store_type_' in col],
            'Platform Dynamics': [col for col in df.columns if any(x in col for x in ['platform_', 'cross_platform', 'multi_platform'])],
            'Customer Behavior': [col for col in df.columns if any(x in col for x in ['relationship_', 'loyalty', 'consistency'])],
            'Competitive Intel': [col for col in df.columns if any(x in col for x in ['competitive_', 'market_share', 'rank'])],
            'Interactions': [col for col in df.columns if 'interaction' in col],
            'Spike Detection': [col for col in df.columns if any(x in col for x in ['spike_', 'deviation', 'z_score'])]
        }
        
        total_engineered = 0
        for category, features in feature_categories.items():
            if features:
                logger.info(f"{category:20s}: {len(features):3d} features")
                total_engineered += len(features)
        
        logger.info("-" * 80)
        logger.info(f"{'TOTAL ENGINEERED':20s}: {total_engineered:3d} features")
        logger.info(f"{'ORIGINAL COLUMNS':20s}: {original_cols:3d}")
        logger.info(f"{'FINAL DATASET':20s}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"{'NET FEATURE GAIN':20s}: {df.shape[1] - original_cols:3d} features")
        logger.info("=" * 80)
    
    def load_engineered_dataset(self, pickle_filepath: str) -> Tuple[pd.DataFrame, List[str], List[Tuple], Dict]:
        """
        Load a previously saved engineered dataset.
        
        Args:
            pickle_filepath: Path to the saved pickle file
            
        Returns:
            Tuple of (dataframe, modeling_features, rolling_splits, metadata)
        """
        logger.info(f"Loading engineered dataset from: {pickle_filepath}")
        
        with open(pickle_filepath, 'rb') as f:
            dataset_package = pickle.load(f)
        
        df = dataset_package['df_final']
        modeling_features = dataset_package['modeling_features']
        rolling_splits = dataset_package['rolling_splits']
        metadata = dataset_package['metadata']
        
        logger.info("✓ Dataset loaded successfully")
        logger.info(f"Records: {len(df):,}")
        logger.info(f"Features: {len(modeling_features)}")
        logger.info(f"Rolling Splits: {len(rolling_splits)}")
        logger.info(f"Created: {metadata['creation_timestamp']}")
        
        return df, modeling_features, rolling_splits, metadata