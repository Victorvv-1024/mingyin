"""
Temporal feature engineering for sales forecasting.

This module focuses exclusively on time-based features including:
- Cyclical encoding
- Seasonality patterns
- Promotional periods
- Trend analysis
- Advanced lag and rolling features
- Momentum and acceleration
- Year-over-year growth
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from ..utils import setup_logging, PROMOTIONAL_MONTHS, calculate_distance_to_events

logger = setup_logging()

class TemporalFeatureEngineer:
    """Handles all temporal feature engineering."""
    
    def __init__(self):
        self.promotional_months = PROMOTIONAL_MONTHS
    
    def create_basic_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic temporal components."""
        logger.info("Creating basic temporal features...")
        
        df = df.copy()
        
        # Convert to datetime if not already
        df['sales_month'] = pd.to_datetime(df['sales_month'])
        
        # Basic temporal components
        df['month'] = df['sales_month'].dt.month
        df['quarter'] = df['sales_month'].dt.quarter
        df['year'] = df['sales_month'].dt.year
        df['month_year'] = df['sales_month'].dt.to_period('M')
        
        # Days since start for trend analysis
        start_date = df['sales_month'].min()
        df['days_since_start'] = (df['sales_month'] - start_date).dt.days
        
        logger.info("✓ Basic temporal features created")
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for temporal features."""
        logger.info("Creating cyclical temporal features...")
        
        df = df.copy()
        
        # Cyclical encoding for months (crucial for neural networks)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cyclical encoding for quarters
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        logger.info("✓ Cyclical features created")
        return df
    
    def create_promotional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Chinese e-commerce promotional period features."""
        logger.info("Creating promotional period features...")
        
        df = df.copy()
        
        # Binary promotional indicator
        df['is_promotional'] = df['month'].isin(self.promotional_months).astype(int)
        
        # Distance to nearest promotional period
        df['distance_to_promo'] = df['month'].apply(
            lambda x: calculate_distance_to_events(x, self.promotional_months)
        )
        
        logger.info("✓ Promotional features created")
        return df
    
    def create_learned_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create data-driven seasonal patterns."""
        logger.info("Creating learned seasonality features...")
        
        df = df.copy()
        
        # Overall monthly intensity (learned from data)
        monthly_baseline = df.groupby('month')['sales_quantity'].mean()
        overall_mean = df['sales_quantity'].mean()
        monthly_intensity = (monthly_baseline / overall_mean).to_dict()
        df['monthly_intensity_learned'] = df['month'].map(monthly_intensity)
        
        # Platform-specific seasonal patterns
        platform_monthly_patterns = df.groupby(['primary_platform', 'month'])['sales_quantity'].mean()
        platform_overall_means = df.groupby('primary_platform')['sales_quantity'].mean()
        
        def get_platform_seasonal_index(row):
            platform = row['primary_platform']
            month = row['month']
            pattern_value = platform_monthly_patterns.get(
                (platform, month), 
                platform_overall_means.get(platform, overall_mean)
            )
            baseline = platform_overall_means.get(platform, overall_mean)
            return pattern_value / baseline if baseline > 0 else 1
        
        df['platform_seasonal_index'] = df.apply(get_platform_seasonal_index, axis=1)
        
        logger.info("✓ Learned seasonality features created")
        return df
    
    def create_year_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create year-specific indicator features."""
        logger.info("Creating year indicator features...")
        
        df = df.copy()
        
        # Year indicators (useful for capturing year-over-year effects)
        df['is_2021'] = (df['year'] == 2021).astype(int)
        df['is_2022'] = (df['year'] == 2022).astype(int)
        df['is_2023'] = (df['year'] == 2023).astype(int)
        
        logger.info("✓ Year indicators created")
        return df
    
    def create_year_over_year_features(self, df: pd.DataFrame, entity_cols: List[str] = None) -> pd.DataFrame:
        """Create year-over-year growth and comparison features."""
        logger.info("Creating year-over-year growth features...")
        
        df = df.copy()
        
        # Default entity columns for grouping
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper calculation
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create year-over-year comparison features
        yoy_comparison = df.groupby(entity_cols + ['month']).agg({
            'sales_quantity': 'mean'
        }).reset_index()
        
        # For each combination, find baseline (previous year) values
        yoy_2021 = yoy_comparison[yoy_comparison['month'].isin(range(4, 13))].copy()  # Douyin available months
        yoy_2021['comparison_key'] = (
            yoy_2021['primary_platform'] + '_' + 
            yoy_2021['store_name'] + '_' + 
            yoy_2021['brand_name'] + '_' + 
            yoy_2021['month'].astype(str)
        )
        yoy_baseline = yoy_2021.set_index('comparison_key')['sales_quantity'].to_dict()
        
        df['comparison_key'] = (
            df['primary_platform'] + '_' + 
            df['store_name'] + '_' + 
            df['brand_name'] + '_' + 
            df['month'].astype(str)
        )
        df['yoy_baseline'] = df['comparison_key'].map(yoy_baseline)
        df['yoy_growth_potential'] = np.where(
            (df['year'] == 2022) & (df['yoy_baseline'].notna()),
            df['sales_quantity'] / df['yoy_baseline'],
            1
        )
        
        # Clean up temporary columns
        df = df.drop(['comparison_key', 'yoy_baseline'], axis=1)
        
        logger.info("✓ Year-over-year growth features created")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, entity_cols: List[str] = None) -> pd.DataFrame:
        """Create comprehensive lag features for time series analysis."""
        logger.info("Creating comprehensive lag features...")
        
        df = df.copy()
        
        # Default entity columns for grouping
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper lag calculation
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create unique identifier for grouping
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Create lag features for different periods - comprehensive set
        lag_periods = [1, 2, 3, 6, 12]
        for lag in lag_periods:
            df[f'sales_lag_{lag}'] = df.groupby('entity_key')['sales_quantity'].shift(lag)
        
        # Clean up temporary columns
        df = df.drop(['entity_key'], axis=1)
        
        logger.info(f"✓ Lag features created for periods: {lag_periods}")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, entity_cols: List[str] = None) -> pd.DataFrame:
        """Create comprehensive rolling window statistical features."""
        logger.info("Creating comprehensive rolling window features...")
        
        df = df.copy()
        
        # Default entity columns for grouping
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper rolling calculation
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create unique identifier for grouping
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Rolling windows
        rolling_windows = [3, 6, 12]
        
        for window in rolling_windows:
            # Rolling mean (trend)
            df[f'sales_rolling_mean_{window}'] = (
                df.groupby('entity_key')['sales_quantity']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling standard deviation (volatility)
            df[f'sales_rolling_std_{window}'] = (
                df.groupby('entity_key')['sales_quantity']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            ).fillna(0)
            
            # Rolling min and max (range patterns)
            df[f'sales_rolling_min_{window}'] = (
                df.groupby('entity_key')['sales_quantity']
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )
            
            df[f'sales_rolling_max_{window}'] = (
                df.groupby('entity_key')['sales_quantity']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
        
        # Clean up temporary columns
        df = df.drop(['entity_key'], axis=1)
        
        logger.info(f"✓ Rolling features created for windows: {rolling_windows}")
        return df
    
    def create_momentum_features(self, df: pd.DataFrame, entity_cols: List[str] = None) -> pd.DataFrame:
        """Create comprehensive momentum and trend indicators."""
        logger.info("Creating comprehensive momentum features...")
        
        df = df.copy()
        
        # Default entity columns for grouping
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Create unique identifier for grouping
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Percentage change features (growth rates)
        for period in [1, 3, 6]:
            df[f'sales_pct_change_{period}'] = (
                df.groupby('entity_key')['sales_quantity']
                .pct_change(periods=period)
                .fillna(0)
            )
        
        # Momentum indicators (comparing different time horizons)
        df['sales_momentum_short'] = np.where(
            df['sales_rolling_mean_3'] != 0,
            df['sales_rolling_mean_6'] / df['sales_rolling_mean_3'],
            1
        )  # 6-month trend vs 3-month trend
        
        df['sales_momentum_long'] = np.where(
            df['sales_rolling_mean_6'] != 0,
            df['sales_rolling_mean_12'] / df['sales_rolling_mean_6'],
            1
        )  # 12-month trend vs 6-month trend
        
        # Acceleration (second derivative - change in growth rate)
        df['sales_acceleration'] = (
            df['sales_pct_change_1'] - 
            df.groupby('entity_key')['sales_pct_change_1'].shift(1)
        ).fillna(0)
        
        # Volatility indicators
        df['sales_volatility_ratio'] = np.where(
            df['sales_rolling_mean_6'] != 0,
            df['sales_rolling_std_6'] / df['sales_rolling_mean_6'],
            0
        )  # Coefficient of variation over 6 months
        
        # Clean up temporary columns
        df = df.drop(['entity_key'], axis=1)
        
        logger.info("✓ Momentum features created")
        return df
    
    def create_outlier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create outlier and spike detection features."""
        logger.info("Creating outlier and spike detection features...")
        
        df = df.copy()
        
        # Historical spike detection
        spike_threshold_99 = df['sales_quantity'].quantile(0.99)
        spike_threshold_95 = df['sales_quantity'].quantile(0.95)
        
        df['is_extreme_spike'] = (df['sales_quantity'] > spike_threshold_99).astype(int)
        df['is_major_spike'] = (df['sales_quantity'] > spike_threshold_95).astype(int)
        
        # Create unique identifier for spike history
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Store-brand spike history
        spike_history = df.groupby('entity_key').agg({
            'is_extreme_spike': 'sum',
            'is_major_spike': 'sum'
        }).rename(columns={
            'is_extreme_spike': 'historical_extreme_spikes',
            'is_major_spike': 'historical_major_spikes'
        })
        
        df = df.merge(spike_history, on='entity_key', how='left')
        
        # Spike propensity score
        df['spike_propensity'] = df['historical_major_spikes'] / df.groupby('entity_key').cumcount().add(1)
        
        # Distance from normal behavior
        df['deviation_from_rolling_mean'] = np.where(
            df['sales_rolling_mean_6'] > 0,
            (df['sales_quantity'] - df['sales_rolling_mean_6']) / df['sales_rolling_mean_6'],
            0
        )
        
        # Z-score based on rolling statistics  
        df['rolling_z_score'] = np.where(
            df['sales_rolling_std_6'] > 0,
            (df['sales_quantity'] - df['sales_rolling_mean_6']) / df['sales_rolling_std_6'],
            0
        )
        
        # Clean up temporary columns
        df = df.drop(['entity_key'], axis=1)
        
        logger.info("✓ Outlier and spike detection features created")
        return df
    
    def engineer_all_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all temporal feature engineering."""
        logger.info("=== TEMPORAL FEATURE ENGINEERING ===")
        
        original_cols = len(df.columns)
        
        # Apply all temporal feature engineering in logical order
        df = self.create_basic_temporal_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_promotional_features(df)
        df = self.create_learned_seasonality(df)
        df = self.create_year_indicators(df)
        df = self.create_year_over_year_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_outlier_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"✓ Temporal engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df 