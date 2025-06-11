"""
Comprehensive temporal feature engineering for sales forecasting.

This module implements sophisticated time-based features including:
- Basic temporal components (month, quarter, year)
- Cyclical transformations for seasonality
- Chinese e-commerce promotional calendar integration
- Year-over-year comparisons
- Lag and rolling window features
- Momentum and trend indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from .promotional_calendar import ChineseEcommerceCalendar

logger = logging.getLogger(__name__)

class TemporalFeatureEngineer:
    """Comprehensive temporal feature engineering for Chinese e-commerce sales data."""
    
    def __init__(self):
        """Initialize the temporal feature engineer."""
        self.promotional_calendar = ChineseEcommerceCalendar()
        logger.info("TemporalFeatureEngineer initialized")
    
    def engineer_all_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all temporal features in the correct sequence.
        
        Args:
            df: DataFrame with sales_month column
            
        Returns:
            DataFrame with comprehensive temporal features
        """
        logger.info("Starting comprehensive temporal feature engineering...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Ensure datetime conversion
        df['sales_month'] = pd.to_datetime(df['sales_month'])
        
        # Step 1: Basic temporal components
        df = self._create_basic_temporal_features(df)
        
        # Step 2: Cyclical transformations
        df = self._create_cyclical_features(df)
        
        # Step 3: Promotional calendar features
        df = self._create_promotional_features(df)
        
        # Step 4: Year-over-year features
        df = self._create_yoy_features(df)
        
        # Step 5: Lag features
        df = self._create_lag_features(df)
        
        # Step 6: Rolling window features
        df = self._create_rolling_features(df)
        
        # Step 7: Momentum and trend features
        df = self._create_momentum_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"Temporal feature engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        return df
    
    def _create_basic_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic temporal components."""
        logger.info("Creating basic temporal features...")
        
        # Basic components
        df['month'] = df['sales_month'].dt.month
        df['quarter'] = df['sales_month'].dt.quarter
        df['year'] = df['sales_month'].dt.year
        df['day_of_year'] = df['sales_month'].dt.dayofyear
        df['week_of_year'] = df['sales_month'].dt.isocalendar().week
        
        # Days since epoch (for trend analysis)
        epoch = pd.Timestamp('2021-01-01')
        df['days_since_epoch'] = (df['sales_month'] - epoch).dt.days
        
        # Month within year progressions
        df['month_progress'] = (df['month'] - 1) / 11  # 0 to 1 scale
        df['quarter_progress'] = (df['quarter'] - 1) / 3  # 0 to 1 scale
        
        logger.info("✓ Basic temporal features created")
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical transformations for proper seasonality encoding."""
        logger.info("Creating cyclical features...")
        
        # Monthly cyclical features
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # Quarterly cyclical features
        df['quarter_sin'] = np.sin(2 * np.pi * (df['quarter'] - 1) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * (df['quarter'] - 1) / 4)
        
        # Day of year cyclical features
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Week of year cyclical features
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        logger.info("✓ Cyclical features created")
        return df
    
    def _create_promotional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Chinese e-commerce promotional calendar features."""
        logger.info("Creating promotional calendar features...")
        
        # Add promotional period indicators
        df['is_promotional_period'] = df['sales_month'].apply(
            self.promotional_calendar.is_promotional_period
        ).astype(int)
        
        # Add specific promotional events
        promo_events = self.promotional_calendar.get_promotional_events()
        for event_name, _ in promo_events.items():
            df[f'is_{event_name.lower().replace(" ", "_")}'] = df['sales_month'].apply(
                lambda x: self.promotional_calendar.is_specific_event(x, event_name)
            ).astype(int)
        
        # Distance to next/previous promotional period
        df['days_to_next_promo'] = df['sales_month'].apply(
            self.promotional_calendar.days_to_next_promotion
        )
        df['days_from_last_promo'] = df['sales_month'].apply(
            self.promotional_calendar.days_from_last_promotion
        )
        
        # Promotional intensity (how close to major events)
        df['promotional_intensity'] = df['sales_month'].apply(
            self.promotional_calendar.get_promotional_intensity
        )
        
        # Seasonal promotional patterns
        df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
        df['is_mid_year_shopping'] = ((df['month'] == 6) | (df['month'] == 7)).astype(int)
        df['is_back_to_school'] = ((df['month'] == 8) | (df['month'] == 9)).astype(int)
        
        logger.info("✓ Promotional features created")
        return df
    
    def _create_yoy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create year-over-year comparison features."""
        logger.info("Creating year-over-year features...")
        
        # Sort for proper YoY calculations
        df = df.sort_values(['primary_platform', 'store_name', 'brand_name', 'sales_month'])
        
        # Create entity grouping key
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # YoY sales comparison (12 months ago)
        df['sales_yoy'] = df.groupby('entity_key')['sales_quantity'].shift(12)
        df['sales_yoy_growth'] = (df['sales_quantity'] - df['sales_yoy']) / (df['sales_yoy'] + 1)
        df['sales_yoy_growth'] = df['sales_yoy_growth'].fillna(0)
        
        # YoY growth rate (percentage)
        df['sales_yoy_pct_change'] = df.groupby('entity_key')['sales_quantity'].pct_change(12).fillna(0)
        
        # YoY trend indicator
        df['yoy_trend_up'] = (df['sales_yoy_growth'] > 0.1).astype(int)
        df['yoy_trend_down'] = (df['sales_yoy_growth'] < -0.1).astype(int)
        df['yoy_trend_stable'] = ((df['sales_yoy_growth'] >= -0.1) & (df['sales_yoy_growth'] <= 0.1)).astype(int)
        
        # Seasonal consistency (same month, different years)
        df['seasonal_consistency'] = df.groupby(['entity_key', 'month'])['sales_quantity'].transform(
            lambda x: x.std() / (x.mean() + 1) if len(x) > 1 else 0
        ).fillna(0)
        
        # Clean up temporary column
        df = df.drop(['entity_key'], axis=1)
        
        logger.info("✓ Year-over-year features created")
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, 
                           entity_cols: List[str] = None) -> pd.DataFrame:
        """Create comprehensive lag features."""
        logger.info("Creating lag features...")
        
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper lag calculation
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create grouping key
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Lag periods matching the notebook
        lag_periods = [1, 2, 3, 6, 12]
        
        for lag in lag_periods:
            # Sales quantity lags
            df[f'sales_lag_{lag}'] = df.groupby('entity_key')['sales_quantity'].shift(lag)
            
            # Sales amount lags (for additional context)
            df[f'amount_lag_{lag}'] = df.groupby('entity_key')['sales_amount'].shift(lag)
            
            # Unit price lags
            if 'unit_price' in df.columns:
                df[f'price_lag_{lag}'] = df.groupby('entity_key')['unit_price'].shift(lag)
        
        # Fill NaN values with 0 for lag features
        lag_cols = [col for col in df.columns if '_lag_' in col]
        df[lag_cols] = df[lag_cols].fillna(0)
        
        # Clean up temporary column
        df = df.drop(['entity_key'], axis=1)
        
        logger.info(f"✓ Lag features created for periods: {lag_periods}")
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, 
                               entity_cols: List[str] = None) -> pd.DataFrame:
        """Create rolling window statistical features."""
        logger.info("Creating rolling window features...")
        
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper rolling calculations
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create grouping key
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Rolling windows matching the notebook
        rolling_windows = [3, 6, 12]
        
        for window in rolling_windows:
            # Rolling statistics for sales quantity
            rolling_group = df.groupby('entity_key')['sales_quantity'].rolling(
                window=window, min_periods=1
            )
            
            df[f'sales_rolling_mean_{window}'] = rolling_group.mean().reset_index(level=0, drop=True)
            df[f'sales_rolling_std_{window}'] = rolling_group.std().reset_index(level=0, drop=True).fillna(0)
            df[f'sales_rolling_min_{window}'] = rolling_group.min().reset_index(level=0, drop=True)
            df[f'sales_rolling_max_{window}'] = rolling_group.max().reset_index(level=0, drop=True)
            
            # Rolling coefficient of variation (volatility measure)
            df[f'sales_rolling_cv_{window}'] = (
                df[f'sales_rolling_std_{window}'] / (df[f'sales_rolling_mean_{window}'] + 1)
            )
            
            # Rolling median and quantiles
            df[f'sales_rolling_median_{window}'] = rolling_group.median().reset_index(level=0, drop=True)
            df[f'sales_rolling_q25_{window}'] = rolling_group.quantile(0.25).reset_index(level=0, drop=True)
            df[f'sales_rolling_q75_{window}'] = rolling_group.quantile(0.75).reset_index(level=0, drop=True)
        
        # Fill NaN values
        rolling_cols = [col for col in df.columns if 'rolling_' in col]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        
        # Clean up temporary column
        df = df.drop(['entity_key'], axis=1)
        
        logger.info(f"✓ Rolling features created for windows: {rolling_windows}")
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame, 
                                entity_cols: List[str] = None) -> pd.DataFrame:
        """Create momentum and trend indicators."""
        logger.info("Creating momentum and trend features...")
        
        if entity_cols is None:
            entity_cols = ['primary_platform', 'store_name', 'brand_name']
        
        # Sort for proper momentum calculations
        df = df.sort_values(entity_cols + ['sales_month'])
        
        # Create grouping key
        df['entity_key'] = (
            df['store_name'].astype(str) + '_' + 
            df['brand_name'].astype(str) + '_' + 
            df['primary_platform'].astype(str)
        )
        
        # Month-over-month change
        df['sales_mom_change'] = df.groupby('entity_key')['sales_quantity'].pct_change().fillna(0)
        df['sales_mom_abs_change'] = df.groupby('entity_key')['sales_quantity'].diff().fillna(0)
        
        # Momentum indicators (3-month and 6-month trends)
        for window in [3, 6]:
            # Linear trend over window
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                try:
                    slope = np.polyfit(x, series, 1)[0]
                    return slope
                except:
                    return 0
            
            df[f'sales_trend_{window}m'] = df.groupby('entity_key')['sales_quantity'].rolling(
                window=window, min_periods=2
            ).apply(calculate_trend, raw=True).reset_index(level=0, drop=True).fillna(0)
        
        # Acceleration (change in momentum)
        df['sales_acceleration'] = df.groupby('entity_key')['sales_mom_change'].diff().fillna(0)
        
        # Volatility measures
        df['sales_volatility_3m'] = df.groupby('entity_key')['sales_quantity'].rolling(
            window=3, min_periods=1
        ).std().reset_index(level=0, drop=True).fillna(0)
        
        df['sales_volatility_6m'] = df.groupby('entity_key')['sales_quantity'].rolling(
            window=6, min_periods=1
        ).std().reset_index(level=0, drop=True).fillna(0)
        
        # Momentum direction indicators
        df['momentum_up'] = (df['sales_mom_change'] > 0.1).astype(int)
        df['momentum_down'] = (df['sales_mom_change'] < -0.1).astype(int)
        df['momentum_stable'] = ((df['sales_mom_change'] >= -0.1) & (df['sales_mom_change'] <= 0.1)).astype(int)
        
        # Clean up temporary column
        df = df.drop(['entity_key'], axis=1)
        
        logger.info("✓ Momentum and trend features created")
        return df
    
    def get_temporal_feature_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize temporal features for deep learning model input preparation.
        
        Returns:
            Dictionary mapping feature categories to feature lists
        """
        temporal_categories = {
            'basic_temporal': [
                'month', 'quarter', 'year', 'day_of_year', 'week_of_year',
                'month_progress', 'quarter_progress'
            ],
            'cyclical': [
                'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                'day_of_year_sin', 'day_of_year_cos', 'week_sin', 'week_cos'
            ],
            'promotional': [
                col for col in df.columns if any(x in col for x in [
                    'promotional', 'promo', 'holiday', 'shopping', 'school'
                ])
            ],
            'yoy': [
                col for col in df.columns if 'yoy' in col or 'seasonal_consistency' in col
            ],
            'lag': [
                col for col in df.columns if '_lag_' in col
            ],
            'rolling': [
                col for col in df.columns if 'rolling_' in col
            ],
            'momentum': [
                col for col in df.columns if any(x in col for x in [
                    'mom_', 'trend_', 'acceleration', 'volatility', 'momentum_'
                ])
            ]
        }
        
        # Filter to only include columns that exist in the dataframe
        filtered_categories = {}
        for category, features in temporal_categories.items():
            existing_features = [f for f in features if f in df.columns]
            if existing_features:
                filtered_categories[category] = existing_features
        
        return filtered_categories