"""
Shared utilities for data processing.

This module contains all common constants, mappings, and utility functions
used across the data processing pipeline to eliminate code duplication.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# ===== LOGGING CONFIGURATION =====
def setup_logging(level=logging.INFO):
    """Set up consistent logging across all modules."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# ===== COLUMN MAPPINGS =====
COLUMN_MAPPING = {
    # Chinese to English column names
    '销售月份': 'sales_month',
    '主营平台': 'primary_platform', 
    '副营平台': 'secondary_platform',
    '店铺名称': 'store_name',
    '品牌名称': 'brand_name',
    '商品编码': 'product_code',
    '销售金额': 'sales_amount',
    '销售数量': 'sales_quantity',
    # Alternative mappings
    '一级平台': 'primary_platform',
    '二级平台': 'secondary_platform'
}

# Standard column order for consistent processing
STANDARD_COLUMNS = [
    'sales_month', 'primary_platform', 'secondary_platform',
    'store_name', 'brand_name', 'product_code',
    'sales_amount', 'sales_quantity'
]

# ===== PLATFORM MAPPINGS =====
PLATFORM_MAPPING = {
    '抖音': 'Douyin',
    '京东': 'JD',
    '天猫': 'Tmall'
}

# ===== BUSINESS CONSTANTS =====
PROMOTIONAL_MONTHS = [1, 6, 9, 11, 12]  # Chinese e-commerce promotional periods

CHINESE_STORE_TYPES = {
    '官方旗舰店': 'Official Flagship',
    '卖场旗舰店': 'Mall Flagship',
    '旗舰店': 'Flagship',
    '专卖店': 'Specialty Store',
    '专营店': 'Specialty Store',
    '卖场店': 'Mall Store',
    '自营': 'Platform Direct',
    '超市': 'Supermarket'
}

# ===== FEATURE GROUPS =====
EXCLUDE_FROM_MODELING = {
    'sales_quantity', 'sales_amount', 'sales_month',
    'store_name', 'brand_name', 'primary_platform',
    'secondary_platform', 'product_code'
}

# ===== UTILITY FUNCTIONS =====

def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate Chinese column names to English.
    
    Args:
        df: DataFrame with potentially Chinese column names
        
    Returns:
        DataFrame with English column names
    """
    logger = logging.getLogger(__name__)
    
    original_columns = df.columns.tolist()
    logger.info(f"Original columns: {original_columns}")
    
    # Try exact mapping first
    if any(col in df.columns for col in COLUMN_MAPPING.keys()):
        df = df.rename(columns=COLUMN_MAPPING)
        logger.info("Used exact column name mapping")
    else:
        # Fallback: positional mapping
        logger.info("Using positional column mapping")
        if len(original_columns) >= len(STANDARD_COLUMNS):
            new_columns = STANDARD_COLUMNS + original_columns[len(STANDARD_COLUMNS):]
            df.columns = new_columns
        else:
            # Handle incomplete data
            df.columns = STANDARD_COLUMNS[:len(original_columns)]
    
    logger.info(f"Final columns: {df.columns.tolist()}")
    return df

def map_platforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Chinese platform names to English.
    
    Args:
        df: DataFrame with platform columns
        
    Returns:
        DataFrame with English platform names
    """
    logger = logging.getLogger(__name__)
    
    df = df.copy()
    
    # Map primary platform
    df['primary_platform'] = df['primary_platform'].map(PLATFORM_MAPPING).fillna(df['primary_platform'])
    
    # Map secondary platform if it exists
    if 'secondary_platform' in df.columns:
        df['secondary_platform'] = df['secondary_platform'].map(PLATFORM_MAPPING).fillna(df['secondary_platform'])
    
    platform_counts = df['primary_platform'].value_counts().to_dict()
    logger.info(f"Platform distribution: {platform_counts}")
    
    return df

def categorize_store(store_name: str) -> str:
    """
    Categorize Chinese store names into English types.
    
    Args:
        store_name: Chinese store name
        
    Returns:
        English store type category
    """
    if pd.isna(store_name):
        return 'Unknown'
    
    store_name = str(store_name)
    
    # Check for exact matches first
    for chinese_type, english_type in CHINESE_STORE_TYPES.items():
        if chinese_type in store_name:
            return english_type
    
    # Default category
    return 'Other'

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN and then fill appropriately.
    
    Args:
        df: DataFrame with potential infinite values
        
    Returns:
        DataFrame with infinite values cleaned
    """
    # Replace infinite values with NaN
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    return df

def calculate_distance_to_events(month: int, event_months: List[int]) -> int:
    """
    Calculate the minimum circular distance to promotional events.
    
    Args:
        month: Current month (1-12)
        event_months: List of event months
        
    Returns:
        Minimum distance to any event month
    """
    if not event_months:
        return 6  # Maximum distance
    
    distances = []
    for event_month in event_months:
        # Calculate circular distance (considering year wraparound)
        distance = min(abs(month - event_month), 12 - abs(month - event_month))
        distances.append(distance)
    
    return min(distances)

def get_modeling_features(df: pd.DataFrame) -> List[str]:
    """
    Get comprehensive list of features suitable for modeling (exclude identifiers and targets).
    Aligned with data_engineer.ipynb feature categories.
    
    Args:
        df: DataFrame with all features
        
    Returns:
        List of feature column names for modeling
    """
    # Exclude columns that are not features (identifiers, targets, intermediate columns)
    exclude_columns = EXCLUDE_FROM_MODELING.union({
        'store_type',  # Categorical version - we use the one-hot encoded versions
        'store_size_category',  # Categorical version
        'month_year'  # Period object - not suitable for modeling directly
    })
    
    # Include all other columns as modeling features
    modeling_features = [
        col for col in df.columns 
        if col not in exclude_columns 
        and not col.startswith('Unnamed')
        and pd.api.types.is_numeric_dtype(df[col])  # Only numeric features
    ]
    
    # Log feature summary
    logger = logging.getLogger(__name__)
    logger.info(f"Identified {len(modeling_features)} modeling features from {df.shape[1]} total columns")
    
    return modeling_features

def validate_data_integrity(df: pd.DataFrame) -> Dict:
    """
    Perform basic data integrity checks.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    
    checks = {
        'total_records': len(df),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'negative_sales': (df['sales_quantity'] < 0).sum() if 'sales_quantity' in df.columns else 0,
        'zero_sales': (df['sales_quantity'] == 0).sum() if 'sales_quantity' in df.columns else 0,
        'date_range': (df['sales_month'].min(), df['sales_month'].max()) if 'sales_month' in df.columns else None
    }
    
    logger.info("Data integrity check results:")
    for key, value in checks.items():
        logger.info(f"  {key}: {value}")
    
    return checks 