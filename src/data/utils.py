"""
Utility functions for sales forecasting data processing.

This module provides common utility functions used across the data processing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('sales_forecasting')
    return logger

def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate Chinese column names to English.
    
    Args:
        df: DataFrame with potentially Chinese column names
        
    Returns:
        DataFrame with English column names
    """
    column_translation = {
        '销售月份': 'sales_month',
        '一级平台': 'primary_platform',
        '主要平台（原始）': 'primary_platform_original', 
        '二级平台': 'secondary_platform',
        '店铺名称': 'store_name',
        '品牌名称': 'brand_name',
        '商品编码': 'product_code',
        '销售金额': 'sales_amount',
        '销售数量': 'sales_quantity',
        '主要平台': 'primary_platform'
    }
    
    # Apply translation
    df_translated = df.rename(columns=column_translation)
    
    # Also handle any variations or English names that might need standardization
    standard_names = {
        'Sales Month': 'sales_month',
        'Primary Platform (Original)': 'primary_platform_original',
        'Secondary Platform': 'secondary_platform', 
        'Store Name': 'store_name',
        'Brand Name': 'brand_name',
        'Product Code': 'product_code',
        'Sales Amount': 'sales_amount',
        'Sales Quantity': 'sales_quantity',
        'Primary Platform': 'primary_platform'
    }
    
    df_translated = df_translated.rename(columns=standard_names)
    
    return df_translated

def map_platforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map platform names to standardized values.
    
    Args:
        df: DataFrame with platform columns
        
    Returns:
        DataFrame with standardized platform names
    """
    platform_mapping = {
        '京东': 'JD',
        'jd': 'JD',
        'JD.com': 'JD',
        '天猫': 'Tmall',
        'tmall': 'Tmall', 
        'Tmall.com': 'Tmall',
        '抖音': 'Douyin',
        'douyin': 'Douyin',
        'TikTok': 'Douyin',
        'tiktok': 'Douyin'
    }
    
    # Apply platform mapping to both original and processed platform columns
    if 'primary_platform_original' in df.columns:
        df['primary_platform_original'] = df['primary_platform_original'].map(
            platform_mapping
        ).fillna(df['primary_platform_original'])
    
    if 'primary_platform' in df.columns:
        df['primary_platform'] = df['primary_platform'].map(
            platform_mapping
        ).fillna(df['primary_platform'])
    else:
        # Create primary_platform from original if it doesn't exist
        if 'primary_platform_original' in df.columns:
            df['primary_platform'] = df['primary_platform_original'].map(
                platform_mapping
            ).fillna(df['primary_platform_original'])
    
    return df

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean infinite values from DataFrame.
    
    Args:
        df: DataFrame potentially containing infinite values
        
    Returns:
        DataFrame with infinite values handled
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Log any infinite values found
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        logger = logging.getLogger('sales_forecasting')
        logger.warning(f"Found infinite values in columns: {inf_counts}")
    
    return df

def validate_data_integrity(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data integrity and return quality metrics.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with data quality metrics
    """
    metrics = {}
    
    # Basic shape information
    metrics['total_rows'] = len(df)
    metrics['total_columns'] = len(df.columns)
    
    # Missing value analysis
    missing_counts = df.isnull().sum()
    metrics['missing_values'] = missing_counts.to_dict()
    metrics['total_missing'] = missing_counts.sum()
    metrics['missing_percentage'] = (missing_counts / len(df) * 100).to_dict()
    
    # Duplicate analysis
    metrics['duplicate_rows'] = df.duplicated().sum()
    
    # Data type analysis
    dtype_counts = df.dtypes.value_counts().to_dict()
    metrics['data_types'] = {str(k): v for k, v in dtype_counts.items()}
    
    # Numeric column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics['numeric_columns'] = len(numeric_cols)
        metrics['zero_values'] = (df[numeric_cols] == 0).sum().to_dict()
        metrics['negative_values'] = (df[numeric_cols] < 0).sum().to_dict()
        
        # Check for infinite values
        inf_values = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_values[col] = inf_count
        metrics['infinite_values'] = inf_values
    
    # Date column analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        metrics['date_columns'] = len(date_cols)
        for col in date_cols:
            metrics[f'{col}_date_range'] = {
                'min': df[col].min().strftime('%Y-%m-%d') if pd.notna(df[col].min()) else None,
                'max': df[col].max().strftime('%Y-%m-%d') if pd.notna(df[col].max()) else None
            }
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        metrics['categorical_columns'] = len(categorical_cols)
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Only for columns with reasonable number of unique values
                metrics[f'{col}_unique_values'] = df[col].value_counts().to_dict()
            else:
                metrics[f'{col}_unique_count'] = unique_count
    
    return metrics

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print a comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        title: Title for the summary
    """
    logger = logging.getLogger('sales_forecasting')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{title}")
    logger.info(f"{'='*60}")
    
    # Basic information
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"\nMissing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            percentage = count / len(df) * 100
            logger.info(f"  {col}: {count:,} ({percentage:.1f}%)")
    else:
        logger.info("\nNo missing values found ✓")
    
    # Data types
    logger.info(f"\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")
    
    # Numeric summaries for key columns
    if 'sales_quantity' in df.columns:
        logger.info(f"\nSales Quantity Summary:")
        logger.info(f"  Min: {df['sales_quantity'].min():,.0f}")
        logger.info(f"  Max: {df['sales_quantity'].max():,.0f}")
        logger.info(f"  Mean: {df['sales_quantity'].mean():,.0f}")
        logger.info(f"  Median: {df['sales_quantity'].median():,.0f}")
    
    if 'sales_amount' in df.columns:
        logger.info(f"\nSales Amount Summary:")
        logger.info(f"  Min: {df['sales_amount'].min():,.2f}")
        logger.info(f"  Max: {df['sales_amount'].max():,.2f}")
        logger.info(f"  Mean: {df['sales_amount'].mean():,.2f}")
    
    # Categorical summaries
    categorical_cols = ['primary_platform', 'store_name', 'brand_name']
    for col in categorical_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            logger.info(f"\n{col.replace('_', ' ').title()}:")
            logger.info(f"  Unique values: {unique_count:,}")
            if unique_count <= 10:
                value_counts = df[col].value_counts()
                for value, count in value_counts.items():
                    percentage = count / len(df) * 100
                    logger.info(f"    {value}: {count:,} ({percentage:.1f}%)")

def get_modeling_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of features suitable for modeling.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        List of feature column names suitable for modeling
    """
    # Exclude non-feature columns
    exclude_patterns = [
        'sales_month', 'store_name', 'brand_name', 'product_code',
        'primary_platform_original', 'secondary_platform',
        'sales_amount', 'sales_quantity',  # Keep only log-transformed version
        '_temp_', '_intermediate_', 'entity_key'
    ]
    
    # Include only numeric columns that are not in exclude patterns
    modeling_features = []
    for col in df.columns:
        # Skip non-numeric columns
        if df[col].dtype not in ['int32', 'int64', 'float32', 'float64', 'bool']:
            continue
            
        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
            
        # Skip target variable (but keep log-transformed version)
        if col == 'sales_quantity':
            continue
            
        modeling_features.append(col)
    
    return modeling_features

def create_feature_importance_analysis(df: pd.DataFrame, 
                                     features: List[str], 
                                     target: str = 'sales_quantity_log') -> Dict[str, float]:
    """
    Create basic feature importance analysis using correlation.
    
    Args:
        df: DataFrame with features
        features: List of feature column names
        target: Target variable name
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if target not in df.columns:
        return {}
    
    # Calculate correlation with target
    feature_importance = {}
    for feature in features:
        if feature in df.columns:
            corr = df[feature].corr(df[target])
            if not np.isnan(corr):
                feature_importance[feature] = abs(corr)
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    return feature_importance

def export_feature_metadata(df: pd.DataFrame, 
                          features: List[str], 
                          output_path: str) -> None:
    """
    Export comprehensive feature metadata to a file.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        output_path: Path to save metadata file
    """
    with open(output_path, 'w') as f:
        f.write("FEATURE METADATA REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Features: {len(features)}\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Date Range: {df['sales_month'].min()} to {df['sales_month'].max()}\n\n")
        
        # Feature statistics
        f.write("FEATURE STATISTICS:\n")
        f.write("-"*30 + "\n")
        
        for feature in features:
            if feature in df.columns:
                f.write(f"\n{feature}:\n")
                f.write(f"  Data Type: {df[feature].dtype}\n")
                f.write(f"  Missing Values: {df[feature].isnull().sum()} ({df[feature].isnull().mean()*100:.1f}%)\n")
                
                if df[feature].dtype in ['int32', 'int64', 'float32', 'float64']:
                    f.write(f"  Min: {df[feature].min():.4f}\n")
                    f.write(f"  Max: {df[feature].max():.4f}\n")
                    f.write(f"  Mean: {df[feature].mean():.4f}\n")
                    f.write(f"  Std: {df[feature].std():.4f}\n")
                elif df[feature].dtype == 'object':
                    f.write(f"  Unique Values: {df[feature].nunique()}\n")
                    if df[feature].nunique() <= 10:
                        f.write(f"  Values: {df[feature].unique().tolist()}\n")

def validate_temporal_integrity(df: pd.DataFrame, 
                              date_col: str = 'sales_month',
                              entity_cols: List[str] = None) -> Dict[str, any]:
    """
    Validate temporal integrity of the dataset.
    
    Args:
        df: DataFrame to validate
        date_col: Name of the date column
        entity_cols: List of entity columns for grouping
        
    Returns:
        Dictionary with temporal validation results
    """
    if entity_cols is None:
        entity_cols = ['primary_platform', 'store_name', 'brand_name']
    
    validation_results = {}
    
    # Check for chronological ordering
    df_sorted = df.sort_values(entity_cols + [date_col])
    is_chronological = df_sorted.equals(df.sort_values(entity_cols + [date_col]))
    validation_results['is_chronologically_ordered'] = is_chronological
    
    # Check for date gaps
    entity_groups = df.groupby(entity_cols)[date_col]
    
    gaps_found = []
    for name, group in entity_groups:
        dates = pd.to_datetime(group).sort_values()
        if len(dates) > 1:
            # Check for gaps larger than 2 months
            date_diffs = dates.diff().dt.days
            large_gaps = date_diffs[date_diffs > 62]  # ~2 months
            if len(large_gaps) > 0:
                gaps_found.append({
                    'entity': name if isinstance(name, tuple) else (name,),
                    'max_gap_days': large_gaps.max(),
                    'gap_count': len(large_gaps)
                })
    
    validation_results['large_date_gaps'] = gaps_found
    validation_results['entities_with_gaps'] = len(gaps_found)
    
    # Check date range coverage
    overall_date_range = {
        'start': df[date_col].min(),
        'end': df[date_col].max(),
        'total_months': ((df[date_col].max() - df[date_col].min()).days / 30.44)
    }
    validation_results['date_range'] = overall_date_range
    
    return validation_results

def prepare_features_for_deep_learning(df: pd.DataFrame, 
                                     features: List[str]) -> Dict[str, Dict]:
    """
    Prepare features for deep learning models by categorizing them appropriately.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        
    Returns:
        Dictionary with categorized features and their properties
    """
    feature_preparation = {
        'temporal_categorical': {
            'features': [],
            'embedding_dims': {},
            'vocab_sizes': {}
        },
        'temporal_continuous': {
            'features': [],
            'scaling_method': 'standard'
        },
        'cyclical': {
            'features': [],
            'scaling_method': 'none'  # Already normalized
        },
        'categorical_binary': {
            'features': [],
            'scaling_method': 'none'
        },
        'continuous_features': {
            'features': [],
            'scaling_method': 'robust'
        },
        'count_features': {
            'features': [],
            'scaling_method': 'standard'
        }
    }
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        # Temporal categorical (month, quarter, year)
        if feature in ['month', 'quarter', 'year']:
            feature_preparation['temporal_categorical']['features'].append(feature)
            if feature == 'month':
                feature_preparation['temporal_categorical']['vocab_sizes'][feature] = 12
                feature_preparation['temporal_categorical']['embedding_dims'][feature] = 8
            elif feature == 'quarter':
                feature_preparation['temporal_categorical']['vocab_sizes'][feature] = 4
                feature_preparation['temporal_categorical']['embedding_dims'][feature] = 4
            elif feature == 'year':
                unique_years = df[feature].nunique()
                feature_preparation['temporal_categorical']['vocab_sizes'][feature] = unique_years
                feature_preparation['temporal_categorical']['embedding_dims'][feature] = min(8, unique_years)
        
        # Cyclical features (sin/cos)
        elif any(x in feature for x in ['_sin', '_cos']):
            feature_preparation['cyclical']['features'].append(feature)
        
        # Binary/categorical features
        elif df[feature].dtype in ['bool'] or (df[feature].dtype in ['int32', 'int64'] and df[feature].max() <= 1):
            feature_preparation['categorical_binary']['features'].append(feature)
        
        # Count features
        elif any(x in feature for x in ['_count', '_nunique', '_rank']):
            feature_preparation['count_features']['features'].append(feature)
        
        # Temporal continuous
        elif any(x in feature for x in ['days_since', '_progress', 'tenure_months']):
            feature_preparation['temporal_continuous']['features'].append(feature)
        
        # Default to continuous
        else:
            feature_preparation['continuous_features']['features'].append(feature)
    
    return feature_preparation

def create_data_quality_report(df: pd.DataFrame, 
                             features: List[str],
                             output_path: str) -> None:
    """
    Create a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        features: List of features to include in report
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("DATA QUALITY REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Features: {len(features)}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
        
        if 'sales_month' in df.columns:
            f.write(f"Date Range: {df['sales_month'].min()} to {df['sales_month'].max()}\n")
        
        f.write(f"Platforms: {df['primary_platform'].unique().tolist() if 'primary_platform' in df.columns else 'N/A'}\n")
        f.write(f"Unique Stores: {df['store_name'].nunique() if 'store_name' in df.columns else 'N/A'}\n")
        f.write(f"Unique Brands: {df['brand_name'].nunique() if 'brand_name' in df.columns else 'N/A'}\n\n")
        
        # Missing values analysis
        f.write("MISSING VALUES ANALYSIS:\n")
        f.write("-"*30 + "\n")
        missing_summary = df[features].isnull().sum()
        missing_features = missing_summary[missing_summary > 0]
        
        if len(missing_features) > 0:
            f.write(f"Features with missing values: {len(missing_features)}\n")
            for feature, count in missing_features.items():
                percentage = count / len(df) * 100
                f.write(f"  {feature}: {count:,} ({percentage:.1f}%)\n")
        else:
            f.write("No missing values found ✓\n")
        f.write("\n")
        
        # Data type analysis
        f.write("DATA TYPES ANALYSIS:\n")
        f.write("-"*30 + "\n")
        dtype_counts = df[features].dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            f.write(f"{str(dtype)}: {count} features\n")
        f.write("\n")
        
        # Feature distribution analysis
        f.write("FEATURE DISTRIBUTION ANALYSIS:\n")
        f.write("-"*30 + "\n")
        
        numeric_features = [f for f in features if df[f].dtype in ['int32', 'int64', 'float32', 'float64']]
        binary_features = [f for f in features if df[f].dtype in ['bool'] or (df[f].max() <= 1 and df[f].min() >= 0)]
        
        f.write(f"Numeric features: {len(numeric_features)}\n")
        f.write(f"Binary features: {len(binary_features)}\n")
        
        # Check for zero variance features
        zero_var_features = []
        for feature in numeric_features:
            if df[feature].var() == 0:
                zero_var_features.append(feature)
        
        if zero_var_features:
            f.write(f"Zero variance features: {len(zero_var_features)}\n")
            for feature in zero_var_features:
                f.write(f"  {feature}\n")
        else:
            f.write("No zero variance features ✓\n")
        f.write("\n")
        
        # Outlier analysis
        f.write("OUTLIER ANALYSIS:\n")
        f.write("-"*30 + "\n")
        outlier_features = []
        for feature in numeric_features[:20]:  # Limit to first 20 for performance
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
            if outliers > 0:
                outlier_percentage = outliers / len(df) * 100
                outlier_features.append((feature, outliers, outlier_percentage))
        
        if outlier_features:
            f.write("Features with outliers (>1.5*IQR):\n")
            for feature, count, percentage in outlier_features:
                f.write(f"  {feature}: {count:,} ({percentage:.1f}%)\n")
        else:
            f.write("No significant outliers detected ✓\n")
        f.write("\n")
        
        # Feature correlation analysis
        f.write("FEATURE CORRELATION ANALYSIS:\n")
        f.write("-"*30 + "\n")
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            if high_corr_pairs:
                f.write(f"High correlation pairs (>0.9): {len(high_corr_pairs)}\n")
                for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
                    f.write(f"  {feat1} <-> {feat2}: {corr:.3f}\n")
            else:
                f.write("No high correlation pairs detected ✓\n")
        f.write("\n")
        
        # Summary and recommendations
        f.write("SUMMARY AND RECOMMENDATIONS:\n")
        f.write("-"*30 + "\n")
        
        total_issues = len(missing_features) + len(zero_var_features) + len(outlier_features)
        
        if total_issues == 0:
            f.write("✓ Data quality is excellent!\n")
            f.write("✓ No missing values, zero variance, or major outlier issues detected.\n")
            f.write("✓ Dataset is ready for modeling.\n")
        else:
            f.write(f"Found {total_issues} potential data quality issues:\n")
            if missing_features:
                f.write(f"- {len(missing_features)} features with missing values\n")
            if zero_var_features:
                f.write(f"- {len(zero_var_features)} features with zero variance\n")
            if outlier_features:
                f.write(f"- {len(outlier_features)} features with outliers\n")
            
            f.write("\nRecommendations:\n")
            if missing_features:
                f.write("- Consider imputation strategies for missing values\n")
            if zero_var_features:
                f.write("- Remove zero variance features before modeling\n")
            if outlier_features:
                f.write("- Consider outlier treatment (capping, transformation, removal)\n")

# Additional utility functions for the pipeline
def get_feature_engineering_config() -> Dict[str, any]:
    """Get default feature engineering configuration."""
    return {
        'temporal': {
            'lag_periods': [1, 2, 3, 6, 12],
            'rolling_windows': [3, 6, 12],
            'momentum_periods': [2, 3, 6]
        },
        'customer_behavior': {
            'consistency_threshold': 0.5,
            'diversity_calculation': True,
            'loyalty_analysis': True
        },
        'store_categorization': {
            'chinese_patterns': True,
            'quality_scoring': True,
            'competitive_analysis': True
        },
        'platform_dynamics': {
            'cross_platform_analysis': True,
            'loyalty_patterns': True,
            'seasonal_effects': True
        },
        'validation': {
            'max_missing_ratio': 0.1,
            'outlier_threshold': 3.0,
            'correlation_threshold': 0.95
        }
    }