"""
Utility functions for sales forecasting project.

This module contains helper functions for data processing, visualization,
and general utilities used across the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path


def setup_logging(level: str = "INFO", format_str: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger(__name__)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    
    if not np.any(mask):
        return 100.0  # Return 100% error if all true values are near zero
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metric names and values
    """
    return {
        'mape': calculate_mape(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }


def plot_feature_importance(feature_names: List[str], importance_values: np.ndarray, 
                          top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                          title: str = "Feature Importance") -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importance_values: Importance values for each feature
        top_n: Number of top features to display
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame and sort by importance
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    df_top = df_importance.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df_top, x='importance', y='feature', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    
    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              figsize: Tuple[int, int] = (10, 6),
                              title: str = "Predictions vs Actual") -> plt.Figure:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Predicted vs Actual')
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str,
                    group_col: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6),
                    title: str = "Time Series Plot") -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column to plot
        group_col: Optional grouping column
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if group_col is not None:
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            ax.plot(group_data[date_col], group_data[value_col], 
                   label=group, marker='o', markersize=4)
        ax.legend()
    else:
        ax.plot(df[date_col], df[value_col], marker='o', markersize=4)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def save_results(results: Dict, output_path: Union[str, Path], 
                timestamp: bool = True) -> Path:
    """
    Save results to a file.
    
    Args:
        results: Dictionary of results to save
        output_path: Path to save the results
        timestamp: Whether to add timestamp to filename
        
    Returns:
        Path where results were saved
    """
    import pickle
    from datetime import datetime
    
    output_path = Path(output_path)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{timestamp_str}{suffix}"
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if output_path.suffix == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif output_path.suffix == '.json':
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        # Default to pickle
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(results, f)
        output_path = output_path.with_suffix('.pkl')
    
    return output_path


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print a comprehensive summary of the DataFrame.
    
    Args:
        df: DataFrame to summarize
        title: Title for the summary
    """
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values")
    else:
        print(missing[missing > 0])
    
    print(f"\nNumerical columns summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print(f"\nCategorical columns:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")


def validate_data_quality(df: pd.DataFrame, 
                         required_columns: List[str] = None) -> Dict[str, bool]:
    """
    Validate data quality and return a report.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        results['has_required_columns'] = len(missing_cols) == 0
        if missing_cols:
            results['missing_columns'] = list(missing_cols)
    
    # Check for empty DataFrame
    results['not_empty'] = len(df) > 0
    
    # Check for duplicate rows
    results['no_duplicates'] = df.duplicated().sum() == 0
    if not results['no_duplicates']:
        results['duplicate_count'] = df.duplicated().sum()
    
    # Check for missing values
    results['no_missing_values'] = df.isnull().sum().sum() == 0
    if not results['no_missing_values']:
        results['missing_value_counts'] = df.isnull().sum().to_dict()
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        results['no_infinite_values'] = len(inf_counts) == 0
        if inf_counts:
            results['infinite_value_counts'] = inf_counts
    
    return results 