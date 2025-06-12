"""
Configuration management for sales forecasting project.

This module contains project-wide configuration settings including data paths,
model parameters, and feature engineering configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import os


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data directories
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    engineered_data_dir: str = "data/engineered"
    
    # File patterns
    excel_files: List[str] = None
    csv_files: List[str] = None
    
    # Platform mapping
    platform_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.excel_files is None:
            self.excel_files = ["2021.xlsx", "2022.xlsx", "2023.xlsx"]
        
        if self.csv_files is None:
            self.csv_files = ["Douyin_sales_data.csv", "JD_sales_data.csv", "Tmall_sales_data.csv"]
        
        if self.platform_mapping is None:
            self.platform_mapping = {
                '抖音': 'Douyin',
                '京东': 'JD',
                '天猫': 'Tmall'
            }


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Temporal features
    promotional_months: List[int] = None
    lag_periods: List[int] = None
    rolling_windows: List[int] = None
    
    # Store categorization
    chinese_store_types: Dict[str, str] = None
    
    # Spike detection
    spike_threshold_percentiles: List[float] = None
    
    def __post_init__(self):
        if self.promotional_months is None:
            self.promotional_months = [1, 6, 9, 11, 12]  # Chinese e-commerce promotional periods
        
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 6, 12]
        
        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12]
        
        if self.chinese_store_types is None:
            self.chinese_store_types = {
                '官方旗舰店': 'Official Flagship',
                '卖场旗舰店': 'Mall Flagship', 
                '旗舰店': 'Flagship',
                '专卖店': 'Specialty Store',
                '专营店': 'Specialty Store',
                '卖场店': 'Mall Store',
                '自营': 'Platform Direct',
                '超市': 'Supermarket'
            }
        
        if self.spike_threshold_percentiles is None:
            self.spike_threshold_percentiles = [0.95, 0.99]


@dataclass
class ModelConfig:
    """Configuration for model training and validation."""
    
    # Model parameters
    random_state: int = 42
    test_size: float = 0.2
    validation_splits: int = 4
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.001
    early_stopping_patience: int = 20
    
    # Model selection
    target_column: str = "sales_quantity"
    metric: str = "mape"  # Mean Absolute Percentage Error
    
    # Feature selection
    exclude_columns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_columns is None:
            self.exclude_columns = [
                'sales_month', 'store_name', 'brand_name', 'product_code', 
                'sales_amount', 'sales_quantity', 'unit_price', 'store_type',
                'month_year', 'primary_platform', 'secondary_platform'
            ]


@dataclass
class OutputConfig:
    """Configuration for output directories and files."""
    
    # Output directories
    models_dir: str = "outputs/models"
    predictions_dir: str = "outputs/predictions"
    reports_dir: str = "outputs/reports"
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ProjectConfig:
    """Main project configuration."""
    
    # Sub-configurations
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    output: OutputConfig = None
    
    # Project info
    project_name: str = "sales_forecasting"
    version: str = "0.1.0"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        
        if self.features is None:
            self.features = FeatureConfig()
        
        if self.model is None:
            self.model = ModelConfig()
        
        if self.output is None:
            self.output = OutputConfig()


def get_config() -> ProjectConfig:
    """
    Get the default project configuration.
    
    Returns:
        ProjectConfig instance with default settings
    """
    return ProjectConfig()


def load_config_from_env() -> ProjectConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        ProjectConfig instance with settings from environment
    """
    config = ProjectConfig()
    
    # Override with environment variables if they exist
    if os.getenv('DATA_DIR'):
        config.data.raw_data_dir = os.getenv('DATA_DIR')
    
    if os.getenv('MODEL_RANDOM_STATE'):
        config.model.random_state = int(os.getenv('MODEL_RANDOM_STATE'))
    
    if os.getenv('LOG_LEVEL'):
        config.output.log_level = os.getenv('LOG_LEVEL')
    
    return config


def save_config(config: ProjectConfig, file_path: Union[str, Path]) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration to save
        file_path: Path to save the configuration
    """
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config_from_file(file_path: Union[str, Path]) -> ProjectConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        ProjectConfig instance loaded from file
    """
    import json
    
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert back to dataclass instances
    data_config = DataConfig(**config_dict['data'])
    features_config = FeatureConfig(**config_dict['features'])
    model_config = ModelConfig(**config_dict['model'])
    output_config = OutputConfig(**config_dict['output'])
    
    return ProjectConfig(
        data=data_config,
        features=features_config,
        model=model_config,
        output=output_config,
        project_name=config_dict.get('project_name', 'sales_forecasting'),
        version=config_dict.get('version', '0.1.0')
    ) 