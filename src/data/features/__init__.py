"""
Feature engineering modules for sales forecasting.

This package contains specialized modules for different types of feature engineering:
- temporal: Time-based features and seasonality
- customer: Customer behavior and store analysis
- platform: Cross-platform competitive dynamics
- store: Store categorization and performance metrics
"""

from .temporal import TemporalFeatureEngineer
from .customer_behavior import CustomerBehaviorFeatureEngineer
from .store_categorization import StoreCategorization

__all__ = [
    'TemporalFeatureEngineer',
    'CustomerBehaviorFeatureEngineer', 
    'StoreCategorization'
] 