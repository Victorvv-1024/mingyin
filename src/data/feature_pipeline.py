"""
Master feature engineering pipeline for sales forecasting.

This module orchestrates the complete feature engineering process,
combining all specialized feature modules into a cohesive pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from .utils import setup_logging, get_modeling_features
from .features.temporal import TemporalFeatureEngineer
from .features.customer_behavior import CustomerBehaviorFeatureEngineer
from .features.store_categorization import StoreCategorization
from .preprocessing import SalesDataProcessor

logger = setup_logging()

class SalesFeaturePipeline:
    """Master pipeline for comprehensive sales forecasting feature engineering."""
    
    def __init__(self, output_dir: str = "data/engineered"):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            output_dir: Directory to save engineered datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature engineers
        self.temporal_engineer = TemporalFeatureEngineer()
        self.customer_behavior_engineer = CustomerBehaviorFeatureEngineer()
        self.store_categorization = StoreCategorization()
        
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
        
        # Step 2: Comprehensive Feature Engineering
        logger.info("Step 2: Comprehensive Feature Engineering")
        df_engineered = self.engineer_all_features(df)
        
        # Step 3: Prepare for modeling
        logger.info("Step 3: Preparing Modeling Dataset")
        modeling_features = get_modeling_features(df_engineered)
        
        # Step 4: Create rolling splits
        logger.info("Step 4: Creating Rolling Time Series Splits")
        rolling_splits = self.create_rolling_splits(df_engineered)
        
        # Step 5: Save engineered dataset
        logger.info("Step 5: Saving Engineered Dataset")
        engineered_data_path = self.save_engineered_dataset(
            df_engineered, modeling_features, rolling_splits, quality_metrics
        )
        
        logger.info("=" * 80)
        logger.info("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final dataset: {df_engineered.shape}")
        logger.info(f"Modeling features: {len(modeling_features)}")
        logger.info(f"Rolling splits: {len(rolling_splits)}")
        logger.info(f"Saved to: {engineered_data_path}")
        
        return engineered_data_path, modeling_features, rolling_splits, quality_metrics
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive feature engineering from data_engineer.ipynb."""
        logger.info("Applying comprehensive feature engineering...")
        
        original_cols = len(df.columns)
        
        # Ensure datetime conversion and basic setup
        df['sales_month'] = pd.to_datetime(df['sales_month'])
        
        # Add unit price calculation (needed for many features)
        df['unit_price'] = df['sales_amount'] / df['sales_quantity']
        df['unit_price'] = df['unit_price'].replace([np.inf, -np.inf], np.nan)
        df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        
        logger.info("✓ Basic setup completed")
        
        # Apply all feature engineering modules in sequence
        df = self.temporal_engineer.engineer_all_temporal_features(df)
        df = self.customer_behavior_engineer.engineer_all_customer_behavior_features(df)
        df = self.store_categorization.engineer_all_store_features(df)
        
        new_cols = len(df.columns)
        logger.info(f"Feature engineering completed: {original_cols} → {new_cols} (+{new_cols - original_cols} features)")
        
        # Feature summary
        self._print_feature_summary(df, original_cols)
        
        return df
    
    def _print_feature_summary(self, df: pd.DataFrame, original_cols: int) -> None:
        """Print a summary of all engineered features."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)
        
        # Count features by type (matching data_engineer.ipynb categories)
        feature_categories = {
            'Temporal Basic': [col for col in df.columns if any(x in col for x in ['month', 'quarter', 'year', 'days_since'])],
            'Cyclical': [col for col in df.columns if any(x in col for x in ['sin', 'cos'])],
            'Promotional': [col for col in df.columns if any(x in col for x in ['promotional', 'distance_to_promo'])],
            'Seasonal': [col for col in df.columns if 'seasonal' in col and 'interaction' not in col],
            'Year-over-Year': [col for col in df.columns if 'yoy_' in col],
            'Lag Features': [col for col in df.columns if 'lag_' in col],
            'Rolling Windows': [col for col in df.columns if 'rolling_' in col],
            'Momentum': [col for col in df.columns if any(x in col for x in ['momentum', 'acceleration', 'pct_change', 'volatility'])],
            'Store Behavior': [col for col in df.columns if any(x in col for x in ['store_sales_cv', 'store_sales_range', 'brand_diversity', 'product_diversity'])],
            'Store Type': [col for col in df.columns if 'store_type_' in col],
            'Brand Market': [col for col in df.columns if any(x in col for x in ['brand_market_share', 'brand_promotional_effectiveness'])],
            'Platform Dynamics': [col for col in df.columns if any(x in col for x in ['competing', 'multi_platform', 'platform_preference'])],
            'Seasonal Interactions': [col for col in df.columns if 'interaction' in col],
            'Spike Detection': [col for col in df.columns if any(x in col for x in ['spike', 'deviation', 'z_score', 'propensity'])]
        }
        
        total_engineered = 0
        for category, features in feature_categories.items():
            if features:
                logger.info(f"{category:20s}: {len(features):3d} features")
                total_engineered += len(features)
        
        logger.info("-" * 60)
        logger.info(f"{'TOTAL ENGINEERED':20s}: {total_engineered:3d} features")
        logger.info(f"{'ORIGINAL COLUMNS':20s}: {original_cols:3d}")
        logger.info(f"{'FINAL DATASET':20s}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info("=" * 60)
    
    def create_rolling_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """Create rolling time series splits for model training and validation."""
        logger.info("Creating rolling time series splits...")
        
        df_sorted = df.sort_values('sales_month').copy()
        rolling_splits = []
        
        # Split 1: 2021 full → 2022 Q1 (3 months validation)
        train_1 = df_sorted[df_sorted['sales_month'].dt.year == 2021].copy()
        val_1 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([1, 2, 3]))
        ].copy()
        
        if len(train_1) > 0 and len(val_1) > 0:
            rolling_splits.append((train_1, val_1, "2021_full → 2022_Q1"))
            logger.info(f"Split 1: Train {len(train_1):,} samples, Val {len(val_1):,} samples")
        
        # Split 2: 2021 full + 2022 Q1 → 2022 Q2 (3 months validation)
        train_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3])))
        ].copy()
        val_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([4, 5, 6]))
        ].copy()
        
        if len(train_2) > 0 and len(val_2) > 0:
            rolling_splits.append((train_2, val_2, "2021_full+2022_Q1 → 2022_Q2"))
            logger.info(f"Split 2: Train {len(train_2):,} samples, Val {len(val_2):,} samples")
        
        # Split 3: 2021 full + 2022 H1 → 2022 Q3 (3 months validation)
        train_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3, 4, 5, 6])))
        ].copy()
        val_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([7, 8, 9]))
        ].copy()
        
        if len(train_3) > 0 and len(val_3) > 0:
            rolling_splits.append((train_3, val_3, "2021_full+2022_H1 → 2022_Q3"))
            logger.info(f"Split 3: Train {len(train_3):,} samples, Val {len(val_3):,} samples")
        
        # Split 4: 2021 full + 2022 Q1-Q3 → 2022 Q4 (3 months validation)
        train_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9])))
        ].copy()
        val_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([10, 11, 12]))
        ].copy()
        
        if len(train_4) > 0 and len(val_4) > 0:
            rolling_splits.append((train_4, val_4, "2021_full+2022_Q1Q2Q3 → 2022_Q4"))
            logger.info(f"Split 4: Train {len(train_4):,} samples, Val {len(val_4):,} samples")
        
        # Optional Split 5: Include some 2023 data if available
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
        
        logger.info(f"Created {len(rolling_splits)} rolling time series splits")
        
        # Validate splits for temporal integrity
        for i, (train_data, val_data, description) in enumerate(rolling_splits):
            train_max_date = train_data['sales_month'].max()
            val_min_date = val_data['sales_month'].min()
            
            if train_max_date >= val_min_date:
                logger.warning(f"Split {i+1} has temporal overlap! Train max: {train_max_date}, Val min: {val_min_date}")
            else:
                logger.info(f"Split {i+1} temporal integrity validated ✓")
        
        return rolling_splits
    
    def save_engineered_dataset(self, 
                               df: pd.DataFrame, 
                               modeling_features: List[str],
                               rolling_splits: List[Tuple],
                               quality_metrics: Dict) -> str:
        """Save the complete engineered dataset with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Saving engineered dataset...")
        
        # Create comprehensive save package
        dataset_package = {
            'df_final': df,
            'modeling_features': modeling_features,
            'rolling_splits': rolling_splits,
            'metadata': {
                'total_records': len(df),
                'total_features': len(modeling_features),
                'date_range': {
                    'start': df['sales_month'].min(),
                    'end': df['sales_month'].max()
                },
                'platforms': df['primary_platform'].unique().tolist(),
                'quality_metrics': quality_metrics,
                'feature_categories': self._categorize_features(modeling_features),
                'creation_timestamp': timestamp,
                'pipeline_version': "comprehensive_v2_from_data_engineer_notebook"
            }
        }
        
        # Save as pickle
        output_path = self.output_dir / f"sales_forecast_engineered_dataset_{timestamp}.pkl"
        pd.to_pickle(dataset_package, output_path)
        
        # Also save CSV for external tools
        csv_path = self.output_dir / f"sales_forecast_engineered_dataset_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save feature documentation
        features_path = self.output_dir / f"modeling_features_{timestamp}.txt"
        self._save_feature_documentation(modeling_features, features_path, timestamp)
        
        logger.info(f"✓ Complete dataset saved as: {output_path}")
        logger.info(f"✓ CSV version saved as: {csv_path}")
        logger.info(f"✓ Feature documentation saved as: {features_path}")
        
        return str(output_path)
    
    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features for metadata."""
        return {
            'temporal_basic': [col for col in features if any(x in col for x in ['month', 'quarter', 'year', 'days_since'])],
            'temporal_cyclical': [col for col in features if any(x in col for x in ['sin', 'cos'])],
            'promotional': [col for col in features if any(x in col for x in ['promotional', 'distance_to_promo'])],
            'seasonal': [col for col in features if 'seasonal' in col and 'interaction' not in col],
            'lag_features': [col for col in features if 'lag_' in col],
            'rolling_features': [col for col in features if 'rolling_' in col],
            'momentum': [col for col in features if any(x in col for x in ['momentum', 'acceleration', 'pct_change', 'volatility'])],
            'store_behavior': [col for col in features if any(x in col for x in ['store_sales_cv', 'store_sales_range', 'brand_diversity', 'product_diversity'])],
            'store_type': [col for col in features if 'store_type_' in col],
            'brand_market': [col for col in features if any(x in col for x in ['brand_market_share', 'brand_promotional_effectiveness'])],
            'platform_dynamics': [col for col in features if any(x in col for x in ['competing', 'multi_platform', 'platform_preference'])],
            'interactions': [col for col in features if 'interaction' in col],
            'spike_detection': [col for col in features if any(x in col for x in ['spike', 'deviation', 'z_score', 'propensity'])]
        }
    
    def _save_feature_documentation(self, features: List[str], file_path: Path, timestamp: str) -> None:
        """Save feature documentation."""
        feature_categories = self._categorize_features(features)
        
        with open(file_path, 'w') as f:
            f.write("COMPREHENSIVE SALES FORECASTING FEATURES\n")
            f.write("Based on data_engineer.ipynb notebook analysis\n")
            f.write("=" * 60 + "\n\n")
            
            for category, feature_list in feature_categories.items():
                if feature_list:
                    f.write(f"{category.upper().replace('_', ' ')} ({len(feature_list)} features):\n")
                    for feature in feature_list:
                        f.write(f"  - {feature}\n")
                    f.write("\n")
            
            f.write(f"TOTAL FEATURES: {len(features)}\n")
            f.write(f"DATASET CREATED: {timestamp}\n")
            f.write(f"PIPELINE VERSION: comprehensive_v2_from_data_engineer_notebook\n")

def load_engineered_dataset(file_path: str) -> Tuple[pd.DataFrame, List[str], List[Tuple], Dict]:
    """
    Load the saved engineered dataset.
    
    Args:
        file_path: Path to the saved pickle file
        
    Returns:
        Tuple of (df_final, modeling_features, rolling_splits, metadata)
    """
    logger.info(f"Loading engineered dataset from: {file_path}")
    
    dataset_package = pd.read_pickle(file_path)
    
    logger.info("✓ Dataset loaded successfully")
    logger.info(f"Records: {len(dataset_package['df_final']):,}")
    logger.info(f"Features: {len(dataset_package['modeling_features'])}")
    logger.info(f"Rolling Splits: {len(dataset_package['rolling_splits'])}")
    logger.info(f"Created: {dataset_package['metadata']['creation_timestamp']}")
    
    return (
        dataset_package['df_final'],
        dataset_package['modeling_features'],
        dataset_package['rolling_splits'],
        dataset_package['metadata']
    )

def main():
    """Main function to run the feature pipeline."""
    pipeline = SalesFeaturePipeline()
    
    # Run the complete pipeline
    engineered_path, features, splits, metadata = pipeline.run_complete_pipeline(
        raw_data_dir="data/raw",
        years=[2021, 2022, 2023]
    )
    
    print(f"Pipeline completed successfully!")
    print(f"Engineered dataset saved to: {engineered_path}")
    print(f"Total features: {len(features)}")
    print(f"Rolling splits: {len(splits)}")

if __name__ == "__main__":
    main() 