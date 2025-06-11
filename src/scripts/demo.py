#!/usr/bin/env python3
"""
Demonstration script for the refactored sales forecasting project.

This script shows how to use the refactored modules to:
1. Load raw data
2. Preprocess the data
3. Engineer features
4. Prepare for modeling

Run with: python scripts/demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import our modules
from data.loader import SalesDataLoader
from data.preprocessing import SalesPreprocessor
from data.feature_engineering import FeatureEngineer
from config.config import get_config
from utils.helpers import setup_logging, print_data_summary, validate_data_quality


def main():
    """Main demonstration function."""
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting sales forecasting demonstration")
    
    # Load configuration
    config = get_config()
    logger.info(f"Loaded configuration for project: {config.project_name} v{config.version}")
    
    try:
        # Step 1: Load raw data
        logger.info("="*60)
        logger.info("STEP 1: LOADING RAW DATA")
        logger.info("="*60)
        
        loader = SalesDataLoader(data_dir=config.data.raw_data_dir)
        
        # Try to load Excel data first
        try:
            raw_data = loader.load_excel_data(years=[2021, 2022])
            logger.info("✓ Successfully loaded Excel data")
        except Exception as e:
            logger.warning(f"Could not load Excel data: {e}")
            logger.info("Trying to load from CSV files...")
            
            # Try CSV data as fallback
            try:
                raw_data = loader.load_raw_data(source="csv")
                logger.info("✓ Successfully loaded CSV data")
            except Exception as e:
                logger.error(f"Could not load any data: {e}")
                logger.info("Creating demo data for illustration...")
                
                # Create minimal demo data
                import pandas as pd
                import numpy as np
                from datetime import datetime, timedelta
                
                dates = pd.date_range(start='2021-01-01', end='2022-12-01', freq='MS')
                platforms = ['Douyin', 'JD', 'Tmall']
                stores = ['Store_A', 'Store_B', 'Store_C']
                brands = ['Brand_X', 'Brand_Y', 'Brand_Z']
                
                data = []
                for date in dates:
                    for platform in platforms:
                        for store in stores:
                            for brand in brands:
                                data.append({
                                    'sales_month': date,
                                    'primary_platform': platform,
                                    'secondary_platform': '',
                                    'store_name': store,
                                    'brand_name': brand,
                                    'product_code': f'P_{platform}_{store}_{brand}',
                                    'sales_quantity': np.random.randint(10, 1000),
                                    'sales_amount': np.random.randint(1000, 50000)
                                })
                
                raw_data = pd.DataFrame(data)
                logger.info("✓ Created demo data with shape: " + str(raw_data.shape))
        
        print_data_summary(raw_data, "Raw Data Summary")
        
        # Step 2: Preprocess data
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("="*60)
        
        preprocessor = SalesPreprocessor()
        clean_data, quality_report = preprocessor.process(raw_data)
        
        print_data_summary(clean_data, "Preprocessed Data Summary")
        
        # Validate data quality
        validation_results = validate_data_quality(
            clean_data, 
            required_columns=['sales_month', 'primary_platform', 'store_name', 'brand_name', 'sales_quantity']
        )
        logger.info(f"Data validation results: {validation_results}")
        
        # Step 3: Feature engineering
        logger.info("\n" + "="*60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*60)
        
        engineer = FeatureEngineer()
        engineered_data, modeling_features, rolling_splits = engineer.create_all_features(clean_data)
        
        print_data_summary(engineered_data, "Engineered Data Summary")
        
        logger.info(f"\nFeature Engineering Results:")
        logger.info(f"- Original features: {len(clean_data.columns)}")
        logger.info(f"- Engineered features: {len(engineered_data.columns)}")
        logger.info(f"- Modeling features: {len(modeling_features)}")
        logger.info(f"- Rolling splits: {len(rolling_splits)}")
        
        # Display sample features by category
        temporal_features = [f for f in modeling_features if any(x in f for x in ['month', 'quarter', 'sin', 'cos', 'days'])]
        seasonal_features = [f for f in modeling_features if 'seasonal' in f or 'promotional' in f]
        
        logger.info(f"\nSample Feature Categories:")
        logger.info(f"- Temporal features ({len(temporal_features)}): {temporal_features[:5]}...")
        logger.info(f"- Seasonal features ({len(seasonal_features)}): {seasonal_features[:3]}...")
        
        # Display rolling splits info
        logger.info(f"\nRolling Splits Information:")
        for i, (train, val, description) in enumerate(rolling_splits):
            logger.info(f"Split {i+1}: {description}")
            logger.info(f"  Train: {len(train)} records, Val: {len(val)} records")
        
        # Step 4: Ready for modeling
        logger.info("\n" + "="*60)
        logger.info("READY FOR MODELING!")
        logger.info("="*60)
        
        logger.info("Next steps would be:")
        logger.info("1. Create model classes in src/models/")
        logger.info("2. Implement training pipelines")
        logger.info("3. Add hyperparameter optimization")
        logger.info("4. Create prediction and evaluation scripts")
        
        logger.info(f"\nData is ready with {len(modeling_features)} features for training deep learning models!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("Demonstration completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 