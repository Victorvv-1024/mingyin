#!/usr/bin/env python3
"""
Phase 1: Feature Engineering Migration Script

This script replaces the feature engineering portions of full_data_prediction.ipynb
with the new modular pipeline. Use this to validate that the new feature engineering
produces the same results as your notebook before moving to Phase 2.

Usage:
    python scripts/phase1_feature_engineering.py --data-dir data/raw --output-dir data/engineered

This script:
1. Loads raw Excel data (replaces notebook data loading cells)
2. Runs complete feature engineering pipeline (replaces all feature engineering cells)
3. Validates results against expected feature counts
4. Saves engineered dataset for Phase 2
5. Generates comparison reports for validation

Author: Sales Forecasting Team
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import feature engineering components
from src.data.feature_pipeline import SalesFeaturePipeline
from src.utils.helpers import setup_logging, print_data_summary, validate_data_quality

def parse_arguments():
    """Parse command line arguments for Phase 1."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Feature Engineering Migration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Directory containing raw Excel data files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/engineered",
        help="Directory for saving engineered features"
    )
    
    parser.add_argument(
        "--years", 
        nargs="+", 
        type=int, 
        default=[2021, 2022],
        help="Years to process"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--validate-against-notebook", 
        action="store_true",
        help="Generate detailed validation reports to compare with notebook results"
    )
    
    parser.add_argument(
        "--expected-features", 
        type=int, 
        default=80,
        help="Expected number of features (from your notebook)"
    )
    
    return parser.parse_args()

def validate_raw_data(data_dir, years, logger):
    """Validate that raw data files exist and are accessible."""
    logger.info("Validating raw data files...")
    
    missing_files = []
    for year in years:
        excel_file = Path(data_dir) / f"{year}.xlsx"
        if not excel_file.exists():
            missing_files.append(str(excel_file))
    
    if missing_files:
        logger.error("Missing required Excel files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        return False
    
    logger.info("✓ All required Excel files found")
    return True

def run_feature_engineering_pipeline(args, logger):
    """Run the complete feature engineering pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 1: FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    logger.info(f"Processing years: {args.years}")
    logger.info(f"Raw data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize feature pipeline
    feature_pipeline = SalesFeaturePipeline(output_dir=args.output_dir)
    
    try:
        # Run complete feature engineering (replaces notebook feature engineering)
        logger.info("Running comprehensive feature engineering...")
        
        engineered_data_path, modeling_features, rolling_splits, metadata = feature_pipeline.run_complete_pipeline(
            raw_data_dir=args.data_dir,
            years=args.years
        )
        
        logger.info("✓ Feature engineering completed successfully")
        
        return engineered_data_path, modeling_features, rolling_splits, metadata
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def validate_feature_engineering_results(engineered_data_path, modeling_features, rolling_splits, metadata, args, logger):
    """Validate feature engineering results against notebook expectations."""
    logger.info("=" * 80)
    logger.info("VALIDATING FEATURE ENGINEERING RESULTS")
    logger.info("=" * 80)
    
    validation_passed = True
    
    # Load engineered dataset for validation
    from src.data.feature_pipeline import SalesFeaturePipeline
    pipeline = SalesFeaturePipeline()
    df_final, _, _, _ = pipeline.load_engineered_dataset(engineered_data_path)
    
    # 1. Validate feature count
    logger.info(f"Feature count validation:")
    logger.info(f"  Features created: {len(modeling_features)}")
    logger.info(f"  Expected features: {args.expected_features}")
    
    if len(modeling_features) < args.expected_features * 0.9:  # Allow 10% tolerance
        logger.warning(f"⚠️ Feature count below expected ({len(modeling_features)} < {args.expected_features})")
        validation_passed = False
    else:
        logger.info("✓ Feature count validation passed")
    
    # 2. Validate dataset shape
    logger.info(f"Dataset shape validation:")
    logger.info(f"  Final dataset shape: {df_final.shape}")
    logger.info(f"  Total records: {metadata['total_records']:,}")
    
    total_records = len(df_final)
    logger.info(f"  Total records: {total_records:,}")

    if total_records < 1000:  # Sanity check
        logger.warning("⚠️ Dataset seems too small")
        validation_passed = False
    else:
        logger.info("✓ Dataset shape validation passed")
    
    # 3. Validate rolling splits
    logger.info(f"Rolling splits validation:")
    logger.info(f"  Rolling splits created: {len(rolling_splits)}")
    
    if len(rolling_splits) < 3:  # Should have at least 3-4 splits
        logger.warning("⚠️ Too few rolling splits created")
        validation_passed = False
    else:
        logger.info("✓ Rolling splits validation passed")
        
        # Log split details
        for i, (train_df, val_df, description) in enumerate(rolling_splits, 1):
            logger.info(f"  Split {i}: {description}")
            logger.info(f"    Train: {len(train_df):,} records, Val: {len(val_df):,} records")
    
    # 4. Validate data quality
    logger.info(f"Data quality validation:")
    
    # Check for missing values in modeling features
    missing_counts = df_final[modeling_features].isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    
    if len(features_with_missing) > len(modeling_features) * 0.1:  # More than 10% features have missing values
        logger.warning(f"⚠️ Many features have missing values: {len(features_with_missing)}")
        validation_passed = False
    else:
        logger.info("✓ Data quality validation passed")
    
    # 5. Validate key feature categories
    logger.info(f"Feature category validation:")
    
    # Check for temporal features
    temporal_features = [f for f in modeling_features if any(x in f for x in ['month', 'quarter', 'year', 'lag_', 'rolling_'])]
    logger.info(f"  Temporal features: {len(temporal_features)}")
    
    # Check for customer behavior features
    behavior_features = [f for f in modeling_features if any(x in f for x in ['store_', 'brand_', 'market_share'])]
    logger.info(f"  Customer behavior features: {len(behavior_features)}")
    
    # Check for promotional features
    promo_features = [f for f in modeling_features if any(x in f for x in ['promotional', 'promo'])]
    logger.info(f"  Promotional features: {len(promo_features)}")
    
    if len(temporal_features) < 20 or len(behavior_features) < 10:
        logger.warning("⚠️ Some feature categories seem incomplete")
        validation_passed = False
    else:
        logger.info("✓ Feature categories validation passed")
    
    return validation_passed

def generate_comparison_report(engineered_data_path, modeling_features, metadata, args, logger):
    """Generate detailed comparison report for notebook validation."""
    logger.info("Generating comparison report for notebook validation...")
    
    # Create reports directory
    reports_dir = Path(args.output_dir).parent / "reports" / "phase1_validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data for analysis
    from src.data.feature_pipeline import SalesFeaturePipeline
    pipeline = SalesFeaturePipeline()
    df_final, _, _, _ = pipeline.load_engineered_dataset(engineered_data_path)
    
    # Generate feature comparison report
    report_file = reports_dir / f"feature_comparison_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PHASE 1 FEATURE ENGINEERING VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("COMPARISON WITH NOTEBOOK RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Years processed: {args.years}\n")
        f.write(f"Expected features: {args.expected_features}\n")
        f.write(f"Actual features: {len(modeling_features)}\n")
        f.write(f"Final dataset shape: {df_final.shape}\n\n")
        
        # Feature categories breakdown
        f.write("FEATURE CATEGORIES BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        
        feature_categories = {
            'Temporal Basic': [f for f in modeling_features if any(x in f for x in ['month', 'quarter', 'year'])],
            'Cyclical': [f for f in modeling_features if any(x in f for x in ['sin', 'cos'])],
            'Lag Features': [f for f in modeling_features if '_lag_' in f],
            'Rolling Features': [f for f in modeling_features if 'rolling_' in f],
            'Promotional': [f for f in modeling_features if any(x in f for x in ['promotional', 'promo'])],
            'Customer Behavior': [f for f in modeling_features if any(x in f for x in ['store_', 'brand_'])],
            'Platform Dynamics': [f for f in modeling_features if 'platform' in f],
            'Interactions': [f for f in modeling_features if 'interaction' in f]
        }
        
        total_categorized = 0
        for category, features in feature_categories.items():
            f.write(f"{category:20s}: {len(features):3d} features\n")
            total_categorized += len(features)
        
        f.write(f"{'Total Categorized':20s}: {total_categorized:3d} features\n")
        f.write(f"{'Uncategorized':20s}: {len(modeling_features) - total_categorized:3d} features\n\n")
        
        # Data quality summary
        f.write("DATA QUALITY SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {len(df_final):,}\n")
        f.write(f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}\n")
        f.write(f"Platforms: {', '.join(metadata['platforms'])}\n")
        f.write(f"Unique stores: {metadata['stores']:,}\n")
        f.write(f"Unique brands: {metadata['brands']:,}\n")
        f.write(f"Unique products: {metadata['products']:,}\n\n")
        
        # Missing values analysis
        missing_counts = df_final[modeling_features].isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        f.write("MISSING VALUES ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if len(features_with_missing) > 0:
            f.write(f"Features with missing values: {len(features_with_missing)}\n")
            for feature, count in features_with_missing.head(10).items():
                pct = count / len(df_final) * 100
                f.write(f"  {feature}: {count:,} ({pct:.1f}%)\n")
        else:
            f.write("✓ No missing values found\n")
        
        f.write("\nNEXT STEPS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Compare feature counts with your notebook\n")
        f.write("2. Validate key features exist (check the feature list)\n")
        f.write("3. If validation passes, proceed to Phase 2 (Model Training)\n")
        f.write("4. If issues found, review feature engineering configuration\n")
    
    # Generate feature list for easy comparison
    features_file = reports_dir / f"feature_list_{timestamp}.txt"
    with open(features_file, 'w') as f:
        f.write("COMPLETE FEATURE LIST\n")
        f.write("=" * 50 + "\n\n")
        
        for category, features in feature_categories.items():
            if features:
                f.write(f"{category.upper()}\n")
                f.write("-" * len(category) + "\n")
                for feature in sorted(features):
                    f.write(f"  {feature}\n")
                f.write("\n")
    
    logger.info(f"Comparison reports generated:")
    logger.info(f"  Main report: {report_file}")
    logger.info(f"  Feature list: {features_file}")
    
    return str(report_file)

def print_phase1_summary(engineered_data_path, modeling_features, rolling_splits, metadata, validation_passed, logger):
    """Print Phase 1 execution summary."""
    logger.info("=" * 80)
    logger.info("PHASE 1 EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"✓ Feature engineering completed: {engineered_data_path}")
    logger.info(f"✓ Features created: {len(modeling_features)}")
    logger.info(f"✓ Rolling splits: {len(rolling_splits)}")
    logger.info(f"✓ Dataset shape: {metadata['total_records']:,} records")
    logger.info(f"✓ Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    
    if validation_passed:
        logger.info("✅ VALIDATION PASSED - Ready for Phase 2")
        logger.info("\nNext steps:")
        logger.info("1. Review the generated reports to compare with your notebook")
        logger.info("2. Validate that key features match your expectations")
        logger.info("3. Run Phase 2: python scripts/phase2_model_training.py")
    else:
        logger.info("⚠️ VALIDATION ISSUES FOUND - Review before Phase 2")
        logger.info("\nRecommended actions:")
        logger.info("1. Review the validation issues mentioned above")
        logger.info("2. Check feature engineering configuration")
        logger.info("3. Compare with notebook results before proceeding")
    
    logger.info(f"\nGenerated files:")
    logger.info(f"  Engineered dataset: {engineered_data_path}")
    logger.info(f"  Reports directory: {Path(engineered_data_path).parent.parent / 'reports' / 'phase1_validation'}")

def main():
    """Main execution function for Phase 1."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("PHASE 1: FEATURE ENGINEERING MIGRATION")
    logger.info("=" * 80)
    logger.info(f"Replacing notebook feature engineering with modular pipeline")
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate raw data
    if not validate_raw_data(args.data_dir, args.years, logger):
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run feature engineering pipeline
        engineered_data_path, modeling_features, rolling_splits, metadata = run_feature_engineering_pipeline(args, logger)
        
        # Validate results
        validation_passed = validate_feature_engineering_results(
            engineered_data_path, modeling_features, rolling_splits, metadata, args, logger
        )
        
        # Generate comparison reports if requested
        if args.validate_against_notebook:
            report_file = generate_comparison_report(
                engineered_data_path, modeling_features, metadata, args, logger
            )
        
        # Print summary
        print_phase1_summary(
            engineered_data_path, modeling_features, rolling_splits, metadata, validation_passed, logger
        )
        
        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETED")
        logger.info("=" * 80)
        
        return 0 if validation_passed else 2  # Return 2 for validation issues
        
    except KeyboardInterrupt:
        logger.info("Phase 1 execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Phase 1 execution failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)