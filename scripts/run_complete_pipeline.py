#!/usr/bin/env python3
"""
Complete Sales Forecasting Pipeline

This script demonstrates the complete end-to-end pipeline for sales forecasting
using the refactored modules. It replaces the full_data_prediction.ipynb notebook
with a production-ready, modular approach.

Usage:
    python scripts/run_complete_pipeline.py --data-dir data/raw --output-dir outputs

Author: Sales Forecasting Team
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import all pipeline components
from data.feature_pipeline import SalesFeaturePipeline
from models.trainer import ModelTrainer
from utils.helpers import setup_logging, print_data_summary

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete Sales Forecasting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Directory containing raw Excel data files (2021.xlsx, 2022.xlsx, etc.)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Base directory for saving all outputs"
    )
    
    parser.add_argument(
        "--years", 
        nargs="+", 
        type=int, 
        default=[2021, 2022],
        help="Years to process (e.g., --years 2021 2022 2023)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=512,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--experiment-name", 
        type=str,
        help="Name for the experiment (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-feature-engineering", 
        action="store_true",
        help="Skip feature engineering and use existing engineered dataset"
    )
    
    parser.add_argument(
        "--engineered-dataset", 
        type=str,
        help="Path to existing engineered dataset (if --skip-feature-engineering is used)"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check data directory exists
    if not Path(args.data_dir).exists() and not args.skip_feature_engineering:
        errors.append(f"Data directory does not exist: {args.data_dir}")
    
    # Check for required Excel files
    if not args.skip_feature_engineering:
        for year in args.years:
            excel_file = Path(args.data_dir) / f"{year}.xlsx"
            if not excel_file.exists():
                errors.append(f"Required Excel file not found: {excel_file}")
    
    # Check engineered dataset if skipping feature engineering
    if args.skip_feature_engineering:
        if not args.engineered_dataset:
            errors.append("--engineered-dataset is required when --skip-feature-engineering is used")
        elif not Path(args.engineered_dataset).exists():
            errors.append(f"Engineered dataset file not found: {args.engineered_dataset}")
    
    # Validate numeric arguments
    if args.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if args.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if args.random_seed < 0:
        errors.append("Random seed must be non-negative")
    
    return errors

def run_feature_engineering_pipeline(args, logger):
    """Run the complete feature engineering pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 1: FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    # Initialize feature pipeline
    feature_pipeline = SalesFeaturePipeline(
        output_dir=str(Path(args.output_dir) / "engineered")
    )
    
    # Run complete feature engineering
    logger.info(f"Processing years: {args.years}")
    logger.info(f"Raw data directory: {args.data_dir}")
    
    try:
        engineered_data_path, modeling_features, rolling_splits, metadata = feature_pipeline.run_complete_pipeline(
            raw_data_dir=args.data_dir,
            years=args.years
        )
        
        logger.info("✓ Feature engineering completed successfully")
        logger.info(f"✓ Engineered dataset saved: {engineered_data_path}")
        logger.info(f"✓ Total features created: {len(modeling_features)}")
        logger.info(f"✓ Rolling splits created: {len(rolling_splits)}")
        
        return engineered_data_path, modeling_features, rolling_splits, metadata
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def load_existing_engineered_dataset(args, logger):
    """Load existing engineered dataset."""
    logger.info("=" * 80)
    logger.info("PHASE 1: LOADING EXISTING ENGINEERED DATASET")
    logger.info("=" * 80)
    
    from data.feature_pipeline import SalesFeaturePipeline
    
    feature_pipeline = SalesFeaturePipeline()
    
    try:
        df_final, modeling_features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(
            args.engineered_dataset
        )
        
        logger.info("✓ Engineered dataset loaded successfully")
        logger.info(f"✓ Dataset shape: {df_final.shape}")
        logger.info(f"✓ Features available: {len(modeling_features)}")
        logger.info(f"✓ Rolling splits: {len(rolling_splits)}")
        
        return args.engineered_dataset, modeling_features, rolling_splits, metadata
        
    except Exception as e:
        logger.error(f"Failed to load engineered dataset: {str(e)}")
        raise

def run_model_training_pipeline(engineered_data_path, modeling_features, rolling_splits, metadata, args, logger):
    """Run the complete model training pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 2: MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Load engineered dataset
    from data.feature_pipeline import SalesFeaturePipeline
    feature_pipeline = SalesFeaturePipeline()
    df_final, _, _, _ = feature_pipeline.load_engineered_dataset(engineered_data_path)
    
    # Initialize model trainer
    trainer = ModelTrainer(
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    # Generate experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"sales_forecasting_experiment_{timestamp}"
    
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    try:
        # Train complete pipeline
        training_results = trainer.train_complete_pipeline(
            df_final=df_final,
            features=modeling_features,
            rolling_splits=rolling_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            experiment_name=experiment_name
        )
        
        logger.info("✓ Model training completed successfully")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def print_final_summary(training_results, logger):
    """Print final pipeline summary."""
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    final_summary = training_results['final_summary']
    comprehensive_results = training_results['comprehensive_results']
    
    # Overall performance
    logger.info(f"Experiment Status: {'✓ COMPLETED' if final_summary['experiment_completed'] else '✗ FAILED'}")
    logger.info(f"Overall Grade: {final_summary['overall_grade']}")
    logger.info(f"Average Validation MAPE: {final_summary['average_validation_mape']:.2f}%")
    logger.info(f"Consistency Grade: {final_summary['consistency_grade']}")
    logger.info(f"Business Ready: {'✓ YES' if final_summary['business_ready'] else '✗ NO'}")
    
    # Performance range
    logger.info(f"Best Split MAPE: {final_summary['best_split_mape']:.2f}%")
    logger.info(f"Worst Split MAPE: {final_summary['worst_split_mape']:.2f}%")
    logger.info(f"Total Splits Trained: {final_summary['total_splits_trained']}")
    
    # Platform performance
    if 'platform_performance' in comprehensive_results:
        logger.info("\nPlatform Performance:")
        for platform, perf in comprehensive_results['platform_performance'].items():
            logger.info(f"  {platform}: {perf['mean_mape']:.2f}% ± {perf['std_mape']:.2f}%")
    
    # Recommendations
    logger.info("\nRecommendations:")
    for recommendation in final_summary['recommendations']:
        logger.info(f"  {recommendation}")
    
    # Saved files
    saved_files = training_results['saved_files']
    logger.info("\nGenerated Outputs:")
    for key, filepath in saved_files.items():
        if filepath and filepath != 'N/A':
            logger.info(f"  {key}: {filepath}")

def main():
    """Main pipeline execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("SALES FORECASTING COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        logger.error("Argument validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Phase 1: Feature Engineering or Load Existing Dataset
        if args.skip_feature_engineering:
            engineered_data_path, modeling_features, rolling_splits, metadata = load_existing_engineered_dataset(args, logger)
        else:
            engineered_data_path, modeling_features, rolling_splits, metadata = run_feature_engineering_pipeline(args, logger)
        
        # Phase 2: Model Training
        training_results = run_model_training_pipeline(
            engineered_data_path, modeling_features, rolling_splits, metadata, args, logger
        )
        
        # Phase 3: Final Summary
        print_final_summary(training_results, logger)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)