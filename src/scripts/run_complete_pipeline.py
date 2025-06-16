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
import numpy as np

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
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vanilla", "enhanced", "both"],
        default="enhanced",
        help="Model type to train: vanilla (baseline), enhanced (improved), or both"
    )
    
    parser.add_argument(
        "--run-phase3",
        action="store_true",
        help="Run Phase 3 (test models on 2023 data) after training"
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
    
    # Initialize model trainer with model type selection
    from models.vanilla_embedding_model import VanillaEmbeddingModel
    from models.enhanced_embedding_model import EnhancedEmbeddingModel
    
    training_results = {}
    
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
        # Train models based on selection
        if args.model_type in ["vanilla", "both"]:
            logger.info("Training Vanilla (Baseline) Model...")
            vanilla_model = VanillaEmbeddingModel(random_seed=args.random_seed)
            vanilla_results = vanilla_model.train_on_rolling_splits(
                df_final=df_final,
                features=modeling_features,
                rolling_splits=rolling_splits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                models_dir=f"{args.output_dir}/{experiment_name}/vanilla_models"
            )
            training_results['vanilla'] = vanilla_results
            logger.info("✓ Vanilla model training completed")
        
        if args.model_type in ["enhanced", "both"]:
            logger.info("Training Enhanced Model...")
            enhanced_model = EnhancedEmbeddingModel(random_seed=args.random_seed)
            enhanced_results = enhanced_model.train_on_rolling_splits(
                df_final=df_final,
                features=modeling_features,
                rolling_splits=rolling_splits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                models_dir=f"{args.output_dir}/{experiment_name}/enhanced_models"
            )
            training_results['enhanced'] = enhanced_results
            logger.info("✓ Enhanced model training completed")
        
        logger.info("✓ Model training completed successfully")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def run_phase3_evaluation(engineered_data_path, training_results, experiment_name, args, logger):
    """Run Phase 3: Test models on 2023 data."""
    logger.info("=" * 80)
    logger.info("PHASE 3: MODEL EVALUATION ON 2023 DATA")
    logger.info("=" * 80)
    
    from scripts.phase3_test_model import FixedModel2023Evaluator
    
    phase3_results = {}
    
    try:
        # Test vanilla models if trained
        if 'vanilla' in training_results:
            logger.info("Evaluating Vanilla Models on 2023 data...")
            vanilla_models_dir = f"{args.output_dir}/{experiment_name}/vanilla_models"
            if Path(vanilla_models_dir).exists():
                evaluator = FixedModel2023Evaluator(vanilla_models_dir)
                vanilla_2023_results = evaluator.run_complete_evaluation(engineered_data_path)
                phase3_results['vanilla'] = vanilla_2023_results
                logger.info("✓ Vanilla model 2023 evaluation completed")
        
        # Test enhanced models if trained
        if 'enhanced' in training_results:
            logger.info("Evaluating Enhanced Models on 2023 data...")
            enhanced_models_dir = f"{args.output_dir}/{experiment_name}/enhanced_models"
            if Path(enhanced_models_dir).exists():
                evaluator = FixedModel2023Evaluator(enhanced_models_dir)
                enhanced_2023_results = evaluator.run_complete_evaluation(engineered_data_path)
                phase3_results['enhanced'] = enhanced_2023_results
                logger.info("✓ Enhanced model 2023 evaluation completed")
        
        logger.info("✓ Phase 3 evaluation completed successfully")
        return phase3_results
        
    except Exception as e:
        logger.error(f"Phase 3 evaluation failed: {str(e)}")
        return {}

def print_final_summary(training_results, phase3_results, logger):
    """Print final pipeline summary."""
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    # Training Results Summary
    logger.info("PHASE 2 TRAINING RESULTS:")
    for model_type, results in training_results.items():
        if results:
            mapes = [result['val_mape'] for result in results.values()]
            avg_mape = np.mean(mapes)
            logger.info(f"  {model_type.title()} Model - Average MAPE: {avg_mape:.2f}%")
    
    # Phase 3 Results Summary
    if phase3_results:
        logger.info("\nPHASE 3 EVALUATION RESULTS (2023 Data):")
        for model_type, results in phase3_results.items():
            if results and 'overall_performance' in results:
                overall_perf = results['overall_performance']
                logger.info(f"  {model_type.title()} Model - 2023 MAPE: {overall_perf['mean_mape']:.2f}%")
    
    # Model Comparison
    if len(training_results) > 1:
        logger.info("\nMODEL COMPARISON:")
        vanilla_mape = None
        enhanced_mape = None
        
        if 'vanilla' in training_results:
            vanilla_mapes = [result['val_mape'] for result in training_results['vanilla'].values()]
            vanilla_mape = np.mean(vanilla_mapes)
            logger.info(f"  Vanilla (Baseline): {vanilla_mape:.2f}%")
        
        if 'enhanced' in training_results:
            enhanced_mapes = [result['val_mape'] for result in training_results['enhanced'].values()]
            enhanced_mape = np.mean(enhanced_mapes)
            logger.info(f"  Enhanced: {enhanced_mape:.2f}%")
        
        if vanilla_mape and enhanced_mape:
            improvement = vanilla_mape - enhanced_mape
            logger.info(f"  Improvement: {improvement:.2f} percentage points")
            if improvement > 0:
                logger.info("  ✅ Enhanced model outperforms vanilla baseline")
            else:
                logger.info("  ⚠️ Enhanced model underperforms vanilla baseline")

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
        
        # Phase 3: Model Evaluation (Optional)
        phase3_results = {}
        if args.run_phase3:
            # Generate experiment name if not provided (same logic as Phase 2)
            experiment_name = args.experiment_name
            if experiment_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"sales_forecasting_experiment_{timestamp}"
            
            phase3_results = run_phase3_evaluation(
                engineered_data_path, training_results, experiment_name, args, logger
            )
        
        # Final Summary
        print_final_summary(training_results, phase3_results, logger)
        
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