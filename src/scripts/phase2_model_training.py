#!/usr/bin/env python3
"""
Phase 2: Model Training Migration Script

This script replaces the model training portions of full_data_prediction.ipynb
with the new advanced embedding model. Use this after Phase 1 has been validated
to ensure the new model architecture produces comparable results to your notebook.

Usage:
    python scripts/phase2_model_training.py --engineered-dataset path_to_dataset.pkl --output-dir outputs

This script:
1. Loads the engineered dataset from Phase 1
2. Runs the advanced embedding model training (replaces notebook model training cells)
3. Validates results against expected performance (MAPE < 20%)
4. Generates comprehensive performance reports
5. Saves trained models and predictions

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

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/scripts/
sys.path.insert(0, str(project_root / "src"))

# Import model training components
from models.trainer import ModelTrainer
from data.feature_pipeline import SalesFeaturePipeline
from utils.helpers import setup_logging

def parse_arguments():
    """Parse command line arguments for Phase 2."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Model Training Migration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--engineered-dataset", 
        type=str, 
        required=True,
        help="Path to engineered dataset from Phase 1 (required)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Directory for saving model outputs"
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
        "--expected-mape", 
        type=float, 
        default=20.0,
        help="Expected maximum MAPE (validation threshold)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--compare-with-notebook", 
        action="store_true",
        help="Generate detailed comparison reports with notebook results"
    )
    
    parser.add_argument(
        "--notebook-mape", 
        type=float,
        help="MAPE achieved in your notebook for comparison"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vanilla", "enhanced", "both"],
        default="enhanced",
        help="Model type to train: vanilla (baseline), enhanced (improved), or both"
    )
    
    return parser.parse_args()

def validate_engineered_dataset(dataset_path, logger):
    """Validate that the engineered dataset exists and is valid."""
    logger.info("Validating engineered dataset...")
    
    if not Path(dataset_path).exists():
        logger.error(f"Engineered dataset not found: {dataset_path}")
        logger.error("Please run Phase 1 first: python scripts/phase1_feature_engineering.py")
        return False
    
    try:
        # Try to load the dataset
        feature_pipeline = SalesFeaturePipeline()
        df_final, modeling_features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(dataset_path)
        
        # Basic validation
        if len(df_final) == 0:
            logger.error("Engineered dataset is empty")
            return False
        
        if len(modeling_features) == 0:
            logger.error("No modeling features found in dataset")
            return False
        
        if len(rolling_splits) == 0:
            logger.error("No rolling splits found in dataset")
            return False
        
        logger.info("✓ Engineered dataset validation passed")
        logger.info(f"  Dataset shape: {df_final.shape}")
        logger.info(f"  Features: {len(modeling_features)}")
        logger.info(f"  Rolling splits: {len(rolling_splits)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load engineered dataset: {str(e)}")
        return False

def run_model_training_pipeline(dataset_path, args, logger):
    """Run the complete model training pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 2: MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Load engineered dataset
    logger.info("Loading engineered dataset...")
    feature_pipeline = SalesFeaturePipeline()
    df_final, modeling_features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(dataset_path)
    
    logger.info(f"✓ Dataset loaded successfully")
    logger.info(f"  Records: {len(df_final):,}")
    logger.info(f"  Features: {len(modeling_features)}")
    logger.info(f"  Rolling splits: {len(rolling_splits)}")
    logger.info(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    
    # Generate experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"phase2_model_training_{timestamp}"
    
    # Import model classes based on selection
    from models.vanilla_embedding_model import VanillaEmbeddingModel
    from models.enhanced_embedding_model import EnhancedEmbeddingModel
    
    # Create experiment-specific output directory
    experiment_output_dir = Path(args.output_dir) / experiment_name
    
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Training parameters:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    training_results = {}
    
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
                models_dir=str(experiment_output_dir / "vanilla_models")
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
                models_dir=str(experiment_output_dir / "enhanced_models")
            )
            training_results['enhanced'] = enhanced_results
            logger.info("✓ Enhanced model training completed")
        
        logger.info("✓ Model training completed successfully")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def validate_model_performance(training_results, args, logger):
    """Validate model performance against expected benchmarks."""
    logger.info("=" * 80)
    logger.info("VALIDATING MODEL PERFORMANCE")
    logger.info("=" * 80)
    
    validation_passed = True
    
    # Validate each model type that was trained
    for model_type, results in training_results.items():
        if results:  # Skip empty results
            logger.info(f"\nValidating {model_type.title()} Model:")
            
            # Calculate average MAPE across splits
            mapes = [result['val_mape'] for result in results.values()]
            avg_mape = np.mean(mapes)
            mape_std = np.std(mapes)
            
            logger.info(f"  MAPE validation:")
            logger.info(f"    Achieved MAPE: {avg_mape:.2f}% ± {mape_std:.2f}%")
            logger.info(f"    Expected MAPE: ≤{args.expected_mape:.1f}%")
            
            if avg_mape > args.expected_mape:
                logger.warning(f"  ⚠️ MAPE above expected threshold ({avg_mape:.2f}% > {args.expected_mape:.1f}%)")
                validation_passed = False
            else:
                logger.info("  ✓ MAPE validation passed")
            
            # Validate consistency
            if mape_std > 5.0:
                logger.warning(f"  ⚠️ High variance across splits ({mape_std:.2f}%)")
                validation_passed = False
            else:
                logger.info("  ✓ Consistency validation passed")
            
            # Business readiness check
            business_ready = avg_mape <= 20.0 and mape_std <= 5.0
            logger.info(f"  Business ready: {'YES' if business_ready else 'NO'}")
            
            # Compare with notebook if provided
            if args.notebook_mape and model_type == 'enhanced':  # Only compare enhanced with notebook
                logger.info(f"  Notebook comparison:")
                logger.info(f"    Notebook MAPE: {args.notebook_mape:.2f}%")
                logger.info(f"    Enhanced model MAPE: {avg_mape:.2f}%")
                
                performance_diff = avg_mape - args.notebook_mape
                if performance_diff > 5.0:
                    logger.warning(f"  ⚠️ Performance worse than notebook by {performance_diff:.2f}%")
                    validation_passed = False
                elif performance_diff < -2.0:
                    logger.info(f"  ✅ Performance better than notebook by {abs(performance_diff):.2f}%")
                else:
                    logger.info("  ✓ Performance comparable to notebook")
    
    return validation_passed

def generate_comparison_report(training_results, args, logger):
    """Generate simplified comparison report."""
    logger.info("Generating comparison report...")
    
    # Create reports directory
    reports_dir = Path(args.output_dir) / "reports" / "phase2_validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"model_comparison_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PHASE 2 MODEL TRAINING VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Expected MAPE: ≤{args.expected_mape:.1f}%\n\n")
        
        for model_type, results in training_results.items():
            if results:
                f.write(f"{model_type.upper()} MODEL RESULTS\n")
                f.write("-" * 40 + "\n")
                
                mapes = [result['val_mape'] for result in results.values()]
                avg_mape = np.mean(mapes)
                f.write(f"Average MAPE: {avg_mape:.2f}%\n")
                
                for split_num, result in results.items():
                    f.write(f"Split {split_num}: MAPE {result['val_mape']:.2f}%, R² {result['val_r2']:.3f}\n")
                f.write("\n")
    
    logger.info(f"Comparison report generated: {report_file}")
    return str(report_file)

def print_phase2_summary(training_results, validation_passed, args, logger):
    """Print Phase 2 execution summary."""
    logger.info("=" * 80)
    logger.info("PHASE 2 EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"✓ Model training completed")
    logger.info(f"✓ Model type: {args.model_type}")
    
    for model_type, results in training_results.items():
        if results:
            mapes = [result['val_mape'] for result in results.values()]
            avg_mape = np.mean(mapes)
            logger.info(f"✓ {model_type.title()} model average MAPE: {avg_mape:.2f}%")
            business_ready = avg_mape <= 20.0
            logger.info(f"✓ {model_type.title()} business ready: {'YES' if business_ready else 'NO'}")
    
    if validation_passed:
        logger.info("✅ VALIDATION PASSED - Model ready for production")
        logger.info("\nNext steps:")
        logger.info("1. Review the trained models")
        logger.info("2. Use Phase 3 to test on 2023 data")
        logger.info("3. Deploy for regular forecasting")
    else:
        logger.info("⚠️ VALIDATION ISSUES FOUND - Review before production")
        logger.info("\nRecommended actions:")
        logger.info("1. Review the performance issues mentioned above")
        logger.info("2. Consider adjusting training parameters")
        logger.info("3. Try different model architectures")
    
    experiment_output_dir = Path(args.output_dir) / (args.experiment_name if args.experiment_name else "phase2_model_training_*")
    logger.info(f"\nGenerated outputs in: {experiment_output_dir}")
    
    # Performance comparison with notebook
    if args.notebook_mape and 'enhanced' in training_results:
        enhanced_results = training_results['enhanced']
        if enhanced_results:
            mapes = [result['val_mape'] for result in enhanced_results.values()]
            avg_mape = np.mean(mapes)
            performance_diff = avg_mape - args.notebook_mape
            if abs(performance_diff) <= 2.0:
                logger.info(f"\n✅ Performance matches notebook expectations")
                logger.info(f"   Difference: {performance_diff:+.2f}% (within acceptable range)")
            else:
                status = "⚠️" if performance_diff > 0 else "✅"
                direction = "worse" if performance_diff > 0 else "better"
                logger.info(f"\n{status} Performance {direction} than notebook")
                logger.info(f"   Difference: {performance_diff:+.2f}%")

def main():
    """Main execution function for Phase 2."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("PHASE 2: MODEL TRAINING MIGRATION")
    logger.info("=" * 80)
    logger.info(f"Replacing notebook model training with advanced embedding model")
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate engineered dataset
    if not validate_engineered_dataset(args.engineered_dataset, logger):
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run model training pipeline
        training_results = run_model_training_pipeline(args.engineered_dataset, args, logger)
        
        # Validate model performance
        validation_passed = validate_model_performance(training_results, args, logger)
        
        # Generate comparison reports if requested
        if args.compare_with_notebook:
            report_file = generate_comparison_report(training_results, args, logger)
        
        # Print summary
        print_phase2_summary(training_results, validation_passed, args, logger)
        
        logger.info("=" * 80)
        logger.info("PHASE 2 COMPLETED")
        logger.info("=" * 80)
        
        return 0 if validation_passed else 2  # Return 2 for validation issues
        
    except KeyboardInterrupt:
        logger.info("Phase 2 execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Phase 2 execution failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)