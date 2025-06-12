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

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import model training components
from src.models.trainer import ModelTrainer
from src.data.feature_pipeline import SalesFeaturePipeline
from src.utils.helpers import setup_logging

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
    
    # Initialize model trainer
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    # Generate experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"phase2_model_training_{timestamp}"
    
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    try:
        # Train complete pipeline (replaces notebook model training)
        logger.info("Starting model training (this replaces your notebook model training cells)...")
        
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

def validate_model_performance(training_results, args, logger):
    """Validate model performance against expected benchmarks."""
    logger.info("=" * 80)
    logger.info("VALIDATING MODEL PERFORMANCE")
    logger.info("=" * 80)
    
    final_summary = training_results['final_summary']
    comprehensive_results = training_results['comprehensive_results']
    
    validation_passed = True
    
    # 1. Validate average MAPE
    avg_mape = final_summary['average_validation_mape']
    logger.info(f"MAPE validation:")
    logger.info(f"  Achieved MAPE: {avg_mape:.2f}%")
    logger.info(f"  Expected MAPE: ≤{args.expected_mape:.1f}%")
    
    if avg_mape > args.expected_mape:
        logger.warning(f"⚠️ MAPE above expected threshold ({avg_mape:.2f}% > {args.expected_mape:.1f}%)")
        validation_passed = False
    else:
        logger.info("✓ MAPE validation passed")
    
    # 2. Validate consistency
    performance_metrics = comprehensive_results['performance_metrics']
    mape_std = performance_metrics['validation_mape']['std']
    logger.info(f"Consistency validation:")
    logger.info(f"  MAPE standard deviation: {mape_std:.2f}%")
    logger.info(f"  Consistency grade: {final_summary['consistency_grade']}")
    
    if mape_std > 5.0:  # High variance
        logger.warning(f"⚠️ High variance across splits ({mape_std:.2f}%)")
        validation_passed = False
    else:
        logger.info("✓ Consistency validation passed")
    
    # 3. Validate business readiness
    business_ready = final_summary['business_ready']
    overall_grade = final_summary['overall_grade']
    logger.info(f"Business readiness validation:")
    logger.info(f"  Overall grade: {overall_grade}")
    logger.info(f"  Business ready: {'YES' if business_ready else 'NO'}")
    
    if not business_ready:
        logger.warning("⚠️ Model not considered business-ready")
        validation_passed = False
    else:
        logger.info("✓ Business readiness validation passed")
    
    # 4. Compare with notebook if provided
    if args.notebook_mape:
        logger.info(f"Notebook comparison:")
        logger.info(f"  Notebook MAPE: {args.notebook_mape:.2f}%")
        logger.info(f"  New model MAPE: {avg_mape:.2f}%")
        
        performance_diff = avg_mape - args.notebook_mape
        if performance_diff > 5.0:  # Significantly worse
            logger.warning(f"⚠️ Performance worse than notebook by {performance_diff:.2f}%")
            validation_passed = False
        elif performance_diff < -2.0:  # Significantly better
            logger.info(f"✅ Performance better than notebook by {abs(performance_diff):.2f}%")
        else:
            logger.info("✓ Performance comparable to notebook")
    
    # 5. Validate split performance
    split_performance = comprehensive_results['split_details']
    logger.info(f"Split performance validation:")
    
    failed_splits = 0
    for split_num, results in split_performance.items():
        split_mape = results['val_mape']
        if split_mape > args.expected_mape * 1.5:  # Allow 50% tolerance for individual splits
            failed_splits += 1
    
    if failed_splits > len(split_performance) // 2:  # More than half splits failed
        logger.warning(f"⚠️ {failed_splits}/{len(split_performance)} splits performed poorly")
        validation_passed = False
    else:
        logger.info("✓ Split performance validation passed")
    
    return validation_passed

def generate_comparison_report(training_results, args, logger):
    """Generate detailed comparison report with notebook results."""
    logger.info("Generating comparison report with notebook...")
    
    # Create reports directory
    reports_dir = Path(args.output_dir) / "reports" / "phase2_validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_summary = training_results['final_summary']
    comprehensive_results = training_results['comprehensive_results']
    
    # Generate comparison report
    report_file = reports_dir / f"model_comparison_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PHASE 2 MODEL TRAINING VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("COMPARISON WITH NOTEBOOK RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Experiment: {training_results['experiment_metadata']['experiment_name']}\n")
        f.write(f"Expected MAPE: ≤{args.expected_mape:.1f}%\n")
        if args.notebook_mape:
            f.write(f"Notebook MAPE: {args.notebook_mape:.2f}%\n")
        f.write(f"Achieved MAPE: {final_summary['average_validation_mape']:.2f}%\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Grade: {final_summary['overall_grade']}\n")
        f.write(f"Consistency: {final_summary['consistency_grade']}\n")
        f.write(f"Business Ready: {'YES' if final_summary['business_ready'] else 'NO'}\n")
        f.write(f"Best Split MAPE: {final_summary['best_split_mape']:.2f}%\n")
        f.write(f"Worst Split MAPE: {final_summary['worst_split_mape']:.2f}%\n\n")
        
        # Individual split performance
        f.write("INDIVIDUAL SPLIT PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        split_results = training_results['training_results']
        for split_num, results in split_results.items():
            f.write(f"Split {split_num}: {results.get('description', 'N/A')}\n")
            f.write(f"  Validation MAPE: {results['val_mape']:.2f}%\n")
            f.write(f"  Validation RMSE: {results['val_rmse']:.0f}\n")
            f.write(f"  Validation R²: {results['val_r2']:.3f}\n")
            f.write(f"  Training samples: {results.get('train_samples', 'N/A'):,}\n")
            f.write(f"  Validation samples: {results.get('val_samples', 'N/A'):,}\n\n")
        
        # Platform performance if available
        if 'platform_performance' in comprehensive_results:
            f.write("PLATFORM-SPECIFIC PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for platform, perf in comprehensive_results['platform_performance'].items():
                f.write(f"{platform}:\n")
                f.write(f"  Average MAPE: {perf['mean_mape']:.2f}% ± {perf['std_mape']:.2f}%\n")
                f.write(f"  Range: {perf['min_mape']:.2f}% - {perf['max_mape']:.2f}%\n")
                f.write(f"  Splits analyzed: {perf['splits_analyzed']}\n\n")
        
        # Training efficiency
        training_efficiency = comprehensive_results['training_efficiency']
        f.write("TRAINING EFFICIENCY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average epochs used: {training_efficiency['average_epochs_used']:.1f}\n")
        f.write(f"Average best epoch: {training_efficiency['average_best_epoch']:.1f}\n")
        f.write(f"Early stopping rate: {training_efficiency['early_stopping_rate']:.1%}\n")
        f.write(f"Training stability: {training_efficiency['training_stability']:.3f}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for recommendation in final_summary['recommendations']:
            f.write(f"• {recommendation}\n")
        
        f.write("\nNEXT STEPS\n")
        f.write("-" * 40 + "\n")
        if final_summary['business_ready']:
            f.write("✅ Model is ready for production use\n")
            f.write("1. Review detailed predictions in outputs/predictions/\n")
            f.write("2. Validate results with business stakeholders\n")
            f.write("3. Deploy model for regular forecasting\n")
        else:
            f.write("⚠️ Model needs improvement before production\n")
            f.write("1. Review performance issues identified above\n")
            f.write("2. Consider tuning hyperparameters or adding features\n")
            f.write("3. Re-run training with adjustments\n")
    
    logger.info(f"Comparison report generated: {report_file}")
    return str(report_file)

def print_phase2_summary(training_results, validation_passed, args, logger):
    """Print Phase 2 execution summary."""
    logger.info("=" * 80)
    logger.info("PHASE 2 EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    final_summary = training_results['final_summary']
    saved_files = training_results['saved_files']
    
    logger.info(f"✓ Model training completed")
    logger.info(f"✓ Average MAPE: {final_summary['average_validation_mape']:.2f}%")
    logger.info(f"✓ Overall grade: {final_summary['overall_grade']}")
    logger.info(f"✓ Business ready: {'YES' if final_summary['business_ready'] else 'NO'}")
    logger.info(f"✓ Splits trained: {final_summary['total_splits_trained']}")
    
    if validation_passed:
        logger.info("✅ VALIDATION PASSED - Model ready for production")
        logger.info("\nMigration completed successfully!")
        logger.info("Your notebook has been successfully replaced with modular code.")
        logger.info("\nNext steps:")
        logger.info("1. Review the generated predictions and reports")
        logger.info("2. Compare performance with your notebook expectations")
        logger.info("3. Use the modular code for regular forecasting")
    else:
        logger.info("⚠️ VALIDATION ISSUES FOUND - Review before production")
        logger.info("\nRecommended actions:")
        logger.info("1. Review the performance issues mentioned above")
        logger.info("2. Consider adjusting training parameters")
        logger.info("3. Check if additional features or tuning needed")
    
    logger.info(f"\nGenerated outputs:")
    for key, filepath in saved_files.items():
        if filepath and filepath != 'N/A':
            logger.info(f"  {key}: {filepath}")
    
    # Performance comparison with notebook
    if args.notebook_mape:
        performance_diff = final_summary['average_validation_mape'] - args.notebook_mape
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