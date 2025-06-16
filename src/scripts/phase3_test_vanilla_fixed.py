#!/usr/bin/env python3
"""
Fixed Phase 3: Vanilla Model Evaluation on 2023 Data with Corrected Preprocessing

This script applies the same preprocessing fixes that were developed for the enhanced model
to ensure proper encoder/scaler consistency between training and inference.
"""

import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.feature_pipeline import FeaturePipeline
from models.vanilla_embedding_model import VanillaEmbeddingModel
from models.feature_processor import FeatureProcessor
from utils.logging_utils import setup_logging

class VanillaModel2023Evaluator:
    """Evaluator for vanilla models on 2023 data with fixed preprocessing."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(__name__)
        
        # Verify models directory exists
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
            
        self.logger.info(f"Initialized evaluator with models from: {self.models_dir}")
    
    def reconstruct_training_preprocessing(self, engineered_data: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Reconstruct the exact preprocessing used during training by re-running 
        the training data through the feature pipeline.
        """
        self.logger.info("🔧 RECONSTRUCTING TRAINING PREPROCESSING...")
        
        # Filter training data (2021-2022)
        training_mask = (engineered_data['date'] >= '2021-01-01') & (engineered_data['date'] <= '2022-12-31')
        training_data = engineered_data[training_mask].copy()
        
        self.logger.info(f"Training data: {len(training_data):,} samples from {training_data['date'].min()} to {training_data['date'].max()}")
        
        # Initialize feature processor with training data
        feature_processor = FeatureProcessor()
        
        # Process training data to get fitted encoders/scalers
        self.logger.info("Processing training data to fit encoders/scalers...")
        X_train_processed, y_train = feature_processor.prepare_features(training_data)
        
        # Store the fitted preprocessors
        training_preprocessors = {
            'feature_processor': feature_processor,
            'fitted_encoders': feature_processor.label_encoders,
            'fitted_scalers': feature_processor.scalers,
            'training_stats': {
                'samples': len(training_data),
                'feature_shapes': {k: v.shape for k, v in X_train_processed.items()},
                'date_range': (training_data['date'].min(), training_data['date'].max())
            }
        }
        
        self.logger.info("✅ Training preprocessing reconstructed successfully")
        return training_preprocessors, training_data
    
    def prepare_2023_data_with_training_preprocessing(self, 
                                                     engineered_data: pd.DataFrame,
                                                     training_preprocessors: Dict) -> Tuple[Dict, pd.DataFrame]:
        """
        Prepare 2023 data using the exact same preprocessing as training.
        """
        self.logger.info("🎯 PREPARING 2023 DATA WITH TRAINING PREPROCESSING...")
        
        # Filter 2023 data
        test_mask = engineered_data['date'] >= '2023-01-01'
        test_data = engineered_data[test_mask].copy()
        
        self.logger.info(f"2023 test data: {len(test_data):,} samples from {test_data['date'].min()} to {test_data['date'].max()}")
        
        # Use the fitted feature processor from training
        feature_processor = training_preprocessors['feature_processor']
        
        # Process 2023 data with training preprocessors (no fitting)
        X_test_processed, y_test = feature_processor.prepare_features(test_data, fit_preprocessors=False)
        
        self.logger.info("✅ 2023 data prepared with training preprocessing")
        return X_test_processed, y_test
    
    def load_model_and_evaluate(self, 
                               model_file: str,
                               X_test: Dict,
                               y_test: np.ndarray) -> Dict[str, Any]:
        """Load a model and evaluate on test data."""
        
        model_path = self.models_dir / model_file
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
            
        try:
            # Load model
            self.logger.info(f"Loading model: {model_file}")
            
            # Create model instance first
            model = VanillaEmbeddingModel()
            
            # Build model by calling it with sample data
            sample_batch = {k: v[:1] for k, v in X_test.items()}
            _ = model(sample_batch)
            
            # Load weights
            model.load_weights(str(model_path))
            
            # Make predictions
            self.logger.info("Making predictions on 2023 data...")
            y_pred = model.predict(X_test, verbose=0, batch_size=1024)
            
            # Flatten predictions if needed
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            results = {
                'model_file': model_file,
                'mape_percent': float(mape),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mae': float(mae),
                'test_samples': len(y_test),
                'predictions': y_pred.tolist(),
                'actuals': y_test.tolist()
            }
            
            self.logger.info(f"✅ {model_file}: MAPE={mape:.2f}%, RMSE={rmse:.1f}, R²={r2:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_file}: {str(e)}")
            return None
    
    def run_complete_evaluation(self, engineered_dataset_path: str) -> Dict[str, Any]:
        """Run complete evaluation with fixed preprocessing."""
        
        self.logger.info("=" * 80)
        self.logger.info("VANILLA MODEL 2023 EVALUATION - FIXED PREPROCESSING")
        self.logger.info("=" * 80)
        
        # Load engineered dataset
        self.logger.info(f"Loading engineered dataset: {engineered_dataset_path}")
        try:
            with open(engineered_dataset_path, 'rb') as f:
                engineered_data = pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return {}
        
        self.logger.info(f"Dataset loaded: {len(engineered_data):,} samples")
        
        # Step 1: Reconstruct training preprocessing
        training_preprocessors, training_data = self.reconstruct_training_preprocessing(engineered_data)
        
        # Step 2: Prepare 2023 data with training preprocessing
        X_test, y_test = self.prepare_2023_data_with_training_preprocessing(
            engineered_data, training_preprocessors
        )
        
        # Step 3: Find all model files
        model_files = sorted([f.name for f in self.models_dir.glob("*.keras")])
        
        if not model_files:
            self.logger.error("No .keras model files found!")
            return {}
        
        self.logger.info(f"Found {len(model_files)} model files to evaluate")
        
        # Step 4: Evaluate each model
        all_results = {}
        split_summaries = []
        
        for i, model_file in enumerate(model_files, 1):
            self.logger.info(f"\n--- EVALUATING MODEL {i}/{len(model_files)} ---")
            
            # Extract split info from filename
            try:
                split_num = int(model_file.split('_')[3])
            except:
                split_num = i
            
            # Get split description
            split_descriptions = {
                1: "2021 Full Year → 2022 Q1",
                2: "2021 + 2022 Q1 → 2022 Q2", 
                3: "2021 + 2022 H1 → 2022 Q3",
                4: "2021 + 2022 Q1-Q3 → 2022 Q4",
                5: "2021 + 2022 Full → 2023 Q1"
            }
            
            split_desc = split_descriptions.get(split_num, f"Split {split_num}")
            
            # Evaluate model
            results = self.load_model_and_evaluate(model_file, X_test, y_test)
            
            if results:
                results['split_number'] = split_num
                results['split_description'] = split_desc
                all_results[f'split_{split_num}'] = results
                
                # Add to summary
                split_summaries.append({
                    'split_number': split_num,
                    'split_description': split_desc,
                    'mape_percent': results['mape_percent'],
                    'rmse': results['rmse'],
                    'r2_score': results['r2_score'],
                    'mae': results['mae'],
                    'test_samples': results['test_samples'],
                    'model_file': model_file
                })
        
        # Step 5: Generate summary statistics
        if split_summaries:
            mapes = [s['mape_percent'] for s in split_summaries]
            avg_mape = np.mean(mapes)
            std_mape = np.std(mapes)
            best_mape = min(mapes)
            worst_mape = max(mapes)
            
            summary = {
                'evaluation_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'total_models_evaluated': len(split_summaries),
                'average_mape': float(avg_mape),
                'mape_std': float(std_mape),
                'best_mape': float(best_mape),
                'worst_mape': float(worst_mape),
                'split_results': split_summaries,
                'detailed_results': all_results
            }
            
            # Step 6: Save results
            self.save_results(summary)
            
            # Step 7: Generate plots
            self.generate_evaluation_plots(summary)
            
            # Step 8: Print summary
            self.print_evaluation_summary(summary)
            
            return summary
        
        else:
            self.logger.error("No successful evaluations!")
            return {}
    
    def save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files."""
        
        timestamp = summary['evaluation_timestamp']
        output_dir = self.models_dir.parent / "2023_evaluation"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON
        json_file = output_dir / f"2023_evaluation_results_FIXED_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary CSV
        csv_file = output_dir / f"2023_evaluation_summary_FIXED_{timestamp}.csv"
        df = pd.DataFrame(summary['split_results'])
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"✅ Results saved:")
        self.logger.info(f"   JSON: {json_file}")
        self.logger.info(f"   CSV: {csv_file}")
    
    def generate_evaluation_plots(self, summary: Dict[str, Any]):
        """Generate evaluation plots."""
        
        timestamp = summary['evaluation_timestamp']
        output_dir = self.models_dir.parent / "2023_evaluation"
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Vanilla Model 2023 Evaluation Results - FIXED PREPROCESSING', fontsize=16, fontweight='bold')
        
        # Extract data
        splits = summary['split_results']
        split_nums = [s['split_number'] for s in splits]
        mapes = [s['mape_percent'] for s in splits]
        rmses = [s['rmse'] for s in splits]
        r2s = [s['r2_score'] for s in splits]
        
        # Plot 1: MAPE by Split
        axes[0,0].bar(split_nums, mapes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[0,0].set_title('MAPE by Split', fontweight='bold')
        axes[0,0].set_xlabel('Split Number')
        axes[0,0].set_ylabel('MAPE (%)')
        axes[0,0].set_ylim(0, max(mapes) * 1.1)
        for i, v in enumerate(mapes):
            axes[0,0].text(split_nums[i], v + max(mapes)*0.01, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Plot 2: R² by Split  
        colors = ['red' if r < 0 else 'green' for r in r2s]
        axes[0,1].bar(split_nums, r2s, color=colors, alpha=0.7)
        axes[0,1].set_title('R² Score by Split', fontweight='bold')
        axes[0,1].set_xlabel('Split Number')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for i, v in enumerate(r2s):
            axes[0,1].text(split_nums[i], v + 0.01 if v >= 0 else v - 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Plot 3: RMSE by Split
        axes[0,2].bar(split_nums, rmses, color='orange', alpha=0.7)
        axes[0,2].set_title('RMSE by Split', fontweight='bold')
        axes[0,2].set_xlabel('Split Number')
        axes[0,2].set_ylabel('RMSE')
        
        # Plot 4: Performance Summary Table
        axes[1,0].axis('tight')
        axes[1,0].axis('off')
        table_data = []
        for s in splits:
            table_data.append([
                f"Split {s['split_number']}",
                f"{s['mape_percent']:.1f}%",
                f"{s['r2_score']:.3f}",
                f"{s['rmse']:.0f}"
            ])
        
        table = axes[1,0].table(cellText=table_data,
                               colLabels=['Split', 'MAPE', 'R²', 'RMSE'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,0].set_title('Performance Summary', fontweight='bold')
        
        # Plot 5: Overall Statistics
        axes[1,1].axis('off')
        stats_text = f"""
FIXED PREPROCESSING RESULTS

Average MAPE: {summary['average_mape']:.2f}% ± {summary['mape_std']:.2f}%
Best Split MAPE: {summary['best_mape']:.2f}%
Worst Split MAPE: {summary['worst_mape']:.2f}%

Models Evaluated: {summary['total_models_evaluated']}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Performance Grade: {"EXCELLENT" if summary['average_mape'] < 15 else "GOOD" if summary['average_mape'] < 25 else "MODERATE" if summary['average_mape'] < 40 else "POOR"}
        """
        axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1,1].set_title('Evaluation Summary', fontweight='bold')
        
        # Plot 6: Best Model Scatter Plot (if available)
        best_split = min(splits, key=lambda x: x['mape_percent'])
        best_results = summary['detailed_results'][f"split_{best_split['split_number']}"]
        
        if 'predictions' in best_results and 'actuals' in best_results:
            # Sample data for plotting (avoid too many points)
            actuals = np.array(best_results['actuals'])
            predictions = np.array(best_results['predictions'])
            
            # Sample if too many points
            if len(actuals) > 5000:
                indices = np.random.choice(len(actuals), 5000, replace=False)
                actuals = actuals[indices]
                predictions = predictions[indices]
            
            axes[1,2].scatter(actuals, predictions, alpha=0.5, s=1)
            axes[1,2].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
            axes[1,2].set_xlabel('Actual Sales')
            axes[1,2].set_ylabel('Predicted Sales')
            axes[1,2].set_title(f'Best Model (Split {best_split["split_number"]}) - MAPE: {best_split["mape_percent"]:.1f}%', fontweight='bold')
        else:
            axes[1,2].text(0.5, 0.5, 'Prediction data\nnot available', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Actual vs Predicted', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"vanilla_model_2023_evaluation_FIXED_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Plots saved: {plot_file}")
    
    def print_evaluation_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary to console."""
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VANILLA MODEL 2023 EVALUATION SUMMARY - FIXED PREPROCESSING")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Models Evaluated: {summary['total_models_evaluated']}")
        self.logger.info(f"Average MAPE: {summary['average_mape']:.2f}% ± {summary['mape_std']:.2f}%")
        self.logger.info(f"Best Split MAPE: {summary['best_mape']:.2f}%")
        self.logger.info(f"Worst Split MAPE: {summary['worst_mape']:.2f}%")
        
        # Performance grade
        avg_mape = summary['average_mape']
        if avg_mape < 15:
            grade = "EXCELLENT"
        elif avg_mape < 25:
            grade = "GOOD"
        elif avg_mape < 40:
            grade = "MODERATE"
        else:
            grade = "POOR"
            
        self.logger.info(f"Overall Grade: {grade}")
        
        self.logger.info("\nDetailed Results by Split:")
        self.logger.info("-" * 80)
        
        for split in summary['split_results']:
            self.logger.info(f"Split {split['split_number']}: {split['split_description']}")
            self.logger.info(f"  MAPE: {split['mape_percent']:.2f}%")
            self.logger.info(f"  RMSE: {split['rmse']:.1f}")
            self.logger.info(f"  R²: {split['r2_score']:.3f}")
            self.logger.info("")
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 VANILLA MODEL EVALUATION COMPLETED!")
        self.logger.info("=" * 80)

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vanilla Model 2023 Evaluation - Fixed Preprocessing")
    
    parser.add_argument("--engineered-dataset",
                       type=str,
                       required=True,
                       help="Path to the engineered dataset pickle file")
    
    parser.add_argument("--models-dir",
                       type=str,
                       help="Directory containing trained vanilla models (required if not using --experiment-name)")
    
    parser.add_argument("--experiment-name",
                       type=str,
                       help="Experiment name to automatically find models (alternative to --models-dir)")
    
    parser.add_argument("--log-level",
                       type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("VANILLA MODEL 2023 EVALUATION - FIXED PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.engineered_dataset}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Determine models directory
    if args.models_dir:
        models_dir = args.models_dir
    elif args.experiment_name:
        models_dir = f"outputs/{args.experiment_name}/vanilla_models"
    else:
        logger.error("Either --models-dir or --experiment-name must be provided")
        return
    
    logger.info(f"Models directory: {models_dir}")
    
    try:
        # Initialize evaluator
        evaluator = VanillaModel2023Evaluator(models_dir)
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(args.engineered_dataset)
        
        if results:
            logger.info("\n🎉 Vanilla model evaluation completed successfully!")
            logger.info(f"Average MAPE: {results['average_mape']:.2f}%")
        else:
            logger.error("❌ Evaluation failed!")
            
    except Exception as e:
        logger.error(f"❌ Fatal error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 