#!/usr/bin/env python3
"""
Fixed Model Testing Script - Load and Test Best Models on 2023 Data

This script properly loads the TensorFlow models saved by phase2_model_training.py
and tests them on 2023 data. Based on analysis of the saving mechanism.

Usage:
    python fixed_model_test.py --engineered-dataset path_to_dataset.pkl

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import AdamW

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent if len(Path(__file__).parents) > 2 else Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.feature_pipeline import SalesFeaturePipeline
from utils.helpers import setup_logging


class FeatureSliceLayer(layers.Layer):
    """
    Custom Keras layer to extract a single feature from input tensor.
    
    This replaces the problematic Lambda layer and is fully serializable.
    Extracts the feature at the specified index while maintaining proper tensor shapes.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
    
    def call(self, inputs):
        # Extract single feature at index and expand dims to maintain shape
        return tf.expand_dims(inputs[:, self.index], axis=1)
    
    def get_config(self):
        """Enable proper serialization"""
        config = super().get_config()
        config.update({"index": self.index})
        return config
    
    @classmethod
    def from_config(cls, config):
        """Enable proper deserialization"""
        return cls(**config)

# CRITICAL: Define the EXACT custom functions as they were defined during training
def mape_metric_original_scale(y_true, y_pred):
    """MAPE metric in original scale for monitoring during training - EXACT COPY"""
    y_true_orig = tf.exp(y_true) - 1
    y_pred_orig = tf.exp(y_pred) - 1
    y_true_orig = tf.clip_by_value(y_true_orig, 1.0, 1e6)
    y_pred_orig = tf.clip_by_value(y_pred_orig, 1.0, 1e6)
    epsilon = 1.0
    mape = tf.reduce_mean(tf.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)) * 100
    return tf.clip_by_value(mape, 0.0, 1000.0)

def rmse_metric_original_scale(y_true, y_pred):
    """RMSE in original scale for monitoring during training - EXACT COPY"""
    y_true_orig = tf.exp(y_true) - 1
    y_pred_orig = tf.exp(y_pred) - 1
    y_true_orig = tf.clip_by_value(y_true_orig, 1.0, 1e6)
    y_pred_orig = tf.clip_by_value(y_pred_orig, 1.0, 1e6)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true_orig - y_pred_orig)))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test best TensorFlow models on 2023 data")
    
    parser.add_argument("--engineered-dataset", 
                       type=str, 
                       required=True,
                       help="Path to engineered dataset pickle file")
    
    parser.add_argument("--models-dir",
                       type=str,
                       default="outputs/models",
                       help="Directory containing trained models")
    
    parser.add_argument("--output-dir",
                       type=str,
                       default="outputs/2023_evaluation_fixed",
                       help="Output directory for evaluation results")
    
    parser.add_argument("--log-level",
                       type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()

class FixedModel2023Evaluator:
    """Fixed evaluator that properly loads TensorFlow models."""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split meanings for legend
        self.split_meanings = {
            1: "2021 Full Year → 2022 Q1",
            2: "2021 + 2022 Q1 → 2022 Q2", 
            3: "2021 + 2022 H1 → 2022 Q3",
            4: "2021 + 2022 Q1-Q3 → 2022 Q4"
        }
        
        self.evaluation_results = {}
        self.logger = logging.getLogger(__name__)
    
    def load_engineered_2023_data(self, engineered_dataset_path: str):
        """Load 2023 data from the engineered dataset."""
        self.logger.info("Loading 2023 data from engineered dataset...")
        
        feature_pipeline = SalesFeaturePipeline()
        df_final, modeling_features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(
            engineered_dataset_path
        )
        
        # Filter for 2023 data
        df_final['sales_month'] = pd.to_datetime(df_final['sales_month'])
        df_2023 = df_final[df_final['sales_month'].dt.year == 2023].copy()
        
        if len(df_2023) == 0:
            raise ValueError("No 2023 data found in the engineered dataset!")
        
        self.logger.info(f"✓ 2023 data loaded successfully:")
        self.logger.info(f"  Records: {len(df_2023):,}")
        self.logger.info(f"  Date range: {df_2023['sales_month'].min()} to {df_2023['sales_month'].max()}")
        self.logger.info(f"  Features: {len(modeling_features)}")
        
        # Store for later use
        self.modeling_features = modeling_features
        self.df_2023 = df_2023
        
        return df_2023, modeling_features
    
    def find_best_models(self):
        """Find the best model files for each split."""
        self.logger.info("Searching for best TensorFlow models...")
        
        best_models = {}
        
        # Look for model files with pattern: best_model_split_X_*.h5 or advanced_embedding_model_split_X_*.h5
        for split_num in range(1, 5):  # Splits 1-4
            patterns = [
                f"best_model_split_{split_num}_*.h5",
                f"advanced_embedding_model_split_{split_num}_*.h5",
                f"model_split_{split_num}_*.h5",
                f"*split_{split_num}*.h5"
            ]
            
            model_file_found = None
            for pattern in patterns:
                model_files = list(self.models_dir.glob(pattern))
                if model_files:
                    # Choose the most recent model file
                    model_file_found = max(model_files, key=lambda x: x.stat().st_mtime)
                    break
            
            if model_file_found:
                best_models[split_num] = str(model_file_found)
                self.logger.info(f"  Split {split_num}: {model_file_found.name}")
            else:
                self.logger.warning(f"  Split {split_num}: No .h5 model found!")
        
        if not best_models:
            # Debug: show what files are actually in the directory
            all_files = list(self.models_dir.glob("*"))
            self.logger.error(f"No .h5 model files found in {self.models_dir}")
            self.logger.error("Available files:")
            for file in all_files[:10]:  # Show first 10 files
                self.logger.error(f"  {file.name}")
            raise FileNotFoundError(f"No .h5 model files found in {self.models_dir}")
        
        self.logger.info(f"Found {len(best_models)} TensorFlow models")
        return best_models
    
    def load_tensorflow_model_properly(self, model_path: str):
        """Load TensorFlow model with the correct custom objects."""
        self.logger.info(f"    Loading model: {Path(model_path).name}")
        
        # Create the custom objects dictionary with EXACT functions
        custom_objects = {
            'mape_metric_original_scale': mape_metric_original_scale,
            'rmse_metric_original_scale': rmse_metric_original_scale,
            'FeatureSliceLayer': FeatureSliceLayer  # Add this line
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        return model
    
    def prepare_features_for_prediction(self, df_2023: pd.DataFrame, features: list):
        """Prepare features for model prediction."""
        self.logger.info("    Preparing features for prediction...")
        
        # Ensure all required features exist
        missing_features = [f for f in features if f not in df_2023.columns]
        if missing_features:
            self.logger.warning(f"    Missing features: {len(missing_features)} features")
            # Fill missing features with 0
            for feature in missing_features:
                df_2023[feature] = 0
        
        # Extract feature matrix
        X = df_2023[features].fillna(0).values
        
        # Handle any remaining NaN values
        if np.isnan(X).any():
            self.logger.warning("    Found NaN values in features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        self.logger.info(f"    ✓ Features prepared: {X.shape}")
        return X
    
    def evaluate_single_model(self, model_path: str, split_num: int):
        """Evaluate a single model on 2023 data."""
        self.logger.info(f"\n--- Evaluating Split {split_num} ---")
        self.logger.info(f"Model: {Path(model_path).name}")
        
        # Load the model
        model = self.load_tensorflow_model_properly(model_path)
        if model is None:
            self.logger.error(f"  ❌ Failed to load model for Split {split_num}")
            return None
        
        try:
            # Prepare test data
            X_test = self.prepare_features_for_prediction(self.df_2023, self.modeling_features)
            
            # Prepare target variable
            if 'sales_quantity_log' in self.df_2023.columns:
                y_test = self.df_2023['sales_quantity_log'].values
            else:
                # Create log-transformed target if it doesn't exist
                y_test = np.log1p(self.df_2023['sales_quantity'].values)
                self.logger.info("    Created log-transformed target from sales_quantity")
            
            # Make predictions
            self.logger.info("    Making predictions...")
            y_pred_log = model.predict(X_test, verbose=0)
            
            # Convert back to original scale
            y_pred_orig = np.exp(y_pred_log.flatten()) - 1
            y_true_orig = np.exp(y_test) - 1
            
            # Ensure positive values
            y_pred_orig = np.maximum(y_pred_orig, 1.0)
            y_true_orig = np.maximum(y_true_orig, 1.0)
            
            # Remove any invalid values
            valid_mask = np.isfinite(y_pred_orig) & np.isfinite(y_true_orig)
            y_pred_orig = y_pred_orig[valid_mask]
            y_true_orig = y_true_orig[valid_mask]
            
            if len(y_pred_orig) == 0:
                raise ValueError("No valid predictions after filtering")
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            r2 = r2_score(y_true_orig, y_pred_orig)
            mae = np.mean(np.abs(y_true_orig - y_pred_orig))
            
            results = {
                'split_num': split_num,
                'split_description': self.split_meanings[split_num],
                'model_path': model_path,
                'mape': mape,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'predictions': y_pred_orig,
                'actuals': y_true_orig,
                'test_samples': len(y_true_orig),
                'valid_predictions': len(y_pred_orig)
            }
            
            self.logger.info(f"  ✅ MAPE: {mape:.2f}%")
            self.logger.info(f"     RMSE: {rmse:.0f}")
            self.logger.info(f"     R²: {r2:.3f}")
            self.logger.info(f"     Valid predictions: {len(y_pred_orig):,}/{len(y_test):,}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"  ❌ Prediction failed for Split {split_num}: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def create_comparison_visualizations(self, all_results: dict) -> str:
        """Create comprehensive comparison visualizations."""
        self.logger.info("Creating comparison visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TensorFlow Model Performance Comparison on 2023 Data', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        splits = list(all_results.keys())
        mapes = [all_results[s]['mape'] for s in splits]
        rmses = [all_results[s]['rmse'] for s in splits]
        r2s = [all_results[s]['r2'] for s in splits]
        
        # Split descriptions for legend
        short_labels = [f"Split {s}" for s in splits]
        
        # 1. MAPE Comparison
        bars1 = axes[0,0].bar(short_labels, mapes, color=sns.color_palette("husl", len(splits)))
        axes[0,0].set_title('MAPE Comparison (%)', fontweight='bold')
        axes[0,0].set_ylabel('MAPE (%)')
        axes[0,0].set_ylim(0, max(mapes) * 1.1)
        
        # Add value labels on bars
        for bar, mape in zip(bars1, mapes):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mapes)*0.01,
                          f'{mape:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE Comparison
        bars2 = axes[0,1].bar(short_labels, rmses, color=sns.color_palette("husl", len(splits)))
        axes[0,1].set_title('RMSE Comparison', fontweight='bold')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].set_ylim(0, max(rmses) * 1.1)
        
        for bar, rmse in zip(bars2, rmses):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmses)*0.01,
                          f'{rmse:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. R² Comparison
        bars3 = axes[0,2].bar(short_labels, r2s, color=sns.color_palette("husl", len(splits)))
        axes[0,2].set_title('R² Score Comparison', fontweight='bold')
        axes[0,2].set_ylabel('R² Score')
        axes[0,2].set_ylim(min(0, min(r2s) * 1.1), max(r2s) * 1.1)
        
        for bar, r2 in zip(bars3, r2s):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prediction vs Actual Scatter (Best Model)
        best_split = min(splits, key=lambda s: all_results[s]['mape'])
        best_results = all_results[best_split]
        
        axes[1,0].scatter(best_results['actuals'], best_results['predictions'], 
                         alpha=0.6, color=sns.color_palette("husl", len(splits))[splits.index(best_split)])
        
        # Perfect prediction line
        min_val = min(best_results['actuals'].min(), best_results['predictions'].min())
        max_val = max(best_results['actuals'].max(), best_results['predictions'].max())
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        axes[1,0].set_xlabel('Actual Sales')
        axes[1,0].set_ylabel('Predicted Sales')
        axes[1,0].set_title(f'Best Model: Split {best_split} Predictions vs Actuals\n'
                           f'MAPE: {best_results["mape"]:.2f}%', fontweight='bold')
        
        # 5. Performance Summary Table
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        
        table_data = []
        for split in splits:
            results = all_results[split]
            table_data.append([
                f"Split {split}",
                f"{results['mape']:.2f}%",
                f"{results['rmse']:.0f}",
                f"{results['r2']:.3f}"
            ])
        
        table = axes[1,1].table(cellText=table_data,
                               colLabels=['Split', 'MAPE', 'RMSE', 'R²'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,1].set_title('Performance Summary', fontweight='bold')
        
        # 6. Split Descriptions Legend
        axes[1,2].axis('off')
        legend_text = "Split Training Descriptions:\n\n"
        for split, description in self.split_meanings.items():
            if split in splits:
                legend_text += f"Split {split}: {description}\n"
        
        axes[1,2].text(0.05, 0.95, legend_text, transform=axes[1,2].transAxes, 
                      fontsize=10, verticalalignment='top', fontweight='bold')
        axes[1,2].set_title('Split Meanings', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.output_dir / f'tensorflow_model_comparison_2023_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✓ Visualizations saved: {viz_file}")
        return str(viz_file)
    
    def generate_evaluation_report(self, all_results: dict) -> str:
        """Generate comprehensive evaluation report."""
        self.logger.info("Generating evaluation report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f'tensorflow_evaluation_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TENSORFLOW MODEL EVALUATION ON 2023 DATA - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: TensorFlow Advanced Embedding Models\n")
            f.write(f"Total models evaluated: {len(all_results)}\n")
            f.write(f"Test period: 2023 (Full Year)\n")
            f.write(f"Test samples: {list(all_results.values())[0]['test_samples']:,}\n\n")
            
            # Overall performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            mapes = [r['mape'] for r in all_results.values()]
            f.write(f"Best MAPE: {min(mapes):.2f}%\n")
            f.write(f"Worst MAPE: {max(mapes):.2f}%\n")
            f.write(f"Average MAPE: {np.mean(mapes):.2f}%\n")
            f.write(f"MAPE Range: {max(mapes) - min(mapes):.2f}%\n\n")
            
            # Individual model performance
            f.write("DETAILED MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            
            # Sort by MAPE performance
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mape'])
            
            for rank, (split, results) in enumerate(sorted_results, 1):
                f.write(f"RANK {rank}: SPLIT {split}\n")
                f.write(f"Training Strategy: {results['split_description']}\n")
                f.write(f"MAPE: {results['mape']:.2f}%\n")
                f.write(f"RMSE: {results['rmse']:.0f}\n")
                f.write(f"R²: {results['r2']:.3f}\n")
                f.write(f"MAE: {results['mae']:.0f}\n")
                f.write(f"Valid Predictions: {results['valid_predictions']:,}/{results['test_samples']:,}\n")
                f.write(f"Model: {Path(results['model_path']).name}\n\n")
            
            # Business insights
            f.write("BUSINESS INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            best_split = min(all_results.keys(), key=lambda s: all_results[s]['mape'])
            best_results = all_results[best_split]
            
            f.write(f"✅ BEST MODEL: Split {best_split}\n")
            f.write(f"   Strategy: {best_results['split_description']}\n")
            f.write(f"   Performance: {best_results['mape']:.2f}% MAPE\n\n")
            
            if best_results['mape'] < 5:
                f.write("🎯 EXCELLENT PERFORMANCE: Model achieves < 5% MAPE\n")
                f.write("   Ready for production deployment\n")
            elif best_results['mape'] < 10:
                f.write("✅ GOOD PERFORMANCE: Model achieves < 10% MAPE\n")
                f.write("   Suitable for business forecasting\n")
            else:
                f.write("⚠️ MODERATE PERFORMANCE: Model achieves > 10% MAPE\n")
                f.write("   Consider model improvements\n")
            
            # Training vs 2023 comparison
            f.write("\nTRAINING VS 2023 PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write("Expected training performance (from logs):\n")
            f.write("  Split 1: ~6.55% MAPE\n")
            f.write("  Split 2: ~3.34% MAPE\n") 
            f.write("  Split 3: ~3.78% MAPE\n")
            f.write("  Split 4: ~3.66% MAPE\n\n")
            
            f.write("Actual 2023 performance:\n")
            for split in sorted(all_results.keys()):
                mape = all_results[split]['mape']
                f.write(f"  Split {split}: {mape:.2f}% MAPE\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Deploy the best performing model for production forecasting\n")
            f.write("2. Monitor performance monthly and retrain quarterly\n")
            f.write("3. Consider ensemble methods combining multiple splits if needed\n")
            f.write("4. Use this model for strategic planning and inventory management\n")
        
        self.logger.info(f"✓ Report saved: {report_file}")
        return str(report_file)
    
    def run_complete_evaluation(self, engineered_dataset_path: str):
        """Run complete evaluation pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING TENSORFLOW MODEL EVALUATION ON 2023 DATA")
        self.logger.info("=" * 80)
        
        # 1. Load 2023 data
        df_2023, modeling_features = self.load_engineered_2023_data(engineered_dataset_path)
        
        # 2. Find best models
        best_models = self.find_best_models()
        
        # 3. Evaluate each model
        all_results = {}
        
        for split_num, model_path in best_models.items():
            result = self.evaluate_single_model(model_path, split_num)
            
            if result:
                all_results[split_num] = result
        
        if not all_results:
            raise RuntimeError("No TensorFlow models could be evaluated successfully!")
        
        # 4. Create visualizations
        viz_file = self.create_comparison_visualizations(all_results)
        
        # 5. Generate report
        report_file = self.generate_evaluation_report(all_results)
        
        # 6. Save detailed results
        results_file = self.output_dir / f'tensorflow_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json_results = {}
            for split, results in all_results.items():
                json_results[split] = {
                    'split_num': results['split_num'],
                    'split_description': results['split_description'],
                    'mape': results['mape'],
                    'rmse': results['rmse'],
                    'r2': results['r2'],
                    'mae': results['mae'],
                    'test_samples': results['test_samples'],
                    'valid_predictions': results['valid_predictions'],
                    'model_path': results['model_path']
                }
            json.dump(json_results, f, indent=2)
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TENSORFLOW EVALUATION COMPLETED")
        self.logger.info("=" * 80)
        
        best_split = min(all_results.keys(), key=lambda s: all_results[s]['mape'])
        best_mape = all_results[best_split]['mape']
        
        self.logger.info(f"✅ Best Model: Split {best_split} ({self.split_meanings[best_split]})")
        self.logger.info(f"✅ Best MAPE: {best_mape:.2f}%")
        self.logger.info(f"✅ Models evaluated: {len(all_results)}")
        self.logger.info(f"✅ Test samples: {list(all_results.values())[0]['test_samples']:,}")
        
        self.logger.info(f"\nGenerated files:")
        self.logger.info(f"  📊 Visualization: {viz_file}")
        self.logger.info(f"  📄 Report: {report_file}")
        self.logger.info(f"  📋 Results: {results_file}")
        
        return {
            'all_results': all_results,
            'best_split': best_split,
            'best_mape': best_mape,
            'visualization_file': viz_file,
            'report_file': report_file,
            'results_file': str(results_file)
        }

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("FIXED TENSORFLOW MODEL EVALUATION ON 2023 DATA")
    logger.info("=" * 80)
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.engineered_dataset}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Initialize evaluator
        evaluator = FixedModel2023Evaluator(args.models_dir, args.output_dir)
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(args.engineered_dataset)
        
        logger.info("\n🎉 TensorFlow model evaluation completed successfully!")
        logger.info(f"Check {args.output_dir} for detailed results")
        
        # Performance analysis
        best_mape = results['best_mape']
        expected_range = "3.34-6.55%"  # From your training logs
        
        if best_mape <= 7:
            logger.info(f"🌟 Excellent performance: {best_mape:.2f}% MAPE")
            logger.info(f"This matches your training performance range ({expected_range})")
        elif best_mape <= 15:
            logger.info(f"✅ Good performance: {best_mape:.2f}% MAPE")
            logger.info(f"Slightly higher than training range ({expected_range}) but acceptable")
        else:
            logger.info(f"⚠️ Performance degradation: {best_mape:.2f}% MAPE")
            logger.info(f"Significantly higher than training range ({expected_range})")
            logger.info("Consider investigating overfitting or data distribution shifts")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)