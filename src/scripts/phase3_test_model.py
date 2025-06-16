#!/usr/bin/env python3
"""
Phase 3: Test Enhanced Models on 2023 Data

This script loads trained TensorFlow models and evaluates them on 2023 data with
proper preprocessing consistency. Features include:
- Proper loading of training encoders/scalers for consistent preprocessing
- Maintains embedding vocabulary sizes and bucket counts from training
- Clean visualization generation for performance analysis

Usage:
    python phase3_test_model.py --models-dir "outputs/enhanced-model/models" --engineered-dataset "path/to/dataset.pkl"
"""

import argparse
import sys
import os
import numpy as np
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
import pickle
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent if len(Path(__file__).parents) > 2 else Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.feature_pipeline import SalesFeaturePipeline
from utils.helpers import setup_logging
from models.custom_objects import get_custom_objects

class Model2023Evaluator:
    """Enhanced model evaluator that loads training encoders/scalers for consistent preprocessing."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        
        # Auto-determine output directory
        experiment_dir = self.models_dir.parent
        self.output_dir = experiment_dir / "2023_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split meanings for legend
        self.split_meanings = {
            1: "2021 Full Year → 2022 Q1",
            2: "2021 + 2022 Q1 → 2022 Q2", 
            3: "2021 + 2022 H1 → 2022 Q3",
            4: "2021 + 2022 Q1-Q3 → 2022 Q4",
            5: "2021 + 2022 Full → 2023 Q1"
        }
        
        self.evaluation_results = {}
        self.logger = logging.getLogger(__name__)
        
        # Storage for loaded training encoders/scalers
        self.training_encoders = {}
        self.training_scalers = {}
    
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
        
        # Look for model files - check splits 1-10 to catch all models
        for split_num in range(1, 11):
            patterns = [
                f"best_model_split_{split_num}_*.keras",
                f"best_model_split_{split_num}_*.h5",
                f"advanced_embedding_model_split_{split_num}_*.keras",
                f"advanced_embedding_model_split_{split_num}_*.h5",
                f"model_split_{split_num}_*.keras",
                f"model_split_{split_num}_*.h5",
                f"*split_{split_num}*.keras",
                f"*split_{split_num}*.h5"
            ]
            
            model_file_found = None
            for pattern in patterns:
                model_files = list(self.models_dir.glob(pattern))
                if model_files:
                    # Filter out models with epoch_000 (initial checkpoints) and prefer models with higher MAPE values
                    valid_models = [f for f in model_files if "epoch_000" not in f.name or "mape_00.00" not in f.name]
                    if valid_models:
                        # Choose the model with the highest MAPE in filename (indicates trained model)
                        model_file_found = max(valid_models, key=lambda x: x.stat().st_mtime)
                    else:
                        # Fallback to most recent if no valid models
                        model_file_found = max(model_files, key=lambda x: x.stat().st_mtime)
                    break
            
            if model_file_found:
                best_models[split_num] = str(model_file_found)
                self.logger.info(f"  Split {split_num}: {model_file_found.name}")
        
        if not best_models:
            # Debug: show what files are actually in the directory
            all_files = list(self.models_dir.glob("*"))
            self.logger.error(f"No model files (.h5 or .keras) found in {self.models_dir}")
            self.logger.error("Available files:")
            for file in all_files[:10]:  # Show first 10 files
                self.logger.error(f"  {file.name}")
            raise FileNotFoundError(f"No model files (.h5 or .keras) found in {self.models_dir}")
        
        self.logger.info(f"Found {len(best_models)} TensorFlow models")
        return best_models
    
    def load_training_encoders_scalers(self, split_num: int):
        """
        Load the training encoders and scalers for a specific split.
        
        This is the KEY FIX: We need to recreate the exact same preprocessing
        that was used during training for this specific split.
        """
        self.logger.info(f"    Loading training encoders/scalers for Split {split_num}...")
        
        # Try to find saved encoders/scalers file
        encoders_file = self.models_dir / f"encoders_scalers_split_{split_num}.pkl"
        
        if encoders_file.exists():
            self.logger.info(f"    Found saved encoders: {encoders_file}")
            with open(encoders_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.training_encoders = saved_data.get('encoders', {})
                self.training_scalers = saved_data.get('scalers', {})
            return True
        else:
            self.logger.warning(f"    No saved encoders found: {encoders_file}")
            self.logger.warning("    Will need to reconstruct from training data...")
            return False
    
    def reconstruct_training_preprocessing(self, split_num: int):
        """Reconstruct training preprocessing to get the exact encoders/scalers used."""
        self.logger.info(f"    Reconstructing training preprocessing for Split {split_num}...")
        
        from models.enhanced_embedding_model import EnhancedEmbeddingModel
        
        # Create a model instance
        enhanced_model = EnhancedEmbeddingModel()
        
        # Load the full dataset to get training data for this split
        feature_pipeline = SalesFeaturePipeline()
        df_final, modeling_features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(
            "data/engineered/sales_forecast_engineered_dataset_20250612_164550.pkl"
        )
        
        # Get the specific training split
        if split_num <= len(rolling_splits):
            train_df, val_df, description = rolling_splits[split_num - 1]
            
            # Recreate the feature categories and preprocessing
            feature_categories = enhanced_model.categorize_features_for_embeddings(df_final, modeling_features)
            
            # Run the training preprocessing to build encoders/scalers
            X_train, y_train = enhanced_model.prepare_embedding_features(
                train_df, feature_categories, is_training=True
            )
            
            # Store the encoders/scalers
            self.training_encoders = enhanced_model.encoders.copy()
            self.training_scalers = enhanced_model.scalers.copy()
            
            self.logger.info(f"    ✓ Reconstructed {len(self.training_encoders)} encoders and {len(self.training_scalers)} scalers")
            return True
        else:
            self.logger.error(f"    Split {split_num} not found in rolling splits")
            return False
    
    def load_tensorflow_model_properly(self, model_path: str):
        """Load TensorFlow model with the correct custom objects."""
        self.logger.info(f"    Loading model: {Path(model_path).name}")
        
        # Get the custom objects dictionary with all required functions
        custom_objects = get_custom_objects()
        
        try:
            # Enable unsafe deserialization for lambda functions
            tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            self.logger.error(f"    Failed to load model: {str(e)}")
            # Try without compilation as fallback
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                self.logger.warning("    Loaded without compilation")
                return model
            except Exception as e2:
                self.logger.error(f"    All loading strategies failed: {str(e2)}")
                return None
    
    def prepare_features_with_training_preprocessing(self, df_2023: pd.DataFrame, features: list):
        """
        FIXED: Prepare features using the EXACT same preprocessing as training.
        
        This is the core fix that ensures consistency between training and inference.
        """
        self.logger.info("    Preparing features with training preprocessing...")
        
        from models.enhanced_embedding_model import EnhancedEmbeddingModel
        
        # Create model instance with the loaded training encoders/scalers
        enhanced_model = EnhancedEmbeddingModel()
        enhanced_model.encoders = self.training_encoders
        enhanced_model.scalers = self.training_scalers
        
        # Categorize features exactly like during training
        feature_categories = enhanced_model.categorize_features_for_embeddings(df_2023, features)
        
        # Ensure all required features exist
        missing_features = [f for f in features if f not in df_2023.columns]
        if missing_features:
            self.logger.warning(f"    Missing {len(missing_features)} features, filling with 0")
            for feature in missing_features:
                df_2023[feature] = 0
        
        # Use the SAME preprocessing as training (is_training=False uses saved encoders/scalers)
        try:
            prepared_data, _ = enhanced_model.prepare_embedding_features(
                df_2023, feature_categories, is_training=False
            )
            
            # Convert to the expected input format for the model
            prepared_inputs = []
            input_order = ['temporal', 'continuous', 'direct']
            
            for key in input_order:
                if key in prepared_data:
                    prepared_inputs.append(prepared_data[key])
            
            self.logger.info(f"    ✓ Prepared {len(prepared_inputs)} inputs using TRAINING preprocessing")
            
            # Log preprocessing details for verification
            for i, input_data in enumerate(prepared_inputs):
                max_val = np.max(input_data)
                min_val = np.min(input_data)
                self.logger.info(f"    Input {i}: shape={input_data.shape}, range=[{min_val:.2f}, {max_val:.2f}]")
            
            return prepared_inputs
            
        except Exception as e:
            self.logger.error(f"    ❌ Training preprocessing failed: {str(e)}")
            raise e
    
    def evaluate_single_model(self, model_path: str, split_num: int):
        """Evaluate a single model on 2023 data with proper preprocessing."""
        self.logger.info(f"\n--- Evaluating Split {split_num} ---")
        self.logger.info(f"Model: {Path(model_path).name}")
        
        # Load the model
        model = self.load_tensorflow_model_properly(model_path)
        if model is None:
            self.logger.error(f"  ❌ Failed to load model for Split {split_num}")
            return None
        
        try:
            # Log model info
            self.logger.info(f"    ✅ Model loaded: {model.count_params():,} parameters")
            
            # FIXED: Load training encoders/scalers for this split
            if not self.load_training_encoders_scalers(split_num):
                if not self.reconstruct_training_preprocessing(split_num):
                    self.logger.error(f"    ❌ Failed to load training preprocessing for Split {split_num}")
                    return None
            
            # FIXED: Prepare test data using training preprocessing
            prepared_inputs = self.prepare_features_with_training_preprocessing(self.df_2023, self.modeling_features)
            
            # Prepare target variable
            if 'sales_quantity_log' in self.df_2023.columns:
                y_test = self.df_2023['sales_quantity_log'].values
            else:
                # Create log-transformed target if it doesn't exist
                y_test = np.log1p(self.df_2023['sales_quantity'].values)
                self.logger.info("    Created log-transformed target from sales_quantity")
            
            # Make predictions
            self.logger.info("    Making predictions...")
            y_pred_log = model.predict(prepared_inputs, verbose=0)
            
            # Convert back to original scale
            y_pred_orig = np.exp(y_pred_log.flatten()) - 1
            y_true_orig = np.exp(y_test) - 1
            
            # Ensure positive values
            y_pred_orig = np.maximum(y_pred_orig, 1.0)
            y_true_orig = np.maximum(y_true_orig, 1.0)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            r2 = r2_score(y_true_orig, y_pred_orig)
            mae = np.mean(np.abs(y_true_orig - y_pred_orig))
            
            results = {
                'split_num': split_num,
                'split_description': self.split_meanings.get(split_num, f'Split {split_num}'),
                'model_path': model_path,
                'mape': mape,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'predictions': y_pred_orig,
                'actuals': y_true_orig,
                'test_samples': len(y_true_orig)
            }
            
            self.logger.info(f"  📊 Results:")
            self.logger.info(f"    MAPE: {mape:.2f}%")
            self.logger.info(f"    RMSE: {rmse:.0f}")
            self.logger.info(f"    R²: {r2:.3f}")
            self.logger.info(f"    Samples: {len(y_true_orig):,}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"  ❌ Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up model to free memory
            del model
            tf.keras.backend.clear_session()
    
    def run_complete_evaluation(self, engineered_dataset_path: str):
        """Run complete evaluation on all models with fixed preprocessing."""
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: ENHANCED MODEL EVALUATION ON 2023 DATA")
        self.logger.info("=" * 80)
        
        # Load 2023 data
        df_2023, modeling_features = self.load_engineered_2023_data(engineered_dataset_path)
        
        # Find available models
        best_models = self.find_best_models()
        
        # Evaluate each model
        all_results = {}
        
        for split_num, model_path in best_models.items():
            result = self.evaluate_single_model(model_path, split_num)
            if result:
                all_results[split_num] = result
        
        if not all_results:
            self.logger.error("❌ No models could be evaluated successfully!")
            return None
        
        # Calculate summary statistics
        mapes = [r['mape'] for r in all_results.values()]
        best_mape = min(mapes)
        worst_mape = max(mapes)
        avg_mape = np.mean(mapes)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Successfully evaluated: {len(all_results)} models")
        self.logger.info(f"📊 Performance Summary:")
        self.logger.info(f"  Best MAPE: {best_mape:.2f}%")
        self.logger.info(f"  Worst MAPE: {worst_mape:.2f}%")
        self.logger.info(f"  Average MAPE: {avg_mape:.2f}%")
        
        # Performance assessment
        if avg_mape <= 10:
            grade = "EXCELLENT"
            assessment = "🎉 Outstanding performance! Models are production-ready."
        elif avg_mape <= 20:
            grade = "GOOD"
            assessment = "✅ Good performance! Models are business-usable."
        elif avg_mape <= 30:
            grade = "FAIR"
            assessment = "⚠️ Fair performance. Consider improvements."
        else:
            grade = "POOR"
            assessment = "❌ Poor performance. Significant improvements needed."
        
        self.logger.info(f"🏆 Overall Grade: {grade}")
        self.logger.info(f"💡 Assessment: {assessment}")
        
        # Save detailed results
        self.save_evaluation_results(all_results)
        
        # Create clean visualizations
        self.create_visualizations(all_results)
        
        return {
            'results': all_results,
            'summary': {
                'best_mape': best_mape,
                'worst_mape': worst_mape,
                'average_mape': avg_mape,
                'grade': grade,
                'models_evaluated': len(all_results)
            }
        }
    
    def save_evaluation_results(self, all_results):
        """Save evaluation results to files."""
        self.logger.info("💾 Saving evaluation results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"2023_evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for split_num, result in all_results.items():
            json_result = result.copy()
            json_result['predictions'] = result['predictions'].tolist()
            json_result['actuals'] = result['actuals'].tolist()
            json_results[str(split_num)] = json_result
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"  ✅ Detailed results: {results_file}")
        
        # Save summary CSV
        summary_data = []
        for split_num, result in all_results.items():
            summary_data.append({
                'split_number': split_num,
                'split_description': result['split_description'],
                'mape_percent': result['mape'],
                'rmse': result['rmse'],
                'r2_score': result['r2'],
                'mae': result['mae'],
                'test_samples': result['test_samples'],
                'model_file': Path(result['model_path']).name
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"2023_evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"  ✅ Summary CSV: {summary_file}")

    def create_visualizations(self, all_results):
        """Create clean visualization plots for the fixed evaluation results."""
        self.logger.info("📊 Creating clean visualizations...")
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Enhanced Model Performance on 2023 Data', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            split_nums = list(all_results.keys())
            mapes = [all_results[s]['mape'] for s in split_nums]
            rmses = [all_results[s]['rmse'] for s in split_nums]
            r2s = [all_results[s]['r2'] for s in split_nums]
            
            # Plot 1: MAPE by Split
            bars1 = axes[0,0].bar(split_nums, mapes, color='skyblue', alpha=0.7)
            axes[0,0].set_title('MAPE by Model Split', fontweight='bold')
            axes[0,0].set_xlabel('Split Number')
            axes[0,0].set_ylabel('MAPE (%)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(mapes):
                axes[0,0].text(split_nums[i], v + max(mapes)*0.01, f'{v:.1f}%', 
                              ha='center', va='bottom', fontweight='bold')
            
            # Add horizontal line for 20% MAPE target
            axes[0,0].axhline(y=20, color='red', linestyle='--', alpha=0.7, 
                             label='Business Target (20%)')
            axes[0,0].legend()
            
            # Plot 2: RMSE by Split
            axes[0,1].bar(split_nums, rmses, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('RMSE by Model Split', fontweight='bold')
            axes[0,1].set_xlabel('Split Number')
            axes[0,1].set_ylabel('RMSE')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: R² Score by Split
            bars3 = axes[1,0].bar(split_nums, r2s, color='lightgreen', alpha=0.7)
            axes[1,0].set_title('R² Score by Model Split', fontweight='bold')
            axes[1,0].set_xlabel('Split Number')
            axes[1,0].set_ylabel('R² Score')
            axes[1,0].grid(True, alpha=0.3)
            
            # Color bars based on R² score quality
            for i, (bar, r2_val) in enumerate(zip(bars3, r2s)):
                if r2_val > 0.1:
                    bar.set_color('green')
                elif r2_val > 0.0:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Plot 4: Actual vs Predicted for best model
            best_split = min(all_results.keys(), key=lambda x: all_results[x]['mape'])
            best_result = all_results[best_split]
            
            # Sample data for plotting (to avoid overcrowding)
            n_samples = len(best_result['predictions'])
            sample_size = min(n_samples, 1000)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            
            actual_sample = best_result['actuals'][indices]
            pred_sample = best_result['predictions'][indices]
            
            axes[1,1].scatter(actual_sample, pred_sample, alpha=0.6, s=20, color='blue')
            
            # Perfect prediction line
            min_val = min(actual_sample.min(), pred_sample.min())
            max_val = max(actual_sample.max(), pred_sample.max())
            axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                          linewidth=2, label='Perfect Prediction')
            
            axes[1,1].set_title(f'Actual vs Predicted (Best Model - Split {best_split})\nMAPE: {best_result["mape"]:.1f}%', 
                               fontweight='bold')
            axes[1,1].set_xlabel('Actual Sales')
            axes[1,1].set_ylabel('Predicted Sales')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"2023_evaluation_clean_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  ✅ Clean visualization: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"  ⚠️ Could not create visualizations: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test enhanced models on 2023 data")
    
    parser.add_argument("--engineered-dataset", 
                       type=str, 
                       required=True,
                       help="Path to engineered dataset pickle file")
    
    parser.add_argument("--models-dir",
                       type=str,
                       help="Directory containing trained models (required if not using --experiment-name)")
    
    parser.add_argument("--experiment-name",
                       type=str,
                       help="Experiment name to automatically find models (alternative to --models-dir)")
    
    parser.add_argument("--log-level",
                       type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    parser.add_argument("--model-type",
                       type=str,
                       choices=["vanilla", "enhanced", "both"],
                       default="enhanced",
                       help="Model type to evaluate: vanilla, enhanced, or both")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.models_dir and not args.experiment_name:
        print("Error: Either --models-dir or --experiment-name must be provided")
        return 1
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    # Print header
    logger.info("=" * 80)
    logger.info("PHASE 3: MODEL EVALUATION ON 2023 DATA")
    logger.info("=" * 80)
    logger.info(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.engineered_dataset}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Determine models directory
        models_dirs = []
        
        if args.models_dir:
            models_dirs.append(("models", args.models_dir))
        elif args.experiment_name:
            # Auto-find model directories based on experiment name and model type
            base_dir = Path("outputs") / args.experiment_name
            
            if args.model_type in ["vanilla", "both"]:
                vanilla_dir = base_dir / "vanilla_models"
                if vanilla_dir.exists():
                    models_dirs.append(("vanilla", str(vanilla_dir)))
            
            if args.model_type in ["enhanced", "both"]:
                enhanced_dir = base_dir / "enhanced_models"
                if enhanced_dir.exists():
                    models_dirs.append(("enhanced", str(enhanced_dir)))
        
        if not models_dirs:
            logger.error("No model directories found!")
            return 1
        
        all_evaluation_results = {}
        
        # Evaluate each model type
        for model_type, models_dir in models_dirs:
            logger.info(f"\nEvaluating {model_type.title()} models from: {models_dir}")
            
            # Initialize evaluator
            evaluator = Model2023Evaluator(models_dir)
            
            # Run complete evaluation
            results = evaluator.run_complete_evaluation(args.engineered_dataset)
            
            if results:
                all_evaluation_results[model_type] = results
                
                # Model-specific summary
                summary = results['summary']
                logger.info(f"\n📊 {model_type.upper()} MODEL SUMMARY:")
                logger.info(f"  Models evaluated: {summary['models_evaluated']}")
                logger.info(f"  Best MAPE: {summary['best_mape']:.2f}%")
                logger.info(f"  Average MAPE: {summary['average_mape']:.2f}%")
                logger.info(f"  Performance grade: {summary['grade']}")
            else:
                logger.warning(f"⚠️ {model_type.title()} model evaluation failed")
        
        if all_evaluation_results:
            logger.info("\n🎉 Model evaluation completed successfully!")
            
            # Compare model types if multiple were evaluated
            if len(all_evaluation_results) > 1:
                logger.info("\n🏆 MODEL COMPARISON:")
                for model_type, results in all_evaluation_results.items():
                    summary = results['summary']
                    logger.info(f"  {model_type.title()}: {summary['average_mape']:.2f}% MAPE")
                
                # Determine best model
                best_model = min(all_evaluation_results.items(), 
                               key=lambda x: x[1]['summary']['average_mape'])
                logger.info(f"  🥇 Best performing: {best_model[0].title()} model")
            
            return 0
        else:
            logger.error("❌ All model evaluations failed")
            return 1
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 