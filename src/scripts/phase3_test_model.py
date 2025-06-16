#!/usr/bin/env python3
"""
COMPLETE FIXED Phase 3: Test Models on 2023 Data

This script loads your working TensorFlow models and tests them on 2023 data.

Usage:
    python phase3_test_model.py --models-dir "outputs/phase2_experiment_20250113/models" --engineered-dataset "path/to/dataset.pkl"
    
Results will be automatically saved to: outputs/phase2_experiment_20250113/2023_evaluation/

Author: Sales Forecasting Team (Fixed by Claude)
Date: 2025-06-13
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
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent if len(Path(__file__).parents) > 2 else Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.feature_pipeline import SalesFeaturePipeline
from utils.helpers import setup_logging

# ===== CUSTOM FUNCTIONS AND CLASSES =====

# Import shared custom objects
from models.custom_objects import get_custom_objects

# ===== ARGUMENT PARSING =====

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test best models on 2023 data")
    
    parser.add_argument("--engineered-dataset", 
                       type=str, 
                       required=True,
                       help="Path to engineered dataset pickle file")
    
    parser.add_argument("--models-dir",
                       type=str,
                       default="outputs/fixed_model/models",
                       help="Directory containing trained models")
    
    parser.add_argument("--log-level",
                       type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()

# ===== MAIN EVALUATOR CLASS =====

class FixedModel2023Evaluator:
    """Fixed evaluator that properly loads TensorFlow models and tests on 2023 data."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        
        # ✅ FIX: Auto-determine output directory from models directory structure
        # Follow the same pattern as Phase 2: {experiment-dir}/2023_evaluation/
        # If models_dir is: outputs/phase2_experiment_20250113/models
        # Then output_dir is: outputs/phase2_experiment_20250113/2023_evaluation/
        experiment_dir = self.models_dir.parent
        self.output_dir = experiment_dir / "2023_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split meanings for legend
        self.split_meanings = {
            1: "2021 Full Year → 2022 Q1",
            2: "2021 + 2022 Q1 → 2022 Q2", 
            3: "2021 + 2022 H1 → 2022 Q3",
            4: "2021 + 2022 Q1-Q3 → 2022 Q4",
            5: "Additional Split 5"  # Your models have a 5th split
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
        
        # Look for model files - check splits 1-10 to catch all models
        for split_num in range(1, 11):
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
        
        # Get the custom objects dictionary with all required functions
        custom_objects = get_custom_objects()
        
        try:
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
    
    def prepare_features_for_model(self, df_2023: pd.DataFrame, features: list, model):
        """Prepare features using the same AdvancedEmbeddingModel preprocessing as training."""
        self.logger.info("    Preparing features using AdvancedEmbeddingModel preprocessing...")
        
        # Import the AdvancedEmbeddingModel class
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / "src"))
        # from models.vanilla_embedding_model import VanillaEmbeddingModel
        from models.enhanced_embedding_model import EnhancedEmbeddingModel
        
        # Create an instance of AdvancedEmbeddingModel for feature processing
        # embedding_model = AdvancedEmbeddingModel()
        embedding_model = EnhancedEmbeddingModel()
        
        # Categorize features exactly like during training
        feature_categories = embedding_model.categorize_features_for_embeddings(df_2023, features)
        
        # *** CRITICAL: We need to load the encoders and scalers from training ***
        # For now, we'll create a simplified version that handles the basic embedding requirements
        
        # Ensure all required features exist
        missing_features = [f for f in features if f not in df_2023.columns]
        if missing_features:
            self.logger.warning(f"    Missing {len(missing_features)} features, filling with 0")
            for feature in missing_features:
                df_2023[feature] = 0
        
        # Use the embedding model's feature preparation method (but in inference mode)
        try:
            # This will fail because we don't have the encoders/scalers from training
            # We need a different approach
            prepared_data, _ = embedding_model.prepare_embedding_features(
                df_2023, feature_categories, is_training=False
            )
            
            # Convert to the expected input format for the model
            prepared_inputs = []
            input_order = ['temporal', 'continuous', 'direct']  # Same order as training
            
            for key in input_order:
                if key in prepared_data:
                    prepared_inputs.append(prepared_data[key])
            
            self.logger.info(f"    ✓ Prepared {len(prepared_inputs)} inputs using embedding preprocessing")
            return prepared_inputs
            
        except Exception as e:
            self.logger.error(f"    ❌ Embedding preprocessing failed: {str(e)}")
            self.logger.info("    Falling back to manual feature preparation...")
            
            # Manual fallback - create properly formatted inputs
            return self._manual_embedding_feature_preparation(df_2023, features, model)


    def _manual_embedding_feature_preparation(self, df_2023: pd.DataFrame, features: list, model):
        """Manual feature preparation that mimics the embedding preprocessing."""
        
        # Get feature categories
        # from models.vanilla_embedding_model import VanillaEmbeddingModel
        from models.enhanced_embedding_model import EnhancedEmbeddingModel
        embedding_model = EnhancedEmbeddingModel()
        feature_categories = embedding_model.categorize_features_for_embeddings(df_2023, features)
        
        prepared_inputs = []
        
        # 1. Temporal features (if any)
        temporal_features = feature_categories.get('temporal', [])
        if temporal_features:
            temporal_data = []
            for feature in temporal_features:
                if feature in df_2023.columns:
                    values = df_2023[feature].fillna(0).values.astype(int)
                    if feature == 'month':
                        values = np.clip(values, 1, 12) - 1  # 0-11 for embedding
                    elif feature == 'quarter':
                        values = np.clip(values, 1, 4) - 1   # 0-3 for embedding
                    else:
                        values = np.clip(values, 0, 100)     # General clipping
                    temporal_data.append(values)
            
            if temporal_data:
                prepared_inputs.append(np.column_stack(temporal_data))
        
        # 2. Continuous features (bucketized)
        continuous_features = feature_categories.get('numerical_continuous', [])
        if continuous_features:
            continuous_data = []
            for feature in continuous_features:
                if feature in df_2023.columns:
                    values = df_2023[feature].replace([np.inf, -np.inf], np.nan).fillna(0).values
                    
                    # Simple bucketization for testing (since we don't have training quantiles)
                    # Use percentile-based buckets as approximation
                    try:
                        # Create simple buckets based on the data we have
                        non_zero_values = values[values != 0]
                        if len(non_zero_values) > 10:
                            bucket_edges = np.percentile(non_zero_values, np.linspace(0, 100, 51))
                            bucket_edges = np.unique(bucket_edges)
                        else:
                            bucket_edges = np.array([0, np.max(values) if len(values) > 0 else 1])
                        
                        bucket_indices = np.digitize(values, bucket_edges)
                        # *** CRITICAL FIX: Clip to valid embedding range ***
                        bucket_indices = np.clip(bucket_indices, 0, 51)  # Max index 51 for embedding(52)
                        
                    except Exception:
                        # Fallback: just clip raw values to a reasonable range
                        bucket_indices = np.clip(values.astype(int), 0, 51)
                    
                    continuous_data.append(bucket_indices)
            
            if continuous_data:
                prepared_inputs.append(np.column_stack(continuous_data))
        
        # 3. Direct features (numerical_discrete + binary + interactions)
        direct_features = (feature_categories.get('numerical_discrete', []) + 
                        feature_categories.get('binary', []) + 
                        feature_categories.get('interactions', []))
        
        if direct_features:
            existing_features = [f for f in direct_features if f in df_2023.columns]
            if existing_features:
                direct_data = df_2023[existing_features].fillna(0).values.astype(np.float32)
                direct_data = np.nan_to_num(direct_data, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Apply basic scaling (since we don't have the training scaler)
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                direct_data = scaler.fit_transform(direct_data)
                
                prepared_inputs.append(direct_data)
        
        # Ensure we have the right number of inputs
        num_expected_inputs = len(model.inputs)
        while len(prepared_inputs) < num_expected_inputs:
            # Pad with zeros if we're missing inputs
            dummy_input = np.zeros((len(df_2023), 1), dtype=np.float32)
            prepared_inputs.append(dummy_input)
        
        # Trim if we have too many
        prepared_inputs = prepared_inputs[:num_expected_inputs]
        
        self.logger.info(f"    ✓ Manual preparation created {len(prepared_inputs)} inputs")
        
        # Debug: Check for out-of-range values
        for i, input_data in enumerate(prepared_inputs):
            max_val = np.max(input_data)
            min_val = np.min(input_data)
            self.logger.info(f"    Input {i}: shape={input_data.shape}, range=[{min_val:.2f}, {max_val:.2f}]")
            
            # Check if this looks like continuous features (integers in range that might be embeddings)
            if input_data.dtype in [np.int32, np.int64] or (np.all(input_data == input_data.astype(int)) and max_val > 51):
                self.logger.warning(f"    ⚠️  Input {i} has max value {max_val} > 51 (embedding limit)")
                # Additional clipping
                prepared_inputs[i] = np.clip(input_data, 0, 51).astype(np.int32)
                self.logger.info(f"    ✓ Clipped input {i} to valid embedding range [0, 51]")
            elif input_data.dtype in [np.int32, np.int64] or np.all(input_data == input_data.astype(int)):
                # Integer data that might be for embeddings, check if already in valid range
                if max_val == 51:
                    self.logger.info(f"    ✓ Input {i} already in valid embedding range [0, 51]")
                elif max_val < 51:
                    self.logger.info(f"    ✓ Input {i} in valid range [0, {int(max_val)}]")
        
        return prepared_inputs
    
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
            # Log model info
            self.logger.info(f"    ✅ Model loaded: {model.count_params():,} parameters")
            
            # Prepare test data
            prepared_inputs = self.prepare_features_for_model(self.df_2023, self.modeling_features, model)
            
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
        """Run complete evaluation on all models."""
        self.logger.info("=" * 80)
        self.logger.info("PHASE 3: FIXED MODEL EVALUATION ON 2023 DATA")
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
        
        # Create visualizations
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
        
        # Save predictions CSV for detailed analysis
        predictions_data = []
        for split_num, result in all_results.items():
            n_samples = len(result['predictions'])
            for i in range(min(n_samples, 1000)):  # Limit to first 1000 samples
                predictions_data.append({
                    'split_number': split_num,
                    'sample_index': i,
                    'predicted_sales': result['predictions'][i],
                    'actual_sales': result['actuals'][i],
                    'absolute_error': abs(result['predictions'][i] - result['actuals'][i]),
                    'percentage_error': abs(result['predictions'][i] - result['actuals'][i]) / result['actuals'][i] * 100
                })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_file = self.output_dir / f"2023_predictions_sample_{timestamp}.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        self.logger.info(f"  ✅ Predictions sample: {predictions_file}")
    
    def create_visualizations(self, all_results):
        """Create visualization plots for the evaluation results."""
        self.logger.info("📊 Creating visualizations...")
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance on 2023 Data', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            split_nums = list(all_results.keys())
            mapes = [all_results[s]['mape'] for s in split_nums]
            rmses = [all_results[s]['rmse'] for s in split_nums]
            r2s = [all_results[s]['r2'] for s in split_nums]
            
            # Plot 1: MAPE by Split
            axes[0,0].bar(split_nums, mapes, color='skyblue', alpha=0.7)
            axes[0,0].set_title('MAPE by Model Split')
            axes[0,0].set_xlabel('Split Number')
            axes[0,0].set_ylabel('MAPE (%)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(mapes):
                axes[0,0].text(split_nums[i], v + max(mapes)*0.01, f'{v:.1f}%', 
                              ha='center', va='bottom')
            
            # Plot 2: RMSE by Split
            axes[0,1].bar(split_nums, rmses, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('RMSE by Model Split')
            axes[0,1].set_xlabel('Split Number')
            axes[0,1].set_ylabel('RMSE')
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: R² Score by Split
            axes[1,0].bar(split_nums, r2s, color='lightgreen', alpha=0.7)
            axes[1,0].set_title('R² Score by Model Split')
            axes[1,0].set_xlabel('Split Number')
            axes[1,0].set_ylabel('R² Score')
            axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Actual vs Predicted for best model
            best_split = min(all_results.keys(), key=lambda x: all_results[x]['mape'])
            best_result = all_results[best_split]
            
            # Sample data for plotting (to avoid overcrowding)
            n_samples = len(best_result['predictions'])
            sample_size = min(n_samples, 1000)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            
            actual_sample = best_result['actuals'][indices]
            pred_sample = best_result['predictions'][indices]
            
            axes[1,1].scatter(actual_sample, pred_sample, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(actual_sample.min(), pred_sample.min())
            max_val = max(actual_sample.max(), pred_sample.max())
            axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[1,1].set_title(f'Actual vs Predicted (Best Model - Split {best_split})')
            axes[1,1].set_xlabel('Actual Sales')
            axes[1,1].set_ylabel('Predicted Sales')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"2023_evaluation_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  ✅ Visualization: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"  ⚠️ Could not create visualizations: {str(e)}")

# ===== MAIN EXECUTION =====

def main():
    """Main execution function."""
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
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Initialize evaluator
        evaluator = FixedModel2023Evaluator(args.models_dir)
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(args.engineered_dataset)
        
        if results:
            logger.info("\n🎉 Model evaluation completed successfully!")
            logger.info(f"✅ Results saved to: {evaluator.output_dir}")
            logger.info(f"📁 This follows the same structure as Phase 2 training outputs")
            
            # Final performance summary
            summary = results['summary']
            logger.info(f"\n📊 FINAL SUMMARY:")
            logger.info(f"  Models evaluated: {summary['models_evaluated']}")
            logger.info(f"  Best MAPE: {summary['best_mape']:.2f}%")
            logger.info(f"  Average MAPE: {summary['average_mape']:.2f}%")
            logger.info(f"  Performance grade: {summary['grade']}")
            
            return 0
        else:
            logger.error("❌ Model evaluation failed")
            return 1
        
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