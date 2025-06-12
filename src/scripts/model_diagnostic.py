#!/usr/bin/env python3
"""
Model Loading Diagnostic Script

This script diagnoses the exact issue with loading TensorFlow models
and provides detailed error information to fix the problem.

Usage:
    python model_diagnostic.py --models-dir outputs/models

Author: Sales Forecasting Team
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path
import tensorflow as tf
import traceback
import h5py
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent if len(Path(__file__).parents) > 2 else Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import the exact custom functions
def mape_metric_original_scale(y_true, y_pred):
    """MAPE metric in original scale for monitoring during training"""
    y_true_orig = tf.exp(y_true) - 1
    y_pred_orig = tf.exp(y_pred) - 1
    y_true_orig = tf.clip_by_value(y_true_orig, 1.0, 1e6)
    y_pred_orig = tf.clip_by_value(y_pred_orig, 1.0, 1e6)
    epsilon = 1.0
    mape = tf.reduce_mean(tf.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)) * 100
    return tf.clip_by_value(mape, 0.0, 1000.0)

def rmse_metric_original_scale(y_true, y_pred):
    """RMSE in original scale for monitoring during training"""
    y_true_orig = tf.exp(y_true) - 1
    y_pred_orig = tf.exp(y_pred) - 1
    y_true_orig = tf.clip_by_value(y_true_orig, 1.0, 1e6)
    y_pred_orig = tf.clip_by_value(y_pred_orig, 1.0, 1e6)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true_orig - y_pred_orig)))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Diagnose TensorFlow model loading issues")
    
    parser.add_argument("--models-dir",
                       type=str,
                       default="outputs/models",
                       help="Directory containing trained models")
    
    return parser.parse_args()

def inspect_h5_file(model_path):
    """Inspect the structure of an HDF5 model file."""
    print(f"\n🔍 INSPECTING H5 FILE STRUCTURE: {Path(model_path).name}")
    print("-" * 60)
    
    try:
        with h5py.File(model_path, 'r') as f:
            print(f"HDF5 file keys: {list(f.keys())}")
            
            if 'model_config' in f.attrs:
                print("Found 'model_config' attribute")
                try:
                    config = json.loads(f.attrs['model_config'])
                    print(f"Model class: {config.get('class_name', 'Unknown')}")
                    print(f"Model config keys: {list(config.get('config', {}).keys())}")
                except Exception as e:
                    print(f"Could not parse model_config: {e}")
            
            if 'model_weights' in f:
                print("Found 'model_weights' group")
                weights_group = f['model_weights']
                print(f"Weight group keys: {list(weights_group.keys())}")
                
                # Check if it has layer names
                if hasattr(weights_group, 'attrs') and 'layer_names' in weights_group.attrs:
                    layer_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                 for name in weights_group.attrs['layer_names']]
                    print(f"Layer names: {layer_names[:5]}{'...' if len(layer_names) > 5 else ''}")
            
            # Check for training config
            if 'training_config' in f.attrs:
                print("Found 'training_config' attribute")
                try:
                    training_config = json.loads(f.attrs['training_config'])
                    print(f"Optimizer: {training_config.get('optimizer_config', {}).get('class_name', 'Unknown')}")
                    print(f"Loss: {training_config.get('loss', 'Unknown')}")
                    if 'metrics' in training_config:
                        metrics = training_config['metrics']
                        print(f"Metrics: {metrics}")
                except Exception as e:
                    print(f"Could not parse training_config: {e}")
                    
    except Exception as e:
        print(f"❌ Could not inspect H5 file: {e}")

def test_loading_strategies(model_path):
    """Test various loading strategies and report detailed errors."""
    print(f"\n🧪 TESTING LOADING STRATEGIES: {Path(model_path).name}")
    print("-" * 60)
    
    # Strategy 1: Standard loading
    print("Strategy 1: Standard loading")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ SUCCESS: Standard loading worked")
        return model
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    # Strategy 2: Loading without compilation
    print("\nStrategy 2: Loading without compilation")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ SUCCESS: Loading without compilation worked")
        return model
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    # Strategy 3: Loading with custom objects
    print("\nStrategy 3: Loading with custom objects")
    try:
        custom_objects = {
            'mape_metric_original_scale': mape_metric_original_scale,
            'rmse_metric_original_scale': rmse_metric_original_scale,
            'AdamW': tf.keras.optimizers.AdamW,
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ SUCCESS: Loading with custom objects worked")
        return model
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    # Strategy 4: Detailed error analysis
    print("\nStrategy 4: Detailed error analysis with full traceback")
    try:
        custom_objects = {
            'mape_metric_original_scale': mape_metric_original_scale,
            'rmse_metric_original_scale': rmse_metric_original_scale,
            'AdamW': tf.keras.optimizers.AdamW,
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam,
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ SUCCESS: Detailed loading worked")
        return model
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
    
    # Strategy 5: Try different TensorFlow settings
    print("\nStrategy 5: Trying with different TensorFlow settings")
    try:
        # Disable some optimizations
        tf.config.optimizer.set_jit(False)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        print("✅ SUCCESS: Modified TensorFlow settings worked")
        return model
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    return None

def get_tensorflow_info():
    """Get detailed TensorFlow environment information."""
    print("\n🔧 TENSORFLOW ENVIRONMENT INFO")
    print("-" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    
    # Check if eager execution is enabled
    print(f"Eager execution: {tf.executing_eagerly()}")
    
    # Check available optimizers
    try:
        optimizer_test = tf.keras.optimizers.AdamW(learning_rate=0.001)
        print("✅ AdamW optimizer available")
    except Exception as e:
        print(f"❌ AdamW optimizer issue: {e}")

def main():
    """Main diagnostic function."""
    args = parse_arguments()
    
    print("=" * 80)
    print("TENSORFLOW MODEL LOADING DIAGNOSTIC")
    print("=" * 80)
    
    # Get TensorFlow info
    get_tensorflow_info()
    
    # Find model files
    models_dir = Path(args.models_dir)
    model_patterns = [
        "best_model_split_*.h5",
        "advanced_embedding_model_split_*.h5",
        "model_split_*.h5",
        "*.h5"
    ]
    
    model_files = []
    for pattern in model_patterns:
        found_files = list(models_dir.glob(pattern))
        model_files.extend(found_files)
    
    if not model_files:
        print(f"\n❌ No model files found in {models_dir}")
        print("Available files:")
        for file in models_dir.iterdir():
            print(f"  {file.name}")
        return 1
    
    # Remove duplicates and sort
    model_files = sorted(list(set(model_files)), key=lambda x: x.name)
    
    print(f"\n📁 FOUND {len(model_files)} MODEL FILES")
    print("-" * 60)
    for file in model_files:
        print(f"  {file.name}")
    
    # Test each model
    successful_models = []
    failed_models = []
    
    for model_file in model_files[:4]:  # Test first 4 files
        print(f"\n" + "=" * 80)
        print(f"TESTING MODEL: {model_file.name}")
        print("=" * 80)
        
        # Inspect file structure
        inspect_h5_file(str(model_file))
        
        # Test loading strategies
        model = test_loading_strategies(str(model_file))
        
        if model is not None:
            successful_models.append(model_file)
            print(f"\n✅ MODEL LOADED SUCCESSFULLY")
            try:
                print(f"  Model input shape: {model.input_shape}")
                print(f"  Model output shape: {model.output_shape}")
                print(f"  Number of parameters: {model.count_params():,}")
                print(f"  Number of layers: {len(model.layers)}")
            except Exception as e:
                print(f"  Could not get model details: {e}")
        else:
            failed_models.append(model_file)
            print(f"\n❌ ALL LOADING STRATEGIES FAILED")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully loaded: {len(successful_models)} models")
    for model in successful_models:
        print(f"  - {model.name}")
    
    print(f"\n❌ Failed to load: {len(failed_models)} models")  
    for model in failed_models:
        print(f"  - {model.name}")
    
    if failed_models:
        print(f"\n🔧 RECOMMENDED FIXES:")
        print("1. Check TensorFlow version compatibility")
        print("2. Try recreating models with current TensorFlow version")
        print("3. Consider saving models in SavedModel format instead of H5")
        print("4. Check if custom functions are properly defined")
        print("5. Try loading on the same system/environment where models were saved")
    
    if successful_models:
        print(f"\n🎉 GOOD NEWS: Some models can be loaded!")
        print("Use the successful loading strategy for your evaluation script.")
    
    return 0 if successful_models else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)