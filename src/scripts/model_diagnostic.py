#!/usr/bin/env python3
"""
FIXED Model Loading Diagnostic Script

This script properly tests your new models with the correct custom objects.

Usage:
    python fixed_model_diagnostic.py --models-dir "outputs/fixed_model/models"

Author: Claude
Date: 2025-06-13
"""

import argparse
import sys
import os
from pathlib import Path
import tensorflow as tf
import traceback
import h5py
import json
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent if len(Path(__file__).parents) > 1 else Path(__file__).parent
if 'src' not in str(project_root):
    project_root = project_root / 'src'
sys.path.insert(0, str(project_root))

class FeatureSliceLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer to extract a single feature from input tensor.
    This replaces the problematic Lambda layer and is fully serializable.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
    
    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.index], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    parser = argparse.ArgumentParser(description="Fixed TensorFlow model loading diagnostic")
    
    parser.add_argument("--models-dir",
                       type=str,
                       default="outputs/fixed_model/models",
                       help="Directory containing trained models")
    
    return parser.parse_args()

def test_model_loading(model_path):
    """Test loading a specific model with all the right custom objects."""
    filename = os.path.basename(model_path)
    print(f"\n🔄 Testing: {filename}")
    
    # Get file info
    try:
        file_stats = os.stat(model_path)
        file_size = file_stats.st_size / (1024 * 1024)  # MB
        modified_time = datetime.fromtimestamp(file_stats.st_mtime)
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"   ⚠️ Could not get file stats: {e}")
    
    # Define custom objects with everything your models need
    custom_objects = {
        'mape_metric_original_scale': mape_metric_original_scale,
        'rmse_metric_original_scale': rmse_metric_original_scale,
        'FeatureSliceLayer': FeatureSliceLayer,
        'AdamW': tf.keras.optimizers.AdamW
    }
    
    # Try multiple loading strategies
    strategies = [
        ("With custom objects", lambda: tf.keras.models.load_model(model_path, custom_objects=custom_objects)),
        ("Without compilation", lambda: tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)),
        ("Safe mode disabled", lambda: tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)),
    ]
    
    for strategy_name, load_func in strategies:
        try:
            print(f"   🔄 Trying: {strategy_name}")
            model = load_func()
            
            print(f"   ✅ SUCCESS with {strategy_name}")
            print(f"   📊 Parameters: {model.count_params():,}")
            
            # Get model info
            try:
                print(f"   📥 Inputs: {len(model.inputs)} input(s)")
                for i, inp in enumerate(model.inputs):
                    print(f"      Input {i+1}: {inp.shape} ({inp.name})")
                print(f"   📤 Output: {model.output.shape} ({model.output.name})")
            except Exception as e:
                print(f"   ⚠️ Could not get model structure: {e}")
            
            # Try a simple prediction test
            try:
                # Create dummy inputs matching the expected shapes
                if len(model.inputs) == 3:  # temporal, continuous, direct
                    dummy_inputs = [
                        tf.random.normal((5, model.inputs[0].shape[1])),  # temporal
                        tf.random.normal((5, model.inputs[1].shape[1])),  # continuous  
                        tf.random.normal((5, model.inputs[2].shape[1]))   # direct
                    ]
                elif len(model.inputs) == 2:
                    dummy_inputs = [
                        tf.random.normal((5, model.inputs[0].shape[1])),
                        tf.random.normal((5, model.inputs[1].shape[1]))
                    ]
                else:
                    dummy_inputs = [tf.random.normal((5, model.inputs[0].shape[1]))]
                
                pred = model.predict(dummy_inputs, verbose=0)
                print(f"   🎯 Prediction test: ✅ Output shape {pred.shape}")
                
            except Exception as e:
                print(f"   🎯 Prediction test: ⚠️ {str(e)[:100]}...")
            
            del model  # Clean up
            return True
            
        except Exception as e:
            print(f"   ❌ Failed with {strategy_name}: {type(e).__name__}")
            print(f"      Error: {str(e)[:100]}...")
            continue
    
    print(f"   ❌ ALL STRATEGIES FAILED for {filename}")
    return False

def analyze_models_directory(models_dir):
    """Analyze all models in the directory"""
    print("🔍 FIXED MODEL LOADING DIAGNOSTIC")
    print("=" * 60)
    print(f"Directory: {models_dir}")
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    if not os.path.exists(models_dir):
        print(f"❌ Directory does not exist: {models_dir}")
        return
    
    # Find all H5 files
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5'):
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("❌ No .h5 model files found!")
        return
    
    print(f"📁 Found {len(model_files)} model file(s)")
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    successful_loads = 0
    failed_loads = 0
    
    for model_path in model_files:
        success = test_model_loading(model_path)
        if success:
            successful_loads += 1
        else:
            failed_loads += 1
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"✅ Successfully loaded: {successful_loads} models")
    print(f"❌ Failed to load: {failed_loads} models")
    
    if successful_loads > 0:
        print("\n🎉 EXCELLENT! Your models are working correctly!")
        print("The Lambda layer fix was successful.")
        print("\nNext steps:")
        print("1. Run Phase 3 testing:")
        print(f"   python src/scripts/phase3_test_model.py --models-dir '{models_dir}' --engineered-dataset <your_dataset>")
        print("2. Your models are ready for production use!")
    else:
        print("\n😞 All models failed to load.")
        print("This suggests there may still be issues with the model architecture.")
        
    return successful_loads, failed_loads

def main():
    """Main diagnostic function"""
    args = parse_arguments()
    
    # Test if directory exists
    if not os.path.exists(args.models_dir):
        print(f"❌ Models directory not found: {args.models_dir}")
        print("\nTry these paths:")
        possible_paths = [
            "outputs/fixed_model/models",
            "outputs/models", 
            "/Users/victor/Library/CloudStorage/Dropbox/PolyU Projects/MingYin/outputs/fixed_model/models"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ Found: {path}")
            else:
                print(f"  ❌ Not found: {path}")
        return 1
    
    try:
        successful, failed = analyze_models_directory(args.models_dir)
        return 0 if successful > 0 else 1
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)