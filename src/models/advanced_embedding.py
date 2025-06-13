"""
Advanced embedding-based deep learning model for sales forecasting.

This module implements the sophisticated neural network architecture
from full_data_prediction.ipynb with proper feature processing and
multi-input embedding strategies.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# FORCE TensorFlow imports first
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Import data loading
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

class AdvancedEmbeddingModel:
    """Fixed TensorFlow implementation matching your notebook exactly"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.model = None
        self.scalers = {}
        self.encoders = {}
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        print("✓ FixedAdvancedEmbeddingModel initialized with TensorFlow")
    
    def categorize_features_for_embeddings(self, df, feature_columns):
        """Categorize features exactly like notebook"""
        feature_categories = {
            'temporal': [],
            'numerical_continuous': [],
            'numerical_discrete': [],
            'binary': [],
            'interactions': []
        }
        
        for feature in feature_columns:
            if feature not in df.columns:
                continue
                
            if any(temporal in feature.lower() for temporal in ['month', 'quarter', 'year', 'day', 'week']):
                feature_categories['temporal'].append(feature)
            elif feature.startswith('lag_') or feature.startswith('rolling_') or feature.startswith('yoy_'):
                feature_categories['numerical_continuous'].append(feature)
            elif df[feature].dtype == bool or df[feature].nunique() == 2:
                feature_categories['binary'].append(feature)
            elif '_x_' in feature or feature.startswith('interaction_'):
                feature_categories['interactions'].append(feature)
            elif df[feature].dtype in ['int64', 'int32'] and df[feature].nunique() < 50:
                feature_categories['numerical_discrete'].append(feature)
            else:
                feature_categories['numerical_continuous'].append(feature)
        
        return feature_categories
    
    def prepare_embedding_features(self, df_work, feature_categories, is_training=True):
        """Prepare features exactly like notebook"""
        prepared_data = {}
        
        # Temporal features
        temporal_features = feature_categories['temporal']
        if temporal_features:
            temporal_data = []
            for feature in temporal_features:
                if feature in df_work.columns:
                    values = df_work[feature].fillna(0).values.astype(int)
                    if feature == 'month':
                        values = np.clip(values, 1, 12) - 1  # 0-11 for embedding
                    elif feature == 'quarter':
                        values = np.clip(values, 1, 4) - 1   # 0-3 for embedding
                    else:
                        values = np.clip(values, 0, 100)     # General clipping
                    temporal_data.append(values)
            
            if temporal_data:
                prepared_data['temporal'] = np.column_stack(temporal_data)
        
        # Numerical continuous - bucketize and embed
        continuous_features = feature_categories['numerical_continuous']
        if continuous_features:
            continuous_data = []
            for feature in continuous_features:
                if feature in df_work.columns:
                    values = df_work[feature].replace([np.inf, -np.inf], np.nan).fillna(0).values
                    
                    if is_training:
                        # Create quantile-based buckets
                        try:
                            buckets = np.quantile(values[values != 0], np.linspace(0, 1, 51))  # 50 buckets
                            buckets = np.unique(buckets)
                            self.encoders[f'{feature}_buckets'] = buckets
                        except:
                            self.encoders[f'{feature}_buckets'] = np.array([0, 1])
                    
                    bucket_edges = self.encoders.get(f'{feature}_buckets', np.array([0, 1]))
                    bucket_indices = np.digitize(values, bucket_edges)
                    bucket_indices = np.clip(bucket_indices, 0, len(bucket_edges))
                    continuous_data.append(bucket_indices)
            
            if continuous_data:
                prepared_data['continuous'] = np.column_stack(continuous_data)
        
        # Direct numerical features
        direct_features = (feature_categories['numerical_discrete'] + 
                          feature_categories['binary'] + 
                          feature_categories['interactions'])
        
        if direct_features:
            existing_features = [f for f in direct_features if f in df_work.columns]
            if existing_features:
                direct_data = df_work[existing_features].values.astype(np.float32)
                direct_data = np.nan_to_num(direct_data, nan=0.0, posinf=1e6, neginf=-1e6)
                
                if is_training:
                    self.scalers['direct'] = RobustScaler()
                    direct_data = self.scalers['direct'].fit_transform(direct_data)
                else:
                    direct_data = self.scalers['direct'].transform(direct_data)
                
                prepared_data['direct'] = direct_data
        
        # Target
        target = df_work['sales_quantity_log'].values.astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-1.0)
        
        return prepared_data, target
    
    def create_advanced_embedding_model(self, feature_categories, data_shapes):
        """Create TensorFlow model exactly like notebook"""
        print("\n=== CREATING TENSORFLOW ADVANCED EMBEDDING MODEL ===")
        
        inputs = {}
        embedding_outputs = []
        total_embedding_dim = 0
        
        # Temporal embeddings
        if 'temporal' in data_shapes:
            temporal_input = layers.Input(shape=(data_shapes['temporal'],), name='temporal_input')
            inputs['temporal'] = temporal_input
            
            # Process each temporal feature with specific embeddings
            temporal_embeddings = []
            for i in range(data_shapes['temporal']):
                # Extract single temporal feature using custom layer
                single_temporal = FeatureSliceLayer(i, name=f'temporal_slice_{i}')(temporal_input)
                
                if i == 0:  # Month
                    emb = layers.Embedding(12, 8, name=f'month_embedding')(single_temporal)
                    emb_dim = 8
                elif i == 1:  # Quarter  
                    emb = layers.Embedding(4, 4, name=f'quarter_embedding')(single_temporal)
                    emb_dim = 4
                else:
                    emb = layers.Embedding(101, 8, name=f'temporal_{i}_embedding')(single_temporal)
                    emb_dim = 8
                
                emb_flat = layers.Flatten()(emb)
                temporal_embeddings.append(emb_flat)
                total_embedding_dim += emb_dim
            
            if len(temporal_embeddings) > 1:
                temporal_combined = layers.Concatenate(name='temporal_combined')(temporal_embeddings)
            else:
                temporal_combined = temporal_embeddings[0]
            
            embedding_outputs.append(temporal_combined)
            print(f"  Temporal embeddings: {len(temporal_embeddings)} features, total dim: {sum([8 if i==0 else 4 if i==1 else 8 for i in range(data_shapes['temporal'])])}")
        
        # Continuous feature embeddings
        if 'continuous' in data_shapes:
            continuous_input = layers.Input(shape=(data_shapes['continuous'],), name='continuous_input')
            inputs['continuous'] = continuous_input
            
            # Process each continuous feature with smaller embeddings
            continuous_embeddings = []
            embedding_dim_per_feature = 8  # Smaller dimension
            
            for i in range(data_shapes['continuous']):
                # Extract single continuous feature using custom layer
                single_continuous = FeatureSliceLayer(i, name=f'continuous_slice_{i}')(continuous_input)
                
                emb = layers.Embedding(52, embedding_dim_per_feature, name=f'continuous_{i}_embedding')(single_continuous)
                emb_flat = layers.Flatten()(emb)
                continuous_embeddings.append(emb_flat)
                total_embedding_dim += embedding_dim_per_feature
            
            if len(continuous_embeddings) > 1:
                continuous_combined = layers.Concatenate(name='continuous_combined')(continuous_embeddings)
            else:
                continuous_combined = continuous_embeddings[0]
            
            embedding_outputs.append(continuous_combined)
            print(f"  Continuous embeddings: {len(continuous_embeddings)} features, total dim: {len(continuous_embeddings) * embedding_dim_per_feature}")
        
        # Direct numerical features
        direct_dim = 0
        if 'direct' in data_shapes:
            direct_input = layers.Input(shape=(data_shapes['direct'],), name='direct_input')
            inputs['direct'] = direct_input
            
            # Process direct features to fixed dimension
            direct_processed = layers.Dense(32, activation='relu', name='direct_dense')(direct_input)
            direct_processed = layers.BatchNormalization(name='direct_bn')(direct_processed)
            direct_processed = layers.Dropout(0.2, name='direct_dropout')(direct_processed)
            
            embedding_outputs.append(direct_processed)
            direct_dim = 32
            total_embedding_dim += direct_dim
            print(f"  Direct features: {data_shapes['direct']} features -> {direct_dim} dimensions")
        
        print(f"  Total embedding dimension: {total_embedding_dim}")
        
        # Combine all embeddings
        if len(embedding_outputs) > 1:
            combined = layers.Concatenate(name='combined_embeddings')(embedding_outputs)
        else:
            combined = embedding_outputs[0]
        
        # Deep network with skip connections (exactly like notebook)
        x = layers.Dense(256, activation='relu', name='deep_1')(combined)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        # First skip connection
        skip_connection_1 = layers.Dense(128, name='skip_1_projection')(combined)
        
        x = layers.Dense(128, activation='relu', name='deep_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        
        x = layers.Add(name='skip_connection_1')([x, skip_connection_1])
        
        # Continue deep network
        skip_connection_2 = x
        
        x = layers.Dense(64, activation='relu', name='deep_3')(x)
        x = layers.Dropout(0.1, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='deep_4')(x)
        x = layers.Dropout(0.1, name='dropout_4')(x)
        
        # Final skip connection
        skip_processed_2 = layers.Dense(32, name='skip_2_projection')(skip_connection_2)
        x = layers.Add(name='skip_connection_2')([x, skip_processed_2])
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='sales_prediction')(x)
        
        # Create model
        model = Model(inputs=list(inputs.values()), outputs=output, name='AdvancedEmbeddingModel')
        
        # Compile with advanced optimizer (EXACTLY like notebook)
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mae',  # MAE loss like notebook
            metrics=[mape_metric_original_scale, rmse_metric_original_scale]
        )
        
        print(f"Model created with {model.count_params():,} parameters")
        print("Model compilation completed")
        
        return model
    
    def save_predictions(self, training_results, df_final, rolling_splits, output_dir):
        """Save predictions from all training splits"""
        import os
        from datetime import datetime
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving predictions to: {output_dir}")
        
        for split_num, results in training_results.items():
            try:
                # Check if we have predictions in results
                if 'val_predictions' not in results or 'val_actuals' not in results:
                    print(f"⚠️ Split {split_num}: No predictions found in results")
                    continue
                
                # Get the validation data for this split
                if split_num <= len(rolling_splits):
                    train_df, val_df, description = rolling_splits[split_num - 1]
                    
                    # Create predictions DataFrame
                    predictions_df = val_df.copy()
                    predictions_df['predicted_sales'] = results['val_predictions']
                    predictions_df['actual_sales'] = results['val_actuals']
                    predictions_df['absolute_error'] = np.abs(results['val_predictions'] - results['val_actuals'])
                    predictions_df['percentage_error'] = (
                        np.abs(results['val_predictions'] - results['val_actuals']) / 
                        (results['val_actuals'] + 1.0) * 100
                    )
                    
                    # Add error categories
                    def categorize_error(ape):
                        if ape < 5: return "Excellent (<5%)"
                        elif ape < 10: return "Very Good (5-10%)"
                        elif ape < 20: return "Good (10-20%)"
                        elif ape < 50: return "Fair (20-50%)"
                        else: return "Poor (>50%)"
                    
                    predictions_df['error_category'] = predictions_df['percentage_error'].apply(categorize_error)
                    
                    # Save predictions
                    pred_filename = f"predictions_split_{split_num}_{timestamp}.csv"
                    pred_path = os.path.join(output_dir, pred_filename)
                    predictions_df.to_csv(pred_path, index=False)
                    
                    saved_files[f'predictions_split_{split_num}'] = pred_path
                    print(f"✅ Split {split_num} predictions saved: {pred_filename}")
                    
                else:
                    print(f"⚠️ Split {split_num}: No rolling split data found")
                    
            except Exception as e:
                print(f"❌ Split {split_num} predictions save failed: {str(e)}")
                continue
        
        # Save summary of all predictions
        try:
            summary_data = []
            for split_num, results in training_results.items():
                summary_data.append({
                    'split_number': split_num,
                    'description': results.get('description', 'N/A'),
                    'validation_mape': results.get('val_mape', 0),
                    'validation_rmse': results.get('val_rmse', 0),
                    'validation_r2': results.get('val_r2', 0),
                    'validation_samples': results.get('val_samples', 0),
                    'model_saved': 'Yes' if results.get('saved_model_path') else 'No'
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_filename = f"predictions_summary_{timestamp}.csv"
            summary_path = os.path.join(output_dir, summary_filename)
            summary_df.to_csv(summary_path, index=False)
            
            saved_files['predictions_summary'] = summary_path
            print(f"✅ Predictions summary saved: {summary_filename}")
            
        except Exception as e:
            print(f"❌ Predictions summary save failed: {str(e)}")
        
        return saved_files
    
    def safe_mape_calculation(self, y_true, y_pred):
        """Safe MAPE calculation"""
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        y_pred_orig = np.clip(y_pred_orig, 0.1, 1e6)
        y_true_orig = np.clip(y_true_orig, 0.1, 1e6)
        epsilon = 1.0
        ape = np.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)
        mape = np.mean(ape) * 100
        return min(mape, 1000.0)
    
    def train_on_rolling_splits(self, df_final, features, rolling_splits, epochs=100, batch_size=512, models_dir="outputs/models"):
        """Train on rolling splits with enhanced model saving and debugging"""
        print("=" * 80)
        print("TRAINING TENSORFLOW ADVANCED EMBEDDING MODEL ON ROLLING SPLITS")
        print("=" * 80)
        
        # Create models directory with better error handling
        import os
        from datetime import datetime
        
        try:
            os.makedirs(models_dir, exist_ok=True)
            print(f"📁 Models directory created/verified: {models_dir}")
            
            # Test write permissions
            test_file = os.path.join(models_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✅ Write permissions confirmed")
            
        except Exception as e:
            print(f"❌ Models directory setup failed: {str(e)}")
            # Fallback to current directory
            models_dir = "."
            print(f"📁 Using fallback directory: {models_dir}")
        
        # Analyze features
        feature_categories = self.categorize_features_for_embeddings(df_final, features)
        
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (train_df, val_df, description) in enumerate(rolling_splits, 1):
            print(f"\nSplit {i}/{len(rolling_splits)}: {description}")
            print(f"Train: {len(train_df):,} samples, Val: {len(val_df):,} samples")
            
            try:
                # Prepare data
                X_train, y_train = self.prepare_embedding_features(train_df, feature_categories, is_training=True)
                X_val, y_val = self.prepare_embedding_features(val_df, feature_categories, is_training=False)
                
                # Get data shapes for model creation
                if i == 1:  # First split - determine shapes
                    self.data_shapes = {}
                    for key, array in X_train.items():
                        self.data_shapes[key] = array.shape[1]
                    print(f"Data shapes determined: {self.data_shapes}")
                
                # Create fresh model for this split
                model = self.create_advanced_embedding_model(feature_categories, self.data_shapes)
                
                # Prepare inputs for training
                train_inputs = []
                val_inputs = []
                input_order = ['temporal', 'continuous', 'direct']  # Define consistent order
                
                for key in input_order:
                    if key in X_train:
                        train_inputs.append(X_train[key])
                        val_inputs.append(X_val[key])
                
                print(f"Prepared {len(train_inputs)} input tensors for training")
                
                # Training callbacks
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                
                early_stopping = EarlyStopping(
                    monitor='val_mape_metric_original_scale',
                    patience=20,
                    restore_best_weights=True,
                    mode='min',
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_mape_metric_original_scale',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    mode='min',
                    verbose=1
                )
                
                callbacks = [early_stopping, reduce_lr]
                
                # Train model
                print("🚀 Starting training...")
                history = model.fit(
                    train_inputs, y_train,
                    validation_data=(val_inputs, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
                print("✅ Training completed")
                
                # Make predictions for evaluation
                train_pred = model.predict(train_inputs, verbose=0)
                val_pred = model.predict(val_inputs, verbose=0)
                
                # Convert to original scale for metrics (with overflow protection)
                try:
                    # Use np.clip to prevent overflow in expm1
                    y_train_clipped = np.clip(y_train, -10, 10)  # Prevent extreme values
                    y_val_clipped = np.clip(y_val, -10, 10)
                    train_pred_clipped = np.clip(train_pred.flatten(), -10, 10)
                    val_pred_clipped = np.clip(val_pred.flatten(), -10, 10)
                    
                    train_true_orig = np.expm1(y_train_clipped)
                    train_pred_orig = np.expm1(train_pred_clipped)
                    val_true_orig = np.expm1(y_val_clipped)
                    val_pred_orig = np.expm1(val_pred_clipped)
                    
                    # Ensure positive values
                    train_true_orig = np.maximum(train_true_orig, 1.0)
                    train_pred_orig = np.maximum(train_pred_orig, 1.0)
                    val_true_orig = np.maximum(val_true_orig, 1.0)
                    val_pred_orig = np.maximum(val_pred_orig, 1.0)
                    
                except Exception as e:
                    print(f"⚠️ Overflow in scale conversion: {str(e)}")
                    # Fallback: use log scale values directly
                    train_true_orig = np.exp(y_train)
                    train_pred_orig = np.exp(train_pred.flatten())
                    val_true_orig = np.exp(y_val)
                    val_pred_orig = np.exp(val_pred.flatten())
                
                # Calculate metrics
                from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
                
                train_mape = mean_absolute_percentage_error(train_true_orig, train_pred_orig) * 100
                val_mape = mean_absolute_percentage_error(val_true_orig, val_pred_orig) * 100
                
                train_rmse = np.sqrt(mean_squared_error(train_true_orig, train_pred_orig))
                val_rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
                
                train_r2 = r2_score(train_true_orig, train_pred_orig)
                val_r2 = r2_score(val_true_orig, val_pred_orig)
                
                # *** ENHANCED MODEL SAVING WITH DEBUGGING ***
                model_filename = f"best_model_split_{i}_{timestamp}.h5"
                model_path = os.path.join(models_dir, model_filename)
                
                print(f"💾 Attempting to save model: {model_path}")
                
                saved_model_path = None
                save_error = None
                
                # Try multiple saving strategies
                strategies = [
                    ("H5_with_custom_objects", "h5"),
                    ("H5_basic", "h5"), 
                    ("SavedModel", "tf")
                ]
                
                for strategy_name, save_format in strategies:
                    try:
                        print(f"  Trying {strategy_name} format...")
                        
                        if save_format == "h5":
                            if strategy_name == "H5_with_custom_objects":
                                # Try with include_optimizer=False to avoid optimizer issues
                                model.save(model_path, save_format='h5', include_optimizer=False)
                            else:
                                # Basic H5 save
                                model.save(model_path)
                        else:
                            # SavedModel format
                            savedmodel_dir = model_path.replace('.h5', '_savedmodel')
                            model.save(savedmodel_dir, save_format='tf')
                            model_path = savedmodel_dir
                        
                        print(f"  ✅ {strategy_name} save successful")
                        
                        # Test loading immediately
                        if save_format == "h5":
                            # Test H5 loading
                            import tensorflow as tf
                            custom_objects = {
                                'mape_metric_original_scale': mape_metric_original_scale,
                                'rmse_metric_original_scale': rmse_metric_original_scale
                            }
                            
                            # Add FeatureSliceLayer if it exists
                            try:
                                custom_objects['FeatureSliceLayer'] = FeatureSliceLayer
                            except NameError:
                                pass  # FeatureSliceLayer not defined
                            
                            test_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                            print(f"  ✅ {strategy_name} load test passed")
                            del test_model
                        else:
                            # Test SavedModel loading
                            test_model = tf.keras.models.load_model(model_path, compile=False)
                            print(f"  ✅ {strategy_name} load test passed")
                            del test_model
                        
                        saved_model_path = model_path
                        break  # Success, exit loop
                        
                    except Exception as e:
                        print(f"  ❌ {strategy_name} failed: {str(e)}")
                        save_error = str(e)
                        continue
                
                if not saved_model_path:
                    print(f"❌ All save strategies failed. Last error: {save_error}")
                
                # Store comprehensive results
                results = {
                    'split_num': i,
                    'train_mape': train_mape,
                    'val_mape': val_mape,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'epochs_trained': len(history.history['loss']),
                    'best_epoch': np.argmin(history.history.get('val_mape_metric_original_scale', history.history.get('val_loss', []))) + 1,
                    'description': description,
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'saved_model_path': saved_model_path,
                    'model_filename': model_filename if saved_model_path else None,
                    'save_error': save_error if not saved_model_path else None,
                    'training_history': history.history,
                    'train_predictions': train_pred_orig,
                    'val_predictions': val_pred_orig,
                    'train_actuals': train_true_orig,
                    'val_actuals': val_true_orig
                }
                
                all_results[i] = results
                
                print(f"Split {i} Results:")
                print(f"  Train MAPE: {train_mape:.2f}%, Val MAPE: {val_mape:.2f}%")
                print(f"  Train RMSE: {train_rmse:.0f}, Val RMSE: {val_rmse:.0f}")
                print(f"  Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}")
                print(f"  Best epoch: {results['best_epoch']}/{results['epochs_trained']}")
                print(f"  Model saved: {'✅' if saved_model_path else '❌'}")
                
                if save_error and not saved_model_path:
                    print(f"  Save error: {save_error}")
                
            except Exception as e:
                print(f"❌ Split {i} training failed: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Store failed result
                all_results[i] = {
                    'split_num': i,
                    'train_mape': 999.0,
                    'val_mape': 999.0,
                    'train_rmse': 999999,
                    'val_rmse': 999999,
                    'train_r2': -999.0,
                    'val_r2': -999.0,
                    'epochs_trained': 0,
                    'best_epoch': 0,
                    'description': description,
                    'train_samples': 0,
                    'val_samples': 0,
                    'saved_model_path': None,
                    'model_filename': None,
                    'save_error': str(e),
                    'training_history': {},
                    'train_predictions': np.array([]),
                    'val_predictions': np.array([]),
                    'train_actuals': np.array([]),
                    'val_actuals': np.array([])
                }
                continue
            
            finally:
                # Clear GPU memory for next split
                import tensorflow as tf
                tf.keras.backend.clear_session()
                if 'model' in locals():
                    del model  # Clean up model reference
        
        # Print overall performance summary
        if all_results:
            successful_results = {k: v for k, v in all_results.items() if v['val_mape'] < 900}
            
            if successful_results:
                val_mapes = [results['val_mape'] for results in successful_results.values()]
                saved_models = [results['saved_model_path'] for results in successful_results.values() if results['saved_model_path']]
                
                print("\n" + "=" * 80)
                print("OVERALL PERFORMANCE SUMMARY")
                print("=" * 80)
                print(f"Successful splits: {len(successful_results)}/{len(all_results)}")
                print(f"Average Validation MAPE: {np.mean(val_mapes):.2f}% ± {np.std(val_mapes):.2f}%")
                print(f"Best Split MAPE: {np.min(val_mapes):.2f}%")
                print(f"Worst Split MAPE: {np.max(val_mapes):.2f}%")
                print(f"Models saved successfully: {len(saved_models)}/{len(all_results)}")
                
                # Print saved model paths
                print(f"\n📁 SAVED MODELS:")
                for split_num, results in all_results.items():
                    if results['saved_model_path']:
                        print(f"  Split {split_num}: {results['model_filename']}")
                    else:
                        error_msg = results.get('save_error', 'Unknown error')
                        print(f"  Split {split_num}: ❌ SAVE FAILED - {error_msg}")
                
                avg_mape = np.mean(val_mapes)
                if avg_mape <= 15:
                    grade = "EXCELLENT"
                elif avg_mape <= 20:
                    grade = "GOOD"
                elif avg_mape <= 30:
                    grade = "FAIR"
                else:
                    grade = "POOR"
                
                print(f"Overall Grade: {grade}")
                
                if len(saved_models) == len(all_results):
                    print("🎉 ALL MODELS SAVED SUCCESSFULLY - Ready for Phase 3!")
                else:
                    print("⚠️ Some models failed to save - Check errors above")
            else:
                print("\n❌ ALL SPLITS FAILED - Check errors above")
        
        return all_results

def main():
    """Main function with fixed TensorFlow implementation"""
    parser = argparse.ArgumentParser(description="Fixed TensorFlow Training")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to engineered dataset")
    parser.add_argument("--output-dir", type=str, default="outputs_tensorflow_fixed",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size")
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("FIXED TENSORFLOW TRAINING (BYPASSING CORRUPTED IMPLEMENTATION)")
    logger.info("=" * 80)
    logger.info(f"Using pure TensorFlow implementation matching notebook exactly")
    
    try:
        # Load dataset
        feature_pipeline = SalesFeaturePipeline()
        df_final, features, rolling_splits, metadata = feature_pipeline.load_engineered_dataset(args.dataset_path)
        
        logger.info(f"Dataset loaded: {len(df_final):,} records, {len(features)} features")
        
        # Initialize FIXED model
        model = AdvancedEmbeddingModel(random_seed=42)
        
        # Train with exact notebook parameters
        results = model.train_on_rolling_splits(
            df_final=df_final,
            features=features,
            rolling_splits=rolling_splits,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Print comparison
        val_mapes = [r['val_mape'] for r in results.values()]
        current_mape = np.mean(val_mapes)
        previous_mape = 125.68
        
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON WITH PREVIOUS POOR RESULTS")
        logger.info("=" * 80)
        logger.info(f"Previous (PyTorch) MAPE: {previous_mape:.2f}%")
        logger.info(f"Current (TensorFlow) MAPE: {current_mape:.2f}%")
        
        if current_mape < previous_mape:
            improvement = ((previous_mape - current_mape) / previous_mape) * 100
            logger.info(f"🎉 IMPROVEMENT: {improvement:.1f}% better!")
        
        logger.info("🎯 FIXED TRAINING COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()