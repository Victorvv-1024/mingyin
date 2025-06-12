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
                # Extract single temporal feature
                single_temporal = layers.Lambda(lambda x, idx=i: x[:, idx:idx+1])(temporal_input)
                
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
                single_continuous = layers.Lambda(lambda x, idx=i: x[:, idx:idx+1])(continuous_input)
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
    
    def train_on_rolling_splits(self, df_final, features, rolling_splits, epochs=100, batch_size=512):
        """Train on rolling splits - TENSORFLOW VERSION"""
        print("=" * 80)
        print("TRAINING TENSORFLOW ADVANCED EMBEDDING MODEL ON ROLLING SPLITS")
        print("=" * 80)
        
        # Analyze features
        feature_categories = self.categorize_features_for_embeddings(df_final, features)
        
        all_results = {}
        
        for i, (train_df, val_df, description) in enumerate(rolling_splits, 1):
            print(f"\nSplit {i}/{len(rolling_splits)}: {description}")
            print(f"Train: {len(train_df):,} samples, Val: {len(val_df):,} samples")
            
            # Prepare data
            X_train, y_train = self.prepare_embedding_features(train_df, feature_categories, is_training=True)
            X_val, y_val = self.prepare_embedding_features(val_df, feature_categories, is_training=False)
            
            print(f"Prepared {len(X_train)} input types for training")
            
            # Get data shapes for model creation
            data_shapes = {key: data.shape[1] for key, data in X_train.items()}
            print(f"Data shapes: {data_shapes}")
            
            # Create model
            self.model = self.create_advanced_embedding_model(feature_categories, data_shapes)
            
            # Prepare inputs in correct order
            X_train_ordered = [X_train[key] for key in ['temporal', 'continuous', 'direct'] if key in X_train]
            X_val_ordered = [X_val[key] for key in ['temporal', 'continuous', 'direct'] if key in X_val]
            
            # Callbacks (EXACTLY like notebook)
            callbacks_list = [
                callbacks.EarlyStopping(
                    patience=20,  # Notebook used 20
                    restore_best_weights=True,
                    monitor='val_mape_metric_original_scale',
                    mode='min'
                ),
                callbacks.ReduceLROnPlateau(
                    patience=10,  # Notebook used 10
                    factor=0.5,   # Notebook used 0.5
                    monitor='val_mape_metric_original_scale',
                    mode='min'
                )
            ]
            
            # Train (EXACTLY like notebook)
            print("Training TensorFlow advanced embedding model...")
            history = self.model.fit(
                X_train_ordered, y_train,
                validation_data=(X_val_ordered, y_val),
                epochs=epochs,        # 100 like notebook
                batch_size=batch_size,  # 512 like notebook
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Calculate final metrics
            train_pred = self.model.predict(X_train_ordered, verbose=0)
            val_pred = self.model.predict(X_val_ordered, verbose=0)
            
            train_mape = self.safe_mape_calculation(y_train, train_pred.flatten())
            val_mape = self.safe_mape_calculation(y_val, val_pred.flatten())
            
            # Convert to original scale for RMSE and R²
            train_true_orig = np.expm1(y_train)
            train_pred_orig = np.expm1(train_pred.flatten())
            val_true_orig = np.expm1(y_val)
            val_pred_orig = np.expm1(val_pred.flatten())
            
            train_rmse = np.sqrt(mean_squared_error(train_true_orig, train_pred_orig))
            val_rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
            
            from sklearn.metrics import r2_score
            train_r2 = r2_score(train_true_orig, train_pred_orig)
            val_r2 = r2_score(val_true_orig, val_pred_orig)
            
            results = {
                'split_num': i,
                'train_mape': train_mape,
                'val_mape': val_mape,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'epochs_trained': len(history.history['loss']),
                'best_epoch': np.argmin(history.history['val_mape_metric_original_scale']) + 1,
                'description': description,
                # ADD THESE LINES:
                'train_predictions': train_pred.flatten(),
                'val_predictions': val_pred.flatten(),
                'training_history': history.history
            }
            
            all_results[i] = results
            
            print(f"Split {i} Results:")
            print(f"  Train MAPE: {train_mape:.2f}%, Val MAPE: {val_mape:.2f}%")
            print(f"  Train RMSE: {train_rmse:.0f}, Val RMSE: {val_rmse:.0f}")
            print(f"  Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}")
            print(f"  Best epoch: {results['best_epoch']}/{results['epochs_trained']}")
            
            # Reset model for next split
            self.model = None
        
        # Print overall performance
        if all_results:
            val_mapes = [results['val_mape'] for results in all_results.values()]
            print("\n" + "=" * 80)
            print("OVERALL PERFORMANCE SUMMARY")
            print("=" * 80)
            print(f"Average Validation MAPE: {np.mean(val_mapes):.2f}% ± {np.std(val_mapes):.2f}%")
            print(f"Best Split MAPE: {np.min(val_mapes):.2f}%")
            print(f"Worst Split MAPE: {np.max(val_mapes):.2f}%")
            
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
            print(f"Business Ready: {'YES' if avg_mape <= 20 else 'NO'}")
        
        print("=" * 80)
        print("TENSORFLOW TRAINING COMPLETED")
        print("=" * 80)
        
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