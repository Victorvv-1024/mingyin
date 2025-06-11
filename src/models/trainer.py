import tensorflow as tf
from tensorflow.keras import callbacks, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from .advanced_embedding import AdvancedEmbeddingModel, mape_metric_original_scale, rmse_metric_original_scale

class CheckpointManager(callbacks.Callback):
    """Custom callback to manage model checkpoints - save every 10 epochs, keep last 3, save best"""
    
    def __init__(self, checkpoint_dir, split_num, timestamp, save_freq=10, keep_last=3):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.split_num = split_num
        self.timestamp = timestamp
        self.save_freq = save_freq
        self.keep_last = keep_last
        self.best_mape = float('inf')
        self.best_epoch = 0
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        current_mape = logs.get('val_mape_metric_original_scale', float('inf'))
        
        # Save every 10 epochs
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"model_split_{self.split_num}_epoch_{epoch+1:03d}_{self.timestamp}.h5"
            )
            self.model.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved: epoch {epoch+1}")
            
            # Clean up old checkpoints (keep only last 3)
            self._cleanup_old_checkpoints()
        
        # Save best model
        if current_mape < self.best_mape:
            self.best_mape = current_mape
            self.best_epoch = epoch + 1
            best_model_path = os.path.join(
                self.checkpoint_dir,
                f"best_model_split_{self.split_num}_epoch_{epoch+1:03d}_mape_{current_mape:.2f}_{self.timestamp}.h5"
            )
            self.model.save(best_model_path)
            print(f"\n🌟 New best model saved: epoch {epoch+1}, MAPE: {current_mape:.2f}%")
            
            # Remove previous best model
            self._cleanup_old_best_models()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoint files"""
        pattern = os.path.join(self.checkpoint_dir, f"model_split_{self.split_num}_epoch_*_{self.timestamp}.h5")
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) > self.keep_last:
            # Sort by epoch number (extract from filename)
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('_')[0]))
            
            # Remove oldest files
            files_to_remove = checkpoint_files[:-self.keep_last]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    epoch_num = file_path.split('_epoch_')[1].split('_')[0]
                    print(f"🗑️ Removed old checkpoint: epoch {epoch_num}")
                except OSError:
                    pass
    
    def _cleanup_old_best_models(self):
        """Remove previous best model files for this split"""
        pattern = os.path.join(self.checkpoint_dir, f"best_model_split_{self.split_num}_*_{self.timestamp}.h5")
        best_files = glob.glob(pattern)
        
        if len(best_files) > 1:
            # Sort by MAPE (extract from filename) and keep only the best one
            best_files.sort(key=lambda x: float(x.split('_mape_')[1].split('_')[0]))
            
            # Remove all but the best one
            files_to_remove = best_files[1:]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

class ModelTrainer:
    """Handles the training process for the advanced embedding model"""
    
    def __init__(self, model: AdvancedEmbeddingModel):
        self.model = model
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = "checkpoints"
    
    def prepare_embedding_features(self, df, feature_categories, is_training=True):
        """Prepare features for embedding-based model"""
        df_work = df.copy()
        
        if 'sales_quantity_log' not in df_work.columns:
            df_work['sales_quantity_log'] = np.log1p(df_work['sales_quantity'])
        
        prepared_data = {}
        
        # Temporal features - create embeddings
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
                            self.model.encoders[f'{feature}_buckets'] = buckets
                        except:
                            self.model.encoders[f'{feature}_buckets'] = np.array([0, 1])
                    
                    bucket_edges = self.model.encoders.get(f'{feature}_buckets', np.array([0, 1]))
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
                    self.model.scalers['direct'] = RobustScaler()
                    direct_data = self.model.scalers['direct'].fit_transform(direct_data)
                else:
                    direct_data = self.model.scalers['direct'].transform(direct_data)
                
                prepared_data['direct'] = direct_data
        
        # Target
        target = df_work['sales_quantity_log'].values.astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-1.0)
        
        return prepared_data, target
    
    def create_advanced_embedding_model(self, feature_categories, data_shapes):
        """Create advanced embedding-based neural network"""
        print("\n=== CREATING ADVANCED EMBEDDING MODEL ===")
        
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
            print(f"  Direct features: {data_shapes['direct']} → {direct_dim} dimensions")
        
        # Calculate actual combined dimension
        print(f"  Expected total embedding dimension: {total_embedding_dim}")
        
        # Combine all embeddings
        if len(embedding_outputs) > 1:
            combined = layers.Concatenate(name='combine_all')(embedding_outputs)
        else:
            combined = embedding_outputs[0]
        
        # Adaptive standardization - use actual input dimension
        standardized = layers.Dense(256, activation='relu', name='standardize')(combined)
        standardized = layers.BatchNormalization(name='std_bn')(standardized)
        standardized = layers.Dropout(0.3, name='std_dropout')(standardized)
        
        # Multi-head attention with smaller heads
        attention_1 = layers.Dense(64, activation='tanh', name='attention_1')(standardized)
        attention_2 = layers.Dense(64, activation='tanh', name='attention_2')(standardized)
        attention_3 = layers.Dense(64, activation='tanh', name='attention_3')(standardized)
        attention_4 = layers.Dense(64, activation='tanh', name='attention_4')(standardized)
        
        multi_head = layers.Concatenate(name='multi_head')([attention_1, attention_2, attention_3, attention_4])
        
        # Residual connection - both inputs now 256-dim
        attended = layers.Add(name='residual_attention')([standardized, multi_head])
        attended = layers.LayerNormalization(name='layer_norm')(attended)
        
        # Deep layers
        x1 = layers.Dense(256, activation='relu', name='deep1')(attended)
        x1 = layers.BatchNormalization(name='bn1')(x1)
        x1 = layers.Dropout(0.3, name='drop1')(x1)
        
        x2 = layers.Dense(128, activation='relu', name='deep2')(x1)
        x2 = layers.BatchNormalization(name='bn2')(x2)
        x2 = layers.Dropout(0.2, name='drop2')(x2)
        
        x3 = layers.Dense(64, activation='relu', name='deep3')(x2)
        x3 = layers.Dropout(0.2, name='drop3')(x3)
        
        # Output
        output = layers.Dense(1, activation='linear', name='sales_prediction')(x3)
        
        model = Model(inputs=list(inputs.values()), outputs=output, name='AdvancedEmbeddingModel')
        
        print(f"  Model created with {model.count_params():,} parameters")
        print(f"  Input types: {list(inputs.keys())}")
        
        return model, list(inputs.keys())
    
    def train(self, df, features, rolling_splits):
        """Train the model on the provided data with proper checkpointing"""
        print("=" * 80)
        print("TRAINING ADVANCED EMBEDDING-BASED MODELS")
        print("=" * 80)
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"📁 Checkpoints will be saved to: {os.path.abspath(self.checkpoint_dir)}")
        
        # Analyze features
        feature_categories = self.model.categorize_features_for_embeddings(df, features)
        
        all_results = {}
        
        for split_idx, (train_data, val_data, description) in enumerate(rolling_splits):
            print(f"\nSplit {split_idx + 1}: {description}")
            print("-" * 50)
            
            try:
                # Prepare data
                X_train, y_train = self.prepare_embedding_features(train_data, feature_categories, is_training=True)
                X_val, y_val = self.prepare_embedding_features(val_data, feature_categories, is_training=False)
                
                print(f"Prepared {len(X_train)} input types for training")
                
                # Get data shapes for model creation
                data_shapes = {key: data.shape[1] for key, data in X_train.items()}
                print(f"Data shapes: {data_shapes}")
                
                # Create model
                model, input_order = self.create_advanced_embedding_model(feature_categories, data_shapes)
                
                # Compile with consistent metrics
                model.compile(
                    optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                    loss='mae',
                    metrics=[mape_metric_original_scale, rmse_metric_original_scale]
                )
                
                # Prepare inputs in correct order
                X_train_ordered = [X_train[key] for key in input_order if key in X_train]
                X_val_ordered = [X_val[key] for key in input_order if key in X_val]
                
                # Initialize checkpoint manager
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=self.checkpoint_dir,
                    split_num=split_idx + 1,
                    timestamp=self.timestamp,
                    save_freq=10,
                    keep_last=3
                )
                
                # Standard callbacks
                callbacks_list = [
                    checkpoint_manager,
                    callbacks.EarlyStopping(
                        patience=20,
                        restore_best_weights=True,
                        monitor='val_mape_metric_original_scale',
                        mode='min'
                    ),
                    callbacks.ReduceLROnPlateau(
                        patience=10,
                        factor=0.5,
                        monitor='val_mape_metric_original_scale',
                        mode='min'
                    )
                ]
                
                # Train
                print(f"🚀 Training advanced embedding model (max 100 epochs)...")
                print(f"💾 Checkpoints: every 10 epochs (keeping last 3)")
                print(f"🌟 Best model: saved automatically")
                
                history = model.fit(
                    X_train_ordered, y_train,
                    validation_data=(X_val_ordered, y_val),
                    epochs=100,
                    batch_size=512,
                    callbacks=callbacks_list,
                    verbose=1 if split_idx == 0 else 0
                )
                
                # Evaluate
                val_pred_log = model.predict(X_val_ordered, verbose=0)
                val_pred_orig = np.expm1(val_pred_log.flatten())
                val_true_orig = np.expm1(y_val)
                
                # Calculate metrics
                mape = self.model.safe_mape_calculation(y_val, val_pred_log.flatten())
                rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
                r2 = r2_score(val_true_orig, val_pred_orig)
                mae = np.mean(np.abs(val_true_orig - val_pred_orig))
                
                # Final model save (last epoch)
                final_model_path = os.path.join(
                    self.checkpoint_dir,
                    f"final_model_split_{split_idx+1}_epoch_{len(history.history['loss']):03d}_{self.timestamp}.h5"
                )
                model.save(final_model_path)
                
                print(f"\n✅ Split {split_idx+1} completed!")
                print(f"📊 Final MAPE: {mape:.2f}%")
                print(f"🌟 Best MAPE: {checkpoint_manager.best_mape:.2f}% (epoch {checkpoint_manager.best_epoch})")
                print(f"💾 Final model saved: {os.path.basename(final_model_path)}")
                
                # Store results
                all_results[f'split_{split_idx+1}'] = {
                    'description': description,
                    'mape': mape,
                    'best_mape': checkpoint_manager.best_mape,
                    'best_epoch': checkpoint_manager.best_epoch,
                    'train_mape': history.history.get('val_mape_metric_original_scale', [None])[-1],
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'saved_files': {
                        'final_model': final_model_path,
                        'best_model': f"best_model_split_{split_idx+1}_epoch_{checkpoint_manager.best_epoch:03d}_mape_{checkpoint_manager.best_mape:.2f}_{self.timestamp}.h5"
                    }
                }
                
            except Exception as e:
                print(f"Error in split {split_idx + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results 