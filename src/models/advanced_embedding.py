"""
Advanced embedding-based deep learning model for sales forecasting.

This module implements the sophisticated neural network architecture
from full_data_prediction.ipynb with proper feature processing and
multi-input embedding strategies.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from .feature_processor import FeatureProcessor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Custom metrics for monitoring training
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
    """
    Advanced embedding-based deep learning model for sales forecasting.
    
    Implements sophisticated multi-input architecture with embeddings for
    categorical features and proper handling of different feature types.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the advanced embedding model.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.feature_processor = FeatureProcessor()
        self.model = None
        self.training_history = None
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info("AdvancedEmbeddingModel initialized")
    
    def safe_mape_calculation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Safe MAPE calculation with proper error handling.
        
        Args:
            y_true: True values (log scale)
            y_pred: Predicted values (log scale)
            
        Returns:
            MAPE value (percentage)
        """
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        y_pred_orig = np.clip(y_pred_orig, 0.1, 1e6)
        y_true_orig = np.clip(y_true_orig, 0.1, 1e6)
        epsilon = 1.0
        ape = np.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)
        mape = np.mean(ape) * 100
        return min(mape, 1000.0)
    
    def create_advanced_embedding_model(self, 
                                      input_shapes: Dict[str, int], 
                                      embedding_configs: Dict[str, Dict]) -> Model:
        """
        Create the advanced embedding-based neural network architecture.
        
        Args:
            input_shapes: Dictionary mapping input names to shapes
            embedding_configs: Dictionary with embedding configurations
            
        Returns:
            Compiled Keras model
        """
        logger.info("Creating advanced embedding model architecture...")
        
        inputs = {}
        processed_inputs = []
        total_embedding_dim = 0
        
        # 1. Temporal categorical inputs (with embeddings)
        if 'temporal_categorical' in input_shapes:
            temporal_categorical_input = layers.Input(
                shape=(input_shapes['temporal_categorical'],), 
                name='temporal_categorical_input'
            )
            inputs['temporal_categorical'] = temporal_categorical_input
            
            # Process each temporal categorical feature with specific embeddings
            temporal_embeddings = []
            for i in range(input_shapes['temporal_categorical']):
                # Extract single feature
                single_feature = layers.Lambda(
                    lambda x, idx=i: x[:, idx:idx+1]
                )(temporal_categorical_input)
                
                # Get embedding config for this feature
                config_key = f'temporal_categorical_{i}'
                if config_key in embedding_configs:
                    config = embedding_configs[config_key]
                    vocab_size = config['vocab_size']
                    embedding_dim = config['embedding_dim']
                    feature_name = config['feature_name']
                else:
                    vocab_size = 50
                    embedding_dim = 8
                    feature_name = f'temporal_{i}'
                
                # Create embedding
                emb = layers.Embedding(
                    vocab_size, 
                    embedding_dim, 
                    name=f'{feature_name}_embedding'
                )(single_feature)
                emb_flat = layers.Flatten()(emb)
                temporal_embeddings.append(emb_flat)
                total_embedding_dim += embedding_dim
            
            if len(temporal_embeddings) > 1:
                temporal_combined = layers.Concatenate(name='temporal_combined')(temporal_embeddings)
            else:
                temporal_combined = temporal_embeddings[0]
            
            processed_inputs.append(temporal_combined)
            logger.info(f"  Temporal categorical: {len(temporal_embeddings)} features, {total_embedding_dim} dims")
        
        # 2. Temporal continuous inputs
        if 'temporal_continuous' in input_shapes:
            temporal_continuous_input = layers.Input(
                shape=(input_shapes['temporal_continuous'],), 
                name='temporal_continuous_input'
            )
            inputs['temporal_continuous'] = temporal_continuous_input
            
            # Process with dense layer
            temporal_continuous_processed = layers.Dense(
                16, activation='relu', name='temporal_continuous_dense'
            )(temporal_continuous_input)
            temporal_continuous_processed = layers.BatchNormalization(
                name='temporal_continuous_bn'
            )(temporal_continuous_processed)
            
            processed_inputs.append(temporal_continuous_processed)
            total_embedding_dim += 16
            logger.info(f"  Temporal continuous: {input_shapes['temporal_continuous']} → 16 dims")
        
        # 3. Cyclical inputs (sin/cos features)
        if 'cyclical' in input_shapes:
            cyclical_input = layers.Input(
                shape=(input_shapes['cyclical'],), 
                name='cyclical_input'
            )
            inputs['cyclical'] = cyclical_input
            
            # Cyclical features don't need much processing
            cyclical_processed = layers.Dense(
                8, activation='tanh', name='cyclical_dense'
            )(cyclical_input)
            
            processed_inputs.append(cyclical_processed)
            total_embedding_dim += 8
            logger.info(f"  Cyclical: {input_shapes['cyclical']} → 8 dims")
        
        # 4. Binary inputs
        if 'binary' in input_shapes:
            binary_input = layers.Input(
                shape=(input_shapes['binary'],), 
                name='binary_input'
            )
            inputs['binary'] = binary_input
            
            # Binary features with light processing
            binary_processed = layers.Dense(
                12, activation='relu', name='binary_dense'
            )(binary_input)
            binary_processed = layers.Dropout(0.2, name='binary_dropout')(binary_processed)
            
            processed_inputs.append(binary_processed)
            total_embedding_dim += 12
            logger.info(f"  Binary: {input_shapes['binary']} → 12 dims")
        
        # 5. Continuous inputs (main numerical features)
        if 'continuous' in input_shapes:
            continuous_input = layers.Input(
                shape=(input_shapes['continuous'],), 
                name='continuous_input'
            )
            inputs['continuous'] = continuous_input
            
            # Main numerical features processing
            continuous_processed = layers.Dense(
                32, activation='relu', name='continuous_dense'
            )(continuous_input)
            continuous_processed = layers.BatchNormalization(
                name='continuous_bn'
            )(continuous_processed)
            continuous_processed = layers.Dropout(0.3, name='continuous_dropout')(continuous_processed)
            
            processed_inputs.append(continuous_processed)
            total_embedding_dim += 32
            logger.info(f"  Continuous: {input_shapes['continuous']} → 32 dims")
        
        # Combine all processed inputs
        if len(processed_inputs) > 1:
            combined = layers.Concatenate(name='combine_all_inputs')(processed_inputs)
        else:
            combined = processed_inputs[0]
        
        logger.info(f"  Total combined dimension: {total_embedding_dim}")
        
        # Advanced neural network architecture
        
        # Input standardization layer
        x = layers.Dense(256, activation='relu', name='input_standardization')(combined)
        x = layers.BatchNormalization(name='input_bn')(x)
        x = layers.Dropout(0.3, name='input_dropout')(x)
        
        # Multi-head attention mechanism
        attention_heads = []
        for i in range(4):
            attention_head = layers.Dense(64, activation='tanh', name=f'attention_head_{i}')(x)
            attention_heads.append(attention_head)
        
        multi_head_attention = layers.Concatenate(name='multi_head_attention')(attention_heads)
        
        # Residual connection
        x = layers.Add(name='residual_connection')([x, multi_head_attention])
        x = layers.LayerNormalization(name='layer_norm')(x)
        
        # Deep layers with skip connections
        skip_connection_1 = x
        
        # Block 1
        x = layers.Dense(256, activation='relu', name='deep_1')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='deep_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        
        # Skip connection
        skip_processed = layers.Dense(128, name='skip_1_projection')(skip_connection_1)
        x = layers.Add(name='skip_connection_1')([x, skip_processed])
        
        skip_connection_2 = x
        
        # Block 2
        x = layers.Dense(64, activation='relu', name='deep_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='deep_4')(x)
        x = layers.Dropout(0.1, name='dropout_4')(x)
        
        # Final skip connection
        skip_processed_2 = layers.Dense(32, name='skip_2_projection')(skip_connection_2)
        x = layers.Add(name='skip_connection_2')([x, skip_processed_2])
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='sales_prediction')(x)
        
        # Create model
        model = Model(inputs=list(inputs.values()), outputs=output, name='AdvancedEmbeddingModel')
        
        # Compile with advanced optimizer
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[
                'mae',
                mape_metric_original_scale,
                rmse_metric_original_scale
            ]
        )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        logger.info("Model compilation completed")
        
        return model
    
    def create_callbacks(self, 
                        split_num: int, 
                        save_dir: str = "outputs/models",
                        patience: int = 15) -> List[callbacks.Callback]:
        """
        Create training callbacks for model training.
        
        Args:
            split_num: Current split number
            save_dir: Directory to save model checkpoints
            patience: Early stopping patience
            
        Returns:
            List of Keras callbacks
        """
        import os
        from datetime import datetime
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model checkpoint
        checkpoint_path = os.path.join(save_dir, f"best_model_split_{split_num}_{timestamp}.h5")
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_mape_metric_original_scale',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_mape_metric_original_scale',
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        
        # Learning rate reduction
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_mape_metric_original_scale',
            factor=0.7,
            patience=7,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
        
        # CSV Logger
        csv_path = os.path.join(save_dir, f"training_log_split_{split_num}_{timestamp}.csv")
        csv_logger = callbacks.CSVLogger(csv_path)
        
        return [checkpoint, early_stopping, lr_reducer, csv_logger]
    
    def train_model(self, 
                   train_data: Dict[str, np.ndarray], 
                   train_target: np.ndarray,
                   val_data: Dict[str, np.ndarray], 
                   val_target: np.ndarray,
                   split_num: int,
                   epochs: int = 100,
                   batch_size: int = 512) -> Dict[str, any]:
        """
        Train the model on given data.
        
        Args:
            train_data: Training data dictionary
            train_target: Training targets
            val_data: Validation data dictionary
            val_target: Validation targets
            split_num: Current split number
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training model for split {split_num}...")
        
        # Create model if not exists
        if self.model is None:
            input_shapes = {k: v.shape[1] for k, v in train_data.items()}
            embedding_configs = self.feature_processor.get_embedding_configs()
            self.model = self.create_advanced_embedding_model(input_shapes, embedding_configs)
        
        # Create callbacks
        callbacks_list = self.create_callbacks(split_num)
        
        # Train model
        history = self.model.fit(
            x=list(train_data.values()),
            y=train_target,
            validation_data=(list(val_data.values()), val_target),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.training_history = history
        
        # Generate predictions
        train_pred = self.model.predict(list(train_data.values()), verbose=0)
        val_pred = self.model.predict(list(val_data.values()), verbose=0)
        
        # Calculate metrics
        train_mape = self.safe_mape_calculation(train_target, train_pred.flatten())
        val_mape = self.safe_mape_calculation(val_target, val_pred.flatten())
        
        train_rmse = np.sqrt(mean_squared_error(
            self.feature_processor.inverse_transform_target(train_target),
            self.feature_processor.inverse_transform_target(train_pred.flatten())
        ))
        val_rmse = np.sqrt(mean_squared_error(
            self.feature_processor.inverse_transform_target(val_target),
            self.feature_processor.inverse_transform_target(val_pred.flatten())
        ))
        
        train_r2 = r2_score(
            self.feature_processor.inverse_transform_target(train_target),
            self.feature_processor.inverse_transform_target(train_pred.flatten())
        )
        val_r2 = r2_score(
            self.feature_processor.inverse_transform_target(val_target),
            self.feature_processor.inverse_transform_target(val_pred.flatten())
        )
        
        results = {
            'split_num': split_num,
            'train_mape': train_mape,
            'val_mape': val_mape,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_predictions': train_pred.flatten(),
            'val_predictions': val_pred.flatten(),
            'epochs_trained': len(history.history['loss']),
            'best_epoch': np.argmin(history.history['val_mape_metric_original_scale']) + 1,
            'training_history': history.history
        }
        
        logger.info(f"Training completed for split {split_num}")
        logger.info(f"  Train MAPE: {train_mape:.2f}%, Val MAPE: {val_mape:.2f}%")
        logger.info(f"  Train RMSE: {train_rmse:.0f}, Val RMSE: {val_rmse:.0f}")
        logger.info(f"  Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}")
        logger.info(f"  Best epoch: {results['best_epoch']}/{results['epochs_trained']}")
        
        return results
    
    def train_on_rolling_splits(self, 
                               df_final: pd.DataFrame, 
                               features: List[str], 
                               rolling_splits: List[Tuple],
                               epochs: int = 100,
                               batch_size: int = 512) -> Dict[int, Dict]:
        """
        Train model on all rolling splits.
        
        Args:
            df_final: Final engineered dataset
            features: List of modeling features
            rolling_splits: List of (train_df, val_df, description) tuples
            epochs: Number of training epochs per split
            batch_size: Training batch size
            
        Returns:
            Dictionary mapping split numbers to results
        """
        logger.info("=" * 80)
        logger.info("TRAINING ADVANCED EMBEDDING MODEL ON ROLLING SPLITS")
        logger.info("=" * 80)
        
        all_results = {}
        
        for i, (train_df, val_df, description) in enumerate(rolling_splits, 1):
            logger.info(f"\nSplit {i}/{len(rolling_splits)}: {description}")
            logger.info(f"Train: {len(train_df):,} samples, Val: {len(val_df):,} samples")
            
            # Prepare training data
            train_data, train_target = self.feature_processor.prepare_data_for_training(
                train_df, features, is_training=True
            )
            
            # Prepare validation data
            val_data, val_target = self.feature_processor.prepare_data_for_training(
                val_df, features, is_training=False
            )
            
            # Validate data
            train_validation = self.feature_processor.validate_prepared_data(train_data)
            val_validation = self.feature_processor.validate_prepared_data(val_data)
            
            if not train_validation['validation_passed'] or not val_validation['validation_passed']:
                logger.warning(f"Data validation failed for split {i}")
                continue
            
            # Train model
            results = self.train_model(
                train_data, train_target, 
                val_data, val_target, 
                split_num=i,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Add additional information
            results.update({
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'description': description,
                'train_date_range': {
                    'start': train_df['sales_month'].min().strftime('%Y-%m-%d'),
                    'end': train_df['sales_month'].max().strftime('%Y-%m-%d')
                },
                'val_date_range': {
                    'start': val_df['sales_month'].min().strftime('%Y-%m-%d'),
                    'end': val_df['sales_month'].max().strftime('%Y-%m-%d')
                }
            })
            
            all_results[i] = results
            
            # Reset model for next split
            self.model = None
        
        # Calculate overall performance
        if all_results:
            self._print_overall_performance_summary(all_results)
        
        logger.info("=" * 80)
        logger.info("ROLLING SPLITS TRAINING COMPLETED")
        logger.info("=" * 80)
        
        return all_results
    
    def _print_overall_performance_summary(self, all_results: Dict[int, Dict]) -> None:
        """Print overall performance summary across all splits."""
        
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Calculate aggregate metrics
        val_mapes = [results['val_mape'] for results in all_results.values()]
        val_rmses = [results['val_rmse'] for results in all_results.values()]
        val_r2s = [results['val_r2'] for results in all_results.values()]
        
        avg_mape = np.mean(val_mapes)
        avg_rmse = np.mean(val_rmses)
        avg_r2 = np.mean(val_r2s)
        
        std_mape = np.std(val_mapes)
        std_rmse = np.std(val_rmses)
        std_r2 = np.std(val_r2s)
        
        logger.info(f"Validation Performance Across {len(all_results)} Splits:")
        logger.info(f"  Average MAPE: {avg_mape:.2f}% ± {std_mape:.2f}%")
        logger.info(f"  Average RMSE: {avg_rmse:.0f} ± {std_rmse:.0f}")
        logger.info(f"  Average R²: {avg_r2:.3f} ± {std_r2:.3f}")
        
        # Individual split performance
        logger.info(f"\nIndividual Split Performance:")
        for split_num, results in all_results.items():
            logger.info(f"  Split {split_num}: MAPE={results['val_mape']:.2f}%, "
                       f"RMSE={results['val_rmse']:.0f}, R²={results['val_r2']:.3f}")
        
        # Performance assessment
        logger.info(f"\nPerformance Assessment:")
        if avg_mape <= 15:
            logger.info("🌟 EXCELLENT: Average MAPE ≤ 15% - Business-ready accuracy!")
        elif avg_mape <= 20:
            logger.info("✅ GOOD: Average MAPE ≤ 20% - Acceptable for most use cases")
        elif avg_mape <= 30:
            logger.info("⚠️ FAIR: Average MAPE ≤ 30% - May need improvement")
        else:
            logger.info("❌ POOR: Average MAPE > 30% - Requires significant improvement")
        
        if std_mape <= 3:
            logger.info("✅ CONSISTENT: Low MAPE variance across splits")
        else:
            logger.info("⚠️ INCONSISTENT: High MAPE variance - Check for overfitting")
    
    def save_predictions(self, 
                        results: Dict[int, Dict],
                        df_final: pd.DataFrame,
                        rolling_splits: List[Tuple],
                        output_dir: str = "outputs/predictions") -> Dict[str, str]:
        """
        Save detailed predictions and analysis.
        
        Args:
            results: Results from rolling splits training
            df_final: Original engineered dataset
            rolling_splits: Rolling splits data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with saved file paths
        """
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save detailed predictions for each split
        for split_num, result in results.items():
            train_df, val_df, description = rolling_splits[split_num - 1]
            
            # Create detailed prediction dataframe
            val_df_pred = val_df.copy()
            val_df_pred['predicted_log'] = result['val_predictions']
            val_df_pred['predicted_quantity'] = self.feature_processor.inverse_transform_target(
                result['val_predictions']
            )
            val_df_pred['actual_quantity'] = self.feature_processor.inverse_transform_target(
                val_df_pred['sales_quantity_log'].values
            )
            val_df_pred['absolute_error'] = np.abs(
                val_df_pred['predicted_quantity'] - val_df_pred['actual_quantity']
            )
            val_df_pred['percentage_error'] = (
                val_df_pred['absolute_error'] / (val_df_pred['actual_quantity'] + 1) * 100
            )
            
            # Save detailed predictions
            pred_file = os.path.join(output_dir, f"detailed_predictions_split_{split_num}_{timestamp}.csv")
            val_df_pred.to_csv(pred_file, index=False)
            saved_files[f'split_{split_num}_predictions'] = pred_file
        
        # Save summary report
        summary_file = os.path.join(output_dir, f"model_performance_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("ADVANCED EMBEDDING MODEL PERFORMANCE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Training completed: {timestamp}\n")
            f.write(f"Total splits trained: {len(results)}\n\n")
            
            # Overall metrics
            val_mapes = [r['val_mape'] for r in results.values()]
            avg_mape = np.mean(val_mapes)
            std_mape = np.std(val_mapes)
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"Average Validation MAPE: {avg_mape:.2f}% ± {std_mape:.2f}%\n")
            f.write(f"Best Split MAPE: {min(val_mapes):.2f}%\n")
            f.write(f"Worst Split MAPE: {max(val_mapes):.2f}%\n\n")
            
            # Individual split details
            f.write("INDIVIDUAL SPLIT PERFORMANCE:\n")
            for split_num, result in results.items():
                f.write(f"\nSplit {split_num}: {result['description']}\n")
                f.write(f"  Validation MAPE: {result['val_mape']:.2f}%\n")
                f.write(f"  Validation RMSE: {result['val_rmse']:.0f}\n")
                f.write(f"  Validation R²: {result['val_r2']:.3f}\n")
                f.write(f"  Training samples: {result['train_samples']:,}\n")
                f.write(f"  Validation samples: {result['val_samples']:,}\n")
                f.write(f"  Epochs trained: {result['epochs_trained']}\n")
                f.write(f"  Best epoch: {result['best_epoch']}\n")
        
        saved_files['summary_report'] = summary_file
        
        logger.info(f"Predictions and analysis saved:")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
        
        return saved_files