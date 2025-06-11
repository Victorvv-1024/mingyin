import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedEmbeddingModel(nn.Module):
    """PyTorch implementation of Advanced Embedding Model for sales forecasting"""
    
    def __init__(self, temporal_vocab_sizes, continuous_vocab_sizes, direct_features_dim):
        super(AdvancedEmbeddingModel, self).__init__()
        
        # Store dimensions
        self.temporal_vocab_sizes = temporal_vocab_sizes
        self.continuous_vocab_sizes = continuous_vocab_sizes
        self.direct_features_dim = direct_features_dim
        
        # Temporal embeddings
        self.temporal_embeddings = nn.ModuleList()
        temporal_embedding_dim = 0
        for i, vocab_size in enumerate(temporal_vocab_sizes):
            if i == 0:  # Month
                emb_dim = 8
                self.temporal_embeddings.append(nn.Embedding(12, emb_dim))
            elif i == 1:  # Quarter
                emb_dim = 4
                self.temporal_embeddings.append(nn.Embedding(4, emb_dim))
            else:
                emb_dim = 8
                self.temporal_embeddings.append(nn.Embedding(101, emb_dim))
            temporal_embedding_dim += emb_dim
        
        # Continuous feature embeddings
        self.continuous_embeddings = nn.ModuleList()
        continuous_embedding_dim = 0
        for vocab_size in continuous_vocab_sizes:
            emb_dim = 8
            self.continuous_embeddings.append(nn.Embedding(52, emb_dim))
            continuous_embedding_dim += emb_dim
        
        # Direct features processing
        direct_dim = 32
        self.direct_dense = nn.Sequential(
            nn.Linear(direct_features_dim, direct_dim),
            nn.ReLU(),
            nn.BatchNorm1d(direct_dim),
            nn.Dropout(0.2)
        )
        
        # Calculate total embedding dimension
        total_embedding_dim = temporal_embedding_dim + continuous_embedding_dim + direct_dim
        
        # Standardization layer
        self.standardize = nn.Sequential(
            nn.Linear(total_embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 64), nn.Tanh())
            for _ in range(4)
        ])
        
        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(256)
        
        # Deep layers
        self.deep_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer
        self.output = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, temporal_input, continuous_input, direct_input):
        """Forward pass"""
        embeddings = []
        
        # Process temporal embeddings
        if temporal_input is not None and len(self.temporal_embeddings) > 0:
            temporal_embeds = []
            for i, embedding_layer in enumerate(self.temporal_embeddings):
                single_temporal = temporal_input[:, i:i+1].long()
                emb = embedding_layer(single_temporal).squeeze(1)
                temporal_embeds.append(emb)
            
            if temporal_embeds:
                temporal_combined = torch.cat(temporal_embeds, dim=1)
                embeddings.append(temporal_combined)
        
        # Process continuous embeddings
        if continuous_input is not None and len(self.continuous_embeddings) > 0:
            continuous_embeds = []
            for i, embedding_layer in enumerate(self.continuous_embeddings):
                single_continuous = continuous_input[:, i:i+1].long()
                emb = embedding_layer(single_continuous).squeeze(1)
                continuous_embeds.append(emb)
            
            if continuous_embeds:
                continuous_combined = torch.cat(continuous_embeds, dim=1)
                embeddings.append(continuous_combined)
        
        # Process direct features
        if direct_input is not None:
            direct_processed = self.direct_dense(direct_input)
            embeddings.append(direct_processed)
        
        # Combine all embeddings
        if len(embeddings) > 1:
            combined = torch.cat(embeddings, dim=1)
        else:
            combined = embeddings[0]
        
        # Standardization
        standardized = self.standardize(combined)
        
        # Multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_outputs.append(attention_head(standardized))
        
        multi_head = torch.cat(attention_outputs, dim=1)
        
        # Residual connection
        attended = standardized + multi_head
        attended = self.layer_norm(attended)
        
        # Deep layers
        deep_output = self.deep_layers(attended)
        
        # Final output
        output = self.output(deep_output)
        
        return output

def safe_mape_calculation(y_true, y_pred, epsilon=1e-8):
    """Calculate MAPE with protection against division by zero"""
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return np.clip(mape, 0, 1000)

class CheckpointManager:
    """Manages model checkpoints - save every N epochs, keep last K, save best"""
    
    def __init__(self, checkpoint_dir, split_num, timestamp, save_freq=10, keep_last=3):
        self.checkpoint_dir = checkpoint_dir
        self.split_num = split_num
        self.timestamp = timestamp
        self.save_freq = save_freq
        self.keep_last = keep_last
        self.best_mape = float('inf')
        self.best_epoch = 0
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, epoch, val_mape):
        """Save checkpoint if it's the right epoch"""
        # Save every N epochs
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"model_split_{self.split_num}_epoch_{epoch+1:03d}_{self.timestamp}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\n💾 Checkpoint saved: epoch {epoch+1}")
            
            # Clean up old checkpoints (keep only last 3)
            self._cleanup_old_checkpoints()
        
        # Save best model
        if val_mape < self.best_mape:
            self.best_mape = val_mape
            self.best_epoch = epoch + 1
            best_model_path = os.path.join(
                self.checkpoint_dir,
                f"best_model_split_{self.split_num}_epoch_{epoch+1:03d}_mape_{val_mape:.2f}_{self.timestamp}.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"\n🌟 New best model saved: epoch {epoch+1}, MAPE: {val_mape:.2f}%")
            
            # Remove previous best model
            self._cleanup_old_best_models()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoint files"""
        pattern = os.path.join(self.checkpoint_dir, f"model_split_{self.split_num}_epoch_*_{self.timestamp}.pth")
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
        pattern = os.path.join(self.checkpoint_dir, f"best_model_split_{self.split_num}_*_{self.timestamp}.pth")
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

class PyTorchAdvancedEmbeddingModel:
    """Main class for handling the PyTorch advanced embedding model"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        # Use CPU to avoid MPS issues with large models
        self.device = torch.device('cpu')
        print(f"🔥 Using device: {self.device} (CPU for stability)")
        
    def categorize_features_for_embeddings(self, df, feature_columns):
        """Categorize features for embedding processing"""
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

class PyTorchModelTrainer:
    """Handles training of PyTorch models with checkpointing"""
    
    def __init__(self, model: PyTorchAdvancedEmbeddingModel):
        self.model = model
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = "checkpoints"
    
    def prepare_embedding_features(self, df, feature_categories, is_training=True):
        """Prepare features for embedding-based model"""
        df_work = df.copy()
        
        if 'sales_quantity_log' not in df_work.columns:
            df_work['sales_quantity_log'] = np.log1p(df_work['sales_quantity'])
        
        prepared_data = {}
        
        # Temporal features
        temporal_features = feature_categories['temporal']
        if temporal_features:
            temporal_data = []
            for feature in temporal_features:
                if feature in df_work.columns:
                    # Handle different data types including Period objects
                    if hasattr(df_work[feature].dtype, 'name') and 'period' in df_work[feature].dtype.name.lower():
                        # Convert Period to int
                        values = df_work[feature].dt.astype(int).fillna(0).values.astype(int)
                    elif df_work[feature].dtype == 'object':
                        # Try to convert object to numeric
                        try:
                            values = pd.to_numeric(df_work[feature], errors='coerce').fillna(0).values.astype(int)
                        except:
                            values = np.zeros(len(df_work), dtype=int)
                    else:
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
        
        # Continuous features - bucketize
        continuous_features = feature_categories['numerical_continuous']
        if continuous_features:
            continuous_data = []
            for feature in continuous_features:
                if feature in df_work.columns:
                    # Handle different column types
                    if pd.api.types.is_categorical_dtype(df_work[feature]):
                        # For categorical columns, convert to numeric codes
                        values = df_work[feature].cat.codes.values
                    else:
                        # For numeric columns, handle inf/nan
                        values = df_work[feature].replace([np.inf, -np.inf], np.nan)
                        if not pd.api.types.is_numeric_dtype(values):
                            values = pd.to_numeric(values, errors='coerce')
                        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
                    
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
        
        # Direct features
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
                    # If scaler doesn't exist, create a new one and fit it
                    if 'direct' not in self.model.scalers:
                        print("Warning: Direct features scaler not found. Creating new scaler...")
                        self.model.scalers['direct'] = RobustScaler()
                        self.model.scalers['direct'].fit(direct_data)
                    direct_data = self.model.scalers['direct'].transform(direct_data)
                
                prepared_data['direct'] = direct_data
        
        # Target
        target = df_work['sales_quantity_log'].values.astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-1.0)
        
        return prepared_data, target
    
    def train(self, df, features, rolling_splits):
        """Train the PyTorch model with checkpointing"""
        print("=" * 80)
        print("TRAINING PYTORCH ADVANCED EMBEDDING-BASED MODELS")
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
                
                # Get dimensions for model creation
                temporal_vocab_sizes = []
                continuous_vocab_sizes = []
                direct_features_dim = 0
                
                if 'temporal' in X_train:
                    temporal_vocab_sizes = [12, 4] + [101] * (X_train['temporal'].shape[1] - 2)
                if 'continuous' in X_train:
                    continuous_vocab_sizes = [52] * X_train['continuous'].shape[1]
                if 'direct' in X_train:
                    direct_features_dim = X_train['direct'].shape[1]
                
                # Create model
                model = AdvancedEmbeddingModel(
                    temporal_vocab_sizes=temporal_vocab_sizes,
                    continuous_vocab_sizes=continuous_vocab_sizes,
                    direct_features_dim=direct_features_dim
                ).to(self.model.device)
                
                print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
                
                # Prepare tensors
                def prepare_tensors(X_dict, y_array):
                    tensors = {}
                    for key, array in X_dict.items():
                        tensors[key] = torch.FloatTensor(array).to(self.model.device)
                    target_tensor = torch.FloatTensor(y_array).unsqueeze(1).to(self.model.device)
                    return tensors, target_tensor
                
                X_train_tensors, y_train_tensor = prepare_tensors(X_train, y_train)
                X_val_tensors, y_val_tensor = prepare_tensors(X_val, y_val)
                
                # Initialize training
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
                criterion = nn.L1Loss()  # MAE loss
                
                # Initialize checkpoint manager
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=self.checkpoint_dir,
                    split_num=split_idx + 1,
                    timestamp=self.timestamp,
                    save_freq=10,
                    keep_last=3
                )
                
                print(f"🚀 Training PyTorch model (max 100 epochs)...")
                print(f"💾 Checkpoints: every 10 epochs (keeping last 3)")
                print(f"🌟 Best model: saved automatically")
                
                # Training loop
                model.train()
                best_val_mape = float('inf')
                patience_counter = 0
                
                for epoch in range(100):
                    # Training phase
                    model.train()
                    optimizer.zero_grad()
                    
                    # Forward pass
                    train_outputs = model(
                        X_train_tensors.get('temporal'),
                        X_train_tensors.get('continuous'), 
                        X_train_tensors.get('direct')
                    )
                    
                    train_loss = criterion(train_outputs, y_train_tensor)
                    train_loss.backward()
                    optimizer.step()
                    
                    # Validation phase
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(
                            X_val_tensors.get('temporal'),
                            X_val_tensors.get('continuous'),
                            X_val_tensors.get('direct')
                        )
                        val_loss = criterion(val_outputs, y_val_tensor)
                        
                        # Calculate MAPE
                        val_pred_np = val_outputs.cpu().numpy().flatten()
                        val_mape = safe_mape_calculation(y_val, val_pred_np)
                    
                    # Checkpoint management
                    checkpoint_manager.save_checkpoint(model, epoch, val_mape)
                    
                    # Learning rate scheduling
                    scheduler.step(val_mape)
                    
                    # Early stopping
                    if val_mape < best_val_mape:
                        best_val_mape = val_mape
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= 20:
                        print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                        break
                    
                    # Print progress
                    if epoch % 10 == 0 or epoch == 99:
                        print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.2f}%")
                
                # Final evaluation
                model.eval()
                with torch.no_grad():
                    val_pred_log = model(
                        X_val_tensors.get('temporal'),
                        X_val_tensors.get('continuous'),
                        X_val_tensors.get('direct')
                    ).cpu().numpy().flatten()
                    
                    val_pred_orig = np.expm1(val_pred_log)
                    val_true_orig = np.expm1(y_val)
                    
                    # Calculate metrics
                    mape = safe_mape_calculation(y_val, val_pred_log)
                    rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
                    r2 = r2_score(val_true_orig, val_pred_orig)
                    mae = np.mean(np.abs(val_true_orig - val_pred_orig))
                
                # Final model save
                final_model_path = os.path.join(
                    self.checkpoint_dir,
                    f"final_model_split_{split_idx+1}_epoch_{epoch+1:03d}_{self.timestamp}.pth"
                )
                torch.save(model.state_dict(), final_model_path)
                
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
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'saved_files': {
                        'final_model': final_model_path,
                        'best_model': f"best_model_split_{split_idx+1}_epoch_{checkpoint_manager.best_epoch:03d}_mape_{checkpoint_manager.best_mape:.2f}_{self.timestamp}.pth"
                    }
                }
                
            except Exception as e:
                print(f"Error in split {split_idx + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results 