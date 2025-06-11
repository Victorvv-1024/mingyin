import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Define custom metrics as standalone functions
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
    Advanced embedding-based deep learning for sales forecasting
    Clean structure without nested methods
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.encoders = {}
        self.scalers = {}
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        print("=" * 70)
        print("ADVANCED EMBEDDING-BASED FRAMEWORK INITIALIZED")
        print("=" * 70)
    
    def safe_mape_calculation(self, y_true, y_pred):
        """Safe MAPE calculation with proper error handling"""
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        y_pred_orig = np.clip(y_pred_orig, 0.1, 1e6)
        y_true_orig = np.clip(y_true_orig, 0.1, 1e6)
        epsilon = 1.0
        ape = np.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)
        mape = np.mean(ape) * 100
        return min(mape, 1000.0)
    
    def categorize_features_for_embeddings(self, df, features):
        """Analyze and categorize features for embedding strategies"""
        print("=== ANALYZING FEATURES FOR EMBEDDING STRATEGIES ===")
        
        feature_categories = {
            'temporal': [],
            'numerical_continuous': [],
            'numerical_discrete': [],
            'binary': [],
            'interactions': []
        }
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            dtype = df[feature].dtype
            unique_count = df[feature].nunique()
            
            if any(x in feature for x in ['month', 'quarter', 'day']):
                if 'sin' in feature or 'cos' in feature:
                    feature_categories['numerical_discrete'].append(feature)
                else:
                    feature_categories['temporal'].append(feature)
            elif any(x in feature for x in ['lag_', 'rolling_', 'sales_', 'momentum', 'volatility']):
                feature_categories['numerical_continuous'].append(feature)
            elif 'store_type_' in feature or dtype == 'bool':
                feature_categories['binary'].append(feature)
            elif 'interaction' in feature:
                feature_categories['interactions'].append(feature)
            else:
                if dtype in ['int64', 'float64', 'int32', 'float32']:
                    if unique_count < 20:
                        feature_categories['numerical_discrete'].append(feature)
                    else:
                        feature_categories['numerical_continuous'].append(feature)
        
        print("Feature categories:")
        for category, feat_list in feature_categories.items():
            if feat_list:
                print(f"  {category}: {len(feat_list)} features")
        
        return feature_categories 