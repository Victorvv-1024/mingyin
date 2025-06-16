"""
Shared custom objects for TensorFlow models.

This module provides reusable custom layers and metrics that are needed
for loading and training sales forecasting models.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


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


def get_custom_objects():
    """Get dictionary of all custom objects for model loading"""
    return {
        'mape_metric_original_scale': mape_metric_original_scale,
        'rmse_metric_original_scale': rmse_metric_original_scale,
        'FeatureSliceLayer': FeatureSliceLayer,
        'AdamW': tf.keras.optimizers.AdamW
    } 