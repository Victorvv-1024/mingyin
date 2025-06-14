"""
Model trainer for sales forecasting.

This module provides a high-level interface for training the advanced
embedding model with proper experiment tracking and result management.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from .advanced_embedding import AdvancedEmbeddingModel
from .enhanced_model import EnhancedEmbeddingModel
from .feature_processor import FeatureProcessor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    High-level trainer for sales forecasting models.
    
    Provides a clean interface for training, evaluation, and result management
    that matches the notebook's comprehensive approach.
    """
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 random_seed: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            output_dir: Base directory for saving outputs
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output subdirectories
        self.models_dir = self.output_dir / "models"
        self.predictions_dir = self.output_dir / "predictions"
        self.reports_dir = self.output_dir / "reports"
        
        for directory in [self.models_dir, self.predictions_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        # self.model = AdvancedEmbeddingModel(random_seed=random_seed)
        self.model = EnhancedEmbeddingModel(random_seed=random_seed)
        self.training_results = {}
        self.experiment_metadata = {}
        
        logger.info(f"ModelTrainer initialized with output directory: {self.output_dir}")
    
    def train_complete_pipeline(self, 
                           df_final: pd.DataFrame,
                           features: List[str],
                           rolling_splits: List[Tuple],
                           epochs: int = 100,
                           batch_size: int = 256,
                           experiment_name: Optional[str] = None) -> Dict[str, any]:
        """
        Train the complete model pipeline on rolling splits.
        
        Args:
            df_final: Final engineered dataset
            features: List of modeling features
            rolling_splits: List of (train_df, val_df, description) tuples
            epochs: Number of training epochs
            batch_size: Training batch size
            experiment_name: Optional name for the experiment
            
        Returns:
            Dictionary with comprehensive training results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = f"sales_forecasting_experiment_{timestamp}"
        
        logger.info("=" * 80)
        logger.info(f"STARTING COMPLETE TRAINING PIPELINE: {experiment_name}")
        logger.info("=" * 80)
        
        # Store experiment metadata
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'total_samples': len(df_final),
            'total_features': len(features),
            'rolling_splits': len(rolling_splits),
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'random_seed': self.random_seed
            },
            'data_info': {
                'date_range': {
                    'start': df_final['sales_month'].min().strftime('%Y-%m-%d'),
                    'end': df_final['sales_month'].max().strftime('%Y-%m-%d')
                },
                'platforms': df_final['primary_platform'].unique().tolist(),
                'stores': df_final['store_name'].nunique(),
                'brands': df_final['brand_name'].nunique(),
                'products': df_final['product_code'].nunique()
            }
        }
        
        # Log experiment information
        self._log_experiment_info()
        
        # Validate input data
        validation_results = self._validate_input_data(df_final, features, rolling_splits)
        if not validation_results['validation_passed']:
            raise ValueError("Input data validation failed")
        
        # Train model on rolling splits
        logger.info("Training model on rolling splits...")
        logger.info(f"Models will be saved to: {self.models_dir}")
        
        # CRITICAL FIX: Pass models_dir to the training method
        training_results = self.model.train_on_rolling_splits(
            df_final=df_final,
            features=features,
            rolling_splits=rolling_splits,
            epochs=epochs,
            batch_size=batch_size,
            models_dir=str(self.models_dir)  # *** ADD THIS PARAMETER ***
        )
        
        self.training_results = training_results
        
        # Generate comprehensive results
        comprehensive_results = self._generate_comprehensive_results(
            training_results, df_final, rolling_splits, features
        )
        
        # Save all results and artifacts
        saved_files = self._save_experiment_results(
            comprehensive_results, df_final, rolling_splits
        )
        
        # Generate final summary
        final_summary = self._generate_final_summary(comprehensive_results)
        
        # Add model file information to saved_files
        model_files = {}
        for split_num, results in training_results.items():
            if results.get('saved_model_path'):
                model_files[f'model_split_{split_num}'] = results['saved_model_path']
        
        saved_files.update(model_files)
        
        logger.info("=" * 80)
        logger.info(f"TRAINING PIPELINE COMPLETED: {experiment_name}")
        logger.info("=" * 80)
        
        # Log saved models
        logger.info("MODELS SAVED:")
        for split_num, results in training_results.items():
            if results.get('saved_model_path'):
                logger.info(f"  Split {split_num}: {results.get('model_filename', 'Unknown')}")
            else:
                logger.warning(f"  Split {split_num}: ❌ SAVE FAILED")
        
        return {
            'experiment_metadata': self.experiment_metadata,
            'training_results': training_results,
            'comprehensive_results': comprehensive_results,
            'saved_files': saved_files,
            'final_summary': final_summary
        }
    
    def _log_experiment_info(self) -> None:
        """Log experiment information."""
        metadata = self.experiment_metadata
        
        logger.info(f"Experiment: {metadata['experiment_name']}")
        logger.info(f"Dataset: {metadata['total_samples']:,} samples, {metadata['total_features']} features")
        logger.info(f"Date range: {metadata['data_info']['date_range']['start']} to {metadata['data_info']['date_range']['end']}")
        logger.info(f"Platforms: {', '.join(metadata['data_info']['platforms'])}")
        logger.info(f"Stores: {metadata['data_info']['stores']:,}, Brands: {metadata['data_info']['brands']:,}")
        logger.info(f"Rolling splits: {metadata['rolling_splits']}")
        logger.info(f"Training params: {metadata['training_params']}")
    
    def _validate_input_data(self, 
                           df_final: pd.DataFrame, 
                           features: List[str],
                           rolling_splits: List[Tuple]) -> Dict[str, any]:
        """Validate input data for training."""
        logger.info("Validating input data...")
        
        validation_results = {
            'validation_passed': True,
            'issues': []
        }
        
        # Check required columns
        required_columns = ['sales_month', 'store_name', 'brand_name', 'primary_platform', 'sales_quantity_log']
        missing_columns = [col for col in required_columns if col not in df_final.columns]
        if missing_columns:
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
            validation_results['validation_passed'] = False
        
        # Check features exist
        missing_features = [f for f in features if f not in df_final.columns]
        if missing_features:
            validation_results['issues'].append(f"Missing features: {len(missing_features)} features not found in dataset")
            if len(missing_features) > len(features) * 0.1:  # More than 10% missing
                validation_results['validation_passed'] = False
        
        # Check rolling splits
        if len(rolling_splits) == 0:
            validation_results['issues'].append("No rolling splits provided")
            validation_results['validation_passed'] = False
        
        for i, (train_df, val_df, description) in enumerate(rolling_splits):
            if len(train_df) == 0 or len(val_df) == 0:
                validation_results['issues'].append(f"Split {i+1} has empty train or validation data")
                validation_results['validation_passed'] = False
        
        # Check target variable
        if 'sales_quantity_log' in df_final.columns:
            target_stats = df_final['sales_quantity_log'].describe()
            if target_stats['std'] == 0:
                validation_results['issues'].append("Target variable has zero variance")
                validation_results['validation_passed'] = False
        
        # Log validation results
        if validation_results['validation_passed']:
            logger.info("✓ Input data validation passed")
        else:
            logger.error("✗ Input data validation failed:")
            for issue in validation_results['issues']:
                logger.error(f"  - {issue}")
        
        return validation_results
    
    def _generate_comprehensive_results(self, 
                                      training_results: Dict[int, Dict],
                                      df_final: pd.DataFrame,
                                      rolling_splits: List[Tuple],
                                      features: List[str]) -> Dict[str, any]:
        """Generate comprehensive analysis of training results."""
        logger.info("Generating comprehensive results analysis...")
        
        # Performance metrics across splits
        val_mapes = [results['val_mape'] for results in training_results.values()]
        val_rmses = [results['val_rmse'] for results in training_results.values()]
        val_r2s = [results['val_r2'] for results in training_results.values()]
        
        performance_metrics = {
            'validation_mape': {
                'mean': np.mean(val_mapes),
                'std': np.std(val_mapes),
                'min': np.min(val_mapes),
                'max': np.max(val_mapes),
                'values': val_mapes
            },
            'validation_rmse': {
                'mean': np.mean(val_rmses),
                'std': np.std(val_rmses),
                'min': np.min(val_rmses),
                'max': np.max(val_rmses),
                'values': val_rmses
            },
            'validation_r2': {
                'mean': np.mean(val_r2s),
                'std': np.std(val_r2s),
                'min': np.min(val_r2s),
                'max': np.max(val_r2s),
                'values': val_r2s
            }
        }
        
        # Performance assessment
        avg_mape = performance_metrics['validation_mape']['mean']
        mape_consistency = performance_metrics['validation_mape']['std']
        
        if avg_mape <= 15:
            performance_grade = "EXCELLENT"
            performance_description = "Business-ready accuracy"
        elif avg_mape <= 20:
            performance_grade = "GOOD"
            performance_description = "Acceptable for most use cases"
        elif avg_mape <= 30:
            performance_grade = "FAIR"
            performance_description = "May need improvement"
        else:
            performance_grade = "POOR"
            performance_description = "Requires significant improvement"
        
        consistency_grade = "CONSISTENT" if mape_consistency <= 3 else "INCONSISTENT"
        
        # Training efficiency analysis
        epochs_used = [results['epochs_trained'] for results in training_results.values()]
        best_epochs = [results['best_epoch'] for results in training_results.values()]
        
        training_efficiency = {
            'average_epochs_used': np.mean(epochs_used),
            'average_best_epoch': np.mean(best_epochs),
            'early_stopping_rate': sum(1 for e in epochs_used if e < 100) / len(epochs_used),
            'training_stability': np.std(best_epochs) / np.mean(best_epochs)
        }
        
        # Feature importance analysis (basic)
        feature_importance = self._analyze_feature_importance(df_final, features)
        
        # Platform performance analysis
        platform_performance = self._analyze_platform_performance(
            training_results, rolling_splits, df_final
        )
        
        # Temporal performance analysis
        temporal_performance = self._analyze_temporal_performance(
            training_results, rolling_splits
        )
        
        return {
            'performance_metrics': performance_metrics,
            'performance_assessment': {
                'grade': performance_grade,
                'description': performance_description,
                'consistency_grade': consistency_grade,
                'average_mape': avg_mape,
                'mape_consistency': mape_consistency
            },
            'training_efficiency': training_efficiency,
            'feature_importance': feature_importance,
            'platform_performance': platform_performance,
            'temporal_performance': temporal_performance,
            'split_details': training_results
        }
    
    def _analyze_feature_importance(self, df_final: pd.DataFrame, features: List[str]) -> Dict[str, any]:
        """Analyze feature importance using correlation with target."""
        if 'sales_quantity_log' not in df_final.columns:
            return {}
        
        # Calculate correlation with target
        correlations = {}
        for feature in features:
            if feature in df_final.columns:
                corr = df_final[feature].corr(df_final['sales_quantity_log'])
                if not np.isnan(corr):
                    correlations[feature] = abs(corr)
        
        # Sort by importance
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Group by feature categories
        feature_groups = {
            'temporal': [f for f, _ in sorted_features if any(x in f for x in ['month', 'quarter', 'year', 'sin', 'cos', 'days'])],
            'lag': [f for f, _ in sorted_features if '_lag_' in f],
            'rolling': [f for f, _ in sorted_features if 'rolling_' in f],
            'promotional': [f for f, _ in sorted_features if any(x in f for x in ['promotional', 'promo'])],
            'customer_behavior': [f for f, _ in sorted_features if any(x in f for x in ['store_', 'brand_', 'market_share'])],
            'platform': [f for f, _ in sorted_features if any(x in f for x in ['platform_', 'cross_platform'])]
        }
        
        return {
            'top_features': sorted_features[:20],
            'feature_groups': feature_groups,
            'total_analyzed': len(correlations)
        }
    
    def _analyze_platform_performance(self, 
                                    training_results: Dict[int, Dict],
                                    rolling_splits: List[Tuple],
                                    df_final: pd.DataFrame) -> Dict[str, any]:
        """Analyze performance by platform."""
        platform_analysis = {}
        
        # Get all platforms
        platforms = df_final['primary_platform'].unique()
        
        for split_num, results in training_results.items():
            train_df, val_df, _ = rolling_splits[split_num - 1]
            
            # Get predictions for this split
            val_predictions = results['val_predictions']
            val_targets = val_df['sales_quantity_log'].values
            
            # Calculate MAPE by platform
            for platform in platforms:
                platform_mask = val_df['primary_platform'] == platform
                if platform_mask.sum() > 0:
                    platform_pred = val_predictions[platform_mask]
                    platform_true = val_targets[platform_mask]
                    
                    platform_mape = self.model.safe_mape_calculation(platform_true, platform_pred)
                    
                    if platform not in platform_analysis:
                        platform_analysis[platform] = []
                    platform_analysis[platform].append(platform_mape)
        
        # Aggregate platform performance
        platform_summary = {}
        for platform, mapes in platform_analysis.items():
            platform_summary[platform] = {
                'mean_mape': np.mean(mapes),
                'std_mape': np.std(mapes),
                'min_mape': np.min(mapes),
                'max_mape': np.max(mapes),
                'splits_analyzed': len(mapes)
            }
        
        return platform_summary
    
    def _analyze_temporal_performance(self, 
                                    training_results: Dict[int, Dict],
                                    rolling_splits: List[Tuple]) -> Dict[str, any]:
        """Analyze performance over time periods."""
        temporal_analysis = {}
        
        for split_num, results in training_results.items():
            train_df, val_df, description = rolling_splits[split_num - 1]
            
            val_date_range = {
                'start': val_df['sales_month'].min(),
                'end': val_df['sales_month'].max(),
                'months': val_df['sales_month'].nunique()
            }
            
            temporal_analysis[split_num] = {
                'description': description,
                'validation_mape': results['val_mape'],
                'date_range': val_date_range,
                'sample_count': len(val_df)
            }
        
        # Identify best and worst performing periods
        best_split = min(temporal_analysis.items(), key=lambda x: x[1]['validation_mape'])
        worst_split = max(temporal_analysis.items(), key=lambda x: x[1]['validation_mape'])
        
        return {
            'split_performance': temporal_analysis,
            'best_period': {
                'split': best_split[0],
                'mape': best_split[1]['validation_mape'],
                'description': best_split[1]['description']
            },
            'worst_period': {
                'split': worst_split[0],
                'mape': worst_split[1]['validation_mape'],
                'description': worst_split[1]['description']
            }
        }
    
    def _save_experiment_results(self, 
                               comprehensive_results: Dict[str, any],
                               df_final: pd.DataFrame,
                               rolling_splits: List[Tuple]) -> Dict[str, str]:
        """Save all experiment results and artifacts."""
        timestamp = self.experiment_metadata['timestamp']
        experiment_name = self.experiment_metadata['experiment_name']
        
        saved_files = {}
        
        # Save model predictions
        prediction_files = self.model.save_predictions(
            self.training_results, df_final, rolling_splits, str(self.predictions_dir)
        )
        saved_files.update(prediction_files)
        
        # Save comprehensive experiment report
        report_file = self.reports_dir / f"experiment_report_{timestamp}.txt"
        self._save_experiment_report(comprehensive_results, report_file)
        saved_files['experiment_report'] = str(report_file)
        
        # Save performance summary CSV
        summary_csv = self.reports_dir / f"performance_summary_{timestamp}.csv"
        self._save_performance_csv(comprehensive_results, summary_csv)
        saved_files['performance_csv'] = str(summary_csv)
        
        # Save metadata JSON
        import json
        metadata_file = self.reports_dir / f"experiment_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metadata_json = self._convert_to_json_serializable(self.experiment_metadata)
            json.dump(metadata_json, f, indent=2, default=str)
        saved_files['metadata'] = str(metadata_file)
        
        logger.info(f"Experiment results saved:")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
        
        return saved_files
    
    def _save_experiment_report(self, comprehensive_results: Dict[str, any], report_file: Path) -> None:
        """Save comprehensive experiment report."""
        with open(report_file, 'w') as f:
            f.write("SALES FORECASTING EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Experiment metadata
            metadata = self.experiment_metadata
            f.write("EXPERIMENT INFORMATION:\n")
            f.write(f"Name: {metadata['experiment_name']}\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Random Seed: {metadata['training_params']['random_seed']}\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write(f"Total Samples: {metadata['total_samples']:,}\n")
            f.write(f"Total Features: {metadata['total_features']}\n")
            f.write(f"Date Range: {metadata['data_info']['date_range']['start']} to {metadata['data_info']['date_range']['end']}\n")
            f.write(f"Platforms: {', '.join(metadata['data_info']['platforms'])}\n")
            f.write(f"Unique Stores: {metadata['data_info']['stores']:,}\n")
            f.write(f"Unique Brands: {metadata['data_info']['brands']:,}\n")
            f.write(f"Unique Products: {metadata['data_info']['products']:,}\n\n")
            
            # Performance assessment
            assessment = comprehensive_results['performance_assessment']
            f.write("PERFORMANCE ASSESSMENT:\n")
            f.write(f"Overall Grade: {assessment['grade']} ({assessment['description']})\n")
            f.write(f"Consistency: {assessment['consistency_grade']}\n")
            f.write(f"Average Validation MAPE: {assessment['average_mape']:.2f}%\n")
            f.write(f"MAPE Standard Deviation: {assessment['mape_consistency']:.2f}%\n\n")
            
            # Performance metrics
            metrics = comprehensive_results['performance_metrics']
            f.write("DETAILED PERFORMANCE METRICS:\n")
            f.write(f"Validation MAPE: {metrics['validation_mape']['mean']:.2f}% ± {metrics['validation_mape']['std']:.2f}%\n")
            f.write(f"  Range: {metrics['validation_mape']['min']:.2f}% - {metrics['validation_mape']['max']:.2f}%\n")
            f.write(f"Validation RMSE: {metrics['validation_rmse']['mean']:.0f} ± {metrics['validation_rmse']['std']:.0f}\n")
            f.write(f"Validation R²: {metrics['validation_r2']['mean']:.3f} ± {metrics['validation_r2']['std']:.3f}\n\n")
            
            # Training efficiency
            efficiency = comprehensive_results['training_efficiency']
            f.write("TRAINING EFFICIENCY:\n")
            f.write(f"Average Epochs Used: {efficiency['average_epochs_used']:.1f}\n")
            f.write(f"Average Best Epoch: {efficiency['average_best_epoch']:.1f}\n")
            f.write(f"Early Stopping Rate: {efficiency['early_stopping_rate']:.1%}\n")
            f.write(f"Training Stability: {efficiency['training_stability']:.3f}\n\n")
            
            # Platform performance
            if 'platform_performance' in comprehensive_results:
                f.write("PLATFORM PERFORMANCE:\n")
                for platform, perf in comprehensive_results['platform_performance'].items():
                    f.write(f"{platform}: {perf['mean_mape']:.2f}% ± {perf['std_mape']:.2f}%\n")
                f.write("\n")
            
            # Top features
            if 'feature_importance' in comprehensive_results and comprehensive_results['feature_importance']:
                f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
                for i, (feature, importance) in enumerate(comprehensive_results['feature_importance']['top_features'][:10], 1):
                    f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
                f.write("\n")
            
            # Split details
            f.write("INDIVIDUAL SPLIT PERFORMANCE:\n")
            for split_num, results in self.training_results.items():
                f.write(f"Split {split_num}: {results.get('description', 'N/A')}\n")
                f.write(f"  Validation MAPE: {results['val_mape']:.2f}%\n")
                f.write(f"  Validation RMSE: {results['val_rmse']:.0f}\n")
                f.write(f"  Validation R²: {results['val_r2']:.3f}\n")
                f.write(f"  Training Samples: {results.get('train_samples', 'N/A'):,}\n")
                f.write(f"  Validation Samples: {results.get('val_samples', 'N/A'):,}\n")
                f.write(f"  Epochs Trained: {results['epochs_trained']}\n")
                f.write(f"  Best Epoch: {results['best_epoch']}\n\n")
    
    def _save_performance_csv(self, comprehensive_results: Dict[str, any], csv_file: Path) -> None:
        """Save performance metrics as CSV."""
        performance_data = []
        
        for split_num, results in self.training_results.items():
            performance_data.append({
                'split_number': split_num,
                'description': results.get('description', 'N/A'),
                'train_samples': results.get('train_samples', 0),
                'val_samples': results.get('val_samples', 0),
                'train_mape': results['train_mape'],
                'val_mape': results['val_mape'],
                'train_rmse': results['train_rmse'],
                'val_rmse': results['val_rmse'],
                'train_r2': results['train_r2'],
                'val_r2': results['val_r2'],
                'epochs_trained': results['epochs_trained'],
                'best_epoch': results['best_epoch']
            })
        
        df_performance = pd.DataFrame(performance_data)
        df_performance.to_csv(csv_file, index=False)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_final_summary(self, comprehensive_results: Dict[str, any]) -> Dict[str, any]:
        """Generate final summary for the experiment."""
        assessment = comprehensive_results['performance_assessment']
        metrics = comprehensive_results['performance_metrics']
        
        summary = {
            'experiment_completed': True,
            'overall_grade': assessment['grade'],
            'average_validation_mape': assessment['average_mape'],
            'consistency_grade': assessment['consistency_grade'],
            'best_split_mape': metrics['validation_mape']['min'],
            'worst_split_mape': metrics['validation_mape']['max'],
            'total_splits_trained': len(self.training_results),
            'business_ready': assessment['average_mape'] <= 20,
            'recommendations': self._generate_recommendations(comprehensive_results)
        }
        
        return summary
    
    def _generate_recommendations(self, comprehensive_results: Dict[str, any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        assessment = comprehensive_results['performance_assessment']
        efficiency = comprehensive_results['training_efficiency']
        
        if assessment['average_mape'] <= 15:
            recommendations.append("✅ Model is ready for production deployment")
            recommendations.append("✅ Performance exceeds business requirements")
        elif assessment['average_mape'] <= 20:
            recommendations.append("✅ Model is suitable for business use")
            recommendations.append("📋 Consider validation with business stakeholders")
        else:
            recommendations.append("⚠️ Model performance needs improvement")
            recommendations.append("📋 Consider feature engineering or architecture changes")
        
        if assessment['consistency_grade'] == "INCONSISTENT":
            recommendations.append("⚠️ High variance across splits - check for overfitting")
            recommendations.append("📋 Consider regularization or cross-validation")
        
        if efficiency['early_stopping_rate'] > 0.8:
            recommendations.append("📋 Consider reducing number of epochs for efficiency")
        elif efficiency['early_stopping_rate'] < 0.2:
            recommendations.append("📋 Consider increasing number of epochs")
        
        if 'platform_performance' in comprehensive_results:
            platform_mapes = [perf['mean_mape'] for perf in comprehensive_results['platform_performance'].values()]
            if max(platform_mapes) - min(platform_mapes) > 10:
                recommendations.append("📋 Consider platform-specific model tuning")
        
        return recommendations
    
    def load_experiment_results(self, metadata_file: str) -> Dict[str, any]:
        """Load previously saved experiment results."""
        import json
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded experiment: {metadata['experiment_name']}")
        return metadata