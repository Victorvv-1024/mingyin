import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pytorch_advanced_embedding import (
    PyTorchAdvancedEmbeddingModel, 
    PyTorchModelTrainer,
    AdvancedEmbeddingModel
)
from src.utils.metrics import safe_mape_calculation
from src.data.feature_pipeline import load_engineered_dataset

def plot_predictions(y_true, y_pred, title, save_path):
    """Plot actual vs predicted values with regression line"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(metrics_df, save_path):
    """Plot comparison of metrics across different splits"""
    plt.figure(figsize=(12, 6))
    
    # Plot MAPE
    plt.subplot(1, 2, 1)
    sns.barplot(x='Split', y='MAPE', data=metrics_df)
    plt.title('MAPE Comparison')
    plt.xticks(rotation=45)
    
    # Plot R²
    plt.subplot(1, 2, 2)
    sns.barplot(x='Split', y='R²', data=metrics_df)
    plt.title('R² Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("PYTORCH MODEL EVALUATION ON 2023 DATA")
    print("=" * 80)
    
    # Load processed data
    print("\nLoading processed data...")
    engineered_data_path = "data/engineered/sales_forecast_engineered_dataset_20250611_100158.pkl"
    df, features, rolling_splits, metadata = load_engineered_dataset(engineered_data_path)
    
    # Print column names for debugging
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Define features
    feature_columns = [
        'month', 'quarter', 'year',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_6', 'sales_lag_12',
        'sales_rolling_mean_3', 'sales_rolling_mean_6', 'sales_rolling_mean_12',
        'sales_rolling_std_3', 'sales_rolling_std_6', 'sales_rolling_std_12',
        'sales_pct_change_1', 'sales_pct_change_3', 'sales_pct_change_6',
        'sales_momentum_short', 'sales_momentum_long',
        'sales_acceleration', 'sales_volatility_ratio',
        'is_extreme_spike', 'is_major_spike',
        'historical_extreme_spikes', 'historical_major_spikes',
        'spike_propensity', 'deviation_from_rolling_mean',
        'rolling_z_score', 'store_sales_cv', 'store_sales_range',
        'avg_revenue_per_transaction', 'brand_diversity', 'product_diversity',
        'price_premium_index', 'store_size_category', 'brand_market_share',
        'monthly_competing_brands', 'monthly_competing_stores',
        'monthly_platform_total_sales', 'monthly_platform_avg_sales',
        'monthly_platform_transactions', 'is_multi_platform_brand',
        'brand_platform_preference_score', 'brand_seasonal_index',
        'brand_promotional_effectiveness', 'brand_seasonal_promo_interaction',
        'platform_brand_seasonal_interaction', 'trend_seasonal_interaction',
        'store_type_sales_index', 'store_type_price_index',
        'store_type_Flagship', 'store_type_Mall Flagship',
        'store_type_Mall Store', 'store_type_Official Flagship',
        'store_type_Other', 'store_type_Platform Direct',
        'store_type_Specialty Store', 'store_type_Supermarket'
    ]
    
    # Split definitions
    split_definitions = {
        'Split 1': 'Trained on 2021 data',
        'Split 2': 'Trained on 2021 + 2022 Q1 data',
        'Split 3': 'Trained on 2021 + 2022 Q1-Q2 data',
        'Split 4': 'Trained on 2021 + 2022 Q1-Q3 data',
        'Split 5': 'Trained on 2021 + 2022 full year data'
    }
    
    # Initialize model and trainer
    model = PyTorchAdvancedEmbeddingModel()
    trainer = PyTorchModelTrainer(model)
    
    # Initialize results storage
    metrics_list = []
    predictions = {}
    
    # Test each best model
    for split_num in range(1, 6):
        print(f"\nTesting Split {split_num}: {split_definitions[f'Split {split_num}']}")
        
        # Get test data from rolling splits
        train_data, test_data, split_name = rolling_splits[split_num - 1]
        print(f"Test data size: {len(test_data)} records")
        
        # Prepare test data
        X_test, y_test = trainer.prepare_embedding_features(
            test_data, 
            model.categorize_features_for_embeddings(df, feature_columns),
            is_training=False
        )
        
        # Load best model
        best_model_path = f"checkpoints/best_model_split_{split_num}_epoch_*.pth"
        model_files = glob.glob(best_model_path)
        if not model_files:
            print(f"❌ No model found for Split {split_num}")
            continue
            
        best_model_path = model_files[0]
        model_state = torch.load(best_model_path)
        
        # Create model instance
        temporal_vocab_sizes = [12, 4] + [101] * (X_test['temporal'].shape[1] - 2)
        continuous_vocab_sizes = [52] * X_test['continuous'].shape[1]
        direct_features_dim = X_test['direct'].shape[1]
        
        model_instance = AdvancedEmbeddingModel(
            temporal_vocab_sizes=temporal_vocab_sizes,
            continuous_vocab_sizes=continuous_vocab_sizes,
            direct_features_dim=direct_features_dim
        ).to(model.device)
        model_instance.load_state_dict(model_state)
        model_instance.eval()
        
        # Prepare tensors
        X_test_tensors = {
            key: torch.FloatTensor(array).to(model.device)
            for key, array in X_test.items()
        }
        
        # Make predictions
        with torch.no_grad():
            test_pred_log = model_instance(
                X_test_tensors.get('temporal'),
                X_test_tensors.get('continuous'),
                X_test_tensors.get('direct')
            ).cpu().numpy().flatten()
        
        # Convert predictions back to original scale
        test_pred_orig = np.expm1(test_pred_log)
        test_true_orig = np.expm1(y_test)
        
        # Store predictions
        predictions[f'Split {split_num}'] = test_pred_orig
        
        # Calculate metrics
        mape = safe_mape_calculation(y_test, test_pred_log)
        rmse = np.sqrt(mean_squared_error(test_true_orig, test_pred_orig))
        r2 = r2_score(test_true_orig, test_pred_orig)
        mae = np.mean(np.abs(test_true_orig - test_pred_orig))
        
        # Store metrics
        metrics_list.append({
            'Split': f'Split {split_num}',
            'Description': split_definitions[f'Split {split_num}'],
            'MAPE': mape,
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae
        })
        
        # Print results
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        
        # Plot predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/prediction_plot_split_{split_num}_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        plot_predictions(
            test_true_orig, 
            test_pred_orig, 
            f"Actual vs Predicted Sales (Split {split_num})\n{split_definitions[f'Split {split_num}']}", 
            plot_path
        )
        print(f"Plot saved to: {plot_path}")
    
    # Create metrics comparison plot
    metrics_df = pd.DataFrame(metrics_list)
    metrics_plot_path = f"results/metrics_comparison_{timestamp}.png"
    plot_metrics_comparison(metrics_df, metrics_plot_path)
    print(f"\nMetrics comparison plot saved to: {metrics_plot_path}")
    
    # Save results to file
    results_path = f"results/test_results_{timestamp}.txt"
    with open(results_path, "w") as f:
        f.write("PyTorch Model Test Results on 2023 Data\n")
        f.write("=" * 60 + "\n\n")
        
        for split_num in range(1, 6):
            f.write(f"Split {split_num}: {split_definitions[f'Split {split_num}']}\n")
            f.write("-" * 60 + "\n")
            metrics = metrics_df[metrics_df['Split'] == f'Split {split_num}'].iloc[0]
            f.write(f"MAPE: {metrics['MAPE']:.2f}%\n")
            f.write(f"RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"R²: {metrics['R²']:.4f}\n")
            f.write(f"MAE: {metrics['MAE']:.2f}\n\n")
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main() 