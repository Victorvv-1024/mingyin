import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PyTorchAdvancedEmbeddingModel, PyTorchModelTrainer
from src.data.data_loader import load_processed_data
from src.utils.metrics import safe_mape_calculation

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

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("PYTORCH MODEL EVALUATION")
    print("=" * 80)
    
    # Load processed data
    print("\nLoading processed data...")
    engineered_data_path = "data/engineered/sales_forecast_engineered_dataset_20250611_100158.pkl"
    df = load_processed_data(engineered_data_path)
    
    # Define features
    feature_columns = [
        'month', 'quarter', 'year',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'yoy_change', 'mom_change'
    ]
    
    # Initialize model and trainer
    model = PyTorchAdvancedEmbeddingModel()
    trainer = PyTorchModelTrainer(model)
    
    # Create a single train/val split for testing
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:].copy()
    
    rolling_splits = [(train_data, val_data, "Test Split")]
    
    # Train model
    print("\nTraining model...")
    results = trainer.train(df, feature_columns, rolling_splits)
    
    # Evaluate on validation set
    print("\nEvaluating model...")
    X_val, y_val = trainer.prepare_embedding_features(val_data, model.categorize_features_for_embeddings(df, feature_columns), is_training=False)
    
    # Load best model
    best_model_path = os.path.join("checkpoints", results['split_1']['saved_files']['best_model'])
    model_state = torch.load(best_model_path)
    
    # Create model instance with same architecture
    temporal_vocab_sizes = [12, 4] + [101] * (X_val['temporal'].shape[1] - 2)
    continuous_vocab_sizes = [52] * X_val['continuous'].shape[1]
    direct_features_dim = X_val['direct'].shape[1]
    
    model_instance = AdvancedEmbeddingModel(
        temporal_vocab_sizes=temporal_vocab_sizes,
        continuous_vocab_sizes=continuous_vocab_sizes,
        direct_features_dim=direct_features_dim
    )
    model_instance.load_state_dict(model_state)
    model_instance.eval()
    
    # Prepare tensors
    X_val_tensors = {
        key: torch.FloatTensor(array).to(model.device)
        for key, array in X_val.items()
    }
    
    # Make predictions
    with torch.no_grad():
        val_pred_log = model_instance(
            X_val_tensors.get('temporal'),
            X_val_tensors.get('continuous'),
            X_val_tensors.get('direct')
        ).cpu().numpy().flatten()
    
    # Convert predictions back to original scale
    val_pred_orig = np.expm1(val_pred_log)
    val_true_orig = np.expm1(y_val)
    
    # Calculate metrics
    mape = safe_mape_calculation(y_val, val_pred_log)
    rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
    r2 = r2_score(val_true_orig, val_pred_orig)
    mae = np.mean(np.abs(val_true_orig - val_pred_orig))
    
    # Print results
    print("\nTest Results:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/prediction_plot_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plot_predictions(val_true_orig, val_pred_orig, "Actual vs Predicted Sales", plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Save results to file
    results_path = f"results/test_results_{timestamp}.txt"
    with open(results_path, "w") as f:
        f.write("PyTorch Model Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"MAE: {mae:.2f}\n")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main() 