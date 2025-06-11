import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use PyTorch version instead of TensorFlow
from src.models.pytorch_advanced_embedding import PyTorchAdvancedEmbeddingModel, PyTorchModelTrainer
from src.models.evaluator import ModelEvaluator
from src.data.feature_pipeline import load_engineered_dataset

def main():
    # Use the specific engineered dataset path
    dataset_path = '/Users/victor/Library/CloudStorage/Dropbox/PolyU Projects/MingYin/data/engineered/sales_forecast_engineered_dataset_20250611_100158.pkl'
    print(f"📁 Loading: {os.path.basename(dataset_path)}")
    
    print("📊 Loading pre-engineered dataset...")
    df_final, features, rolling_splits, metadata = load_engineered_dataset(dataset_path)
    
    print(f"✅ Dataset loaded successfully:")
    print(f"  📊 Records: {len(df_final):,}")
    print(f"  🎯 Features: {len(features)}")
    print(f"  🔄 Rolling splits: {len(rolling_splits)}")
    print(f"  📅 Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    print(f"  🏪 Platforms: {metadata['platforms']}")
    print(f"  💾 File size: {os.path.getsize(dataset_path) / (1024**3):.1f} GB")
    
    # Initialize the PyTorch model framework
    print("\n🚀 Initializing PyTorch advanced embedding framework...")
    model = PyTorchAdvancedEmbeddingModel()
    
    # Initialize trainer with checkpointing
    trainer = PyTorchModelTrainer(model)
    print(f"💾 Checkpointing: Every 10 epochs (keeping last 3)")
    print(f"🌟 Best models: Saved automatically")
    print(f"📁 Location: checkpoints/")
    
    # Train the model with pre-engineered features
    print("\n🏋️ Starting model training with pre-engineered features...")
    print("=" * 80)
    results = trainer.train(df_final, features, rolling_splits)
    
    # Evaluate and analyze results
    print("\n📈 Analyzing results...")
    evaluator = ModelEvaluator(model)
    evaluator.print_final_results(results)
    
    print("\n✅ Training and evaluation complete!")
    print(f"📁 All models saved in: checkpoints/")

if __name__ == "__main__":
    main() 