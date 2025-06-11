import os
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from data.preprocessing import SalesDataProcessor
from data.dataset import DatasetManager
from models.advanced_embedding import AdvancedEmbeddingModel
from models.trainer import ModelTrainer
from evaluation.test_evaluator import TestEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run sales forecasting pipeline')
    
    parser.add_argument('--raw_data_dir', type=str, default='data/raw',
                      help='Directory containing raw data files')
    parser.add_argument('--processed_data_dir', type=str, default='data/processed',
                      help='Directory to save processed data')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--split_strategy', type=str, default='split_4_test',
                      choices=['split_4_test', 'split_3_replication'],
                      help='Strategy for splitting data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Run complete pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting pipeline run at {timestamp}")
    
    try:
        # 1. Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        processor = SalesDataProcessor(args.raw_data_dir, args.processed_data_dir)
        engineered_data_path, quality_metrics = processor.process()
        
        # 2. Dataset Creation
        logger.info("Step 2: Dataset Creation")
        dataset_manager = DatasetManager(
            engineered_data_path,
            split_strategy=args.split_strategy,
            random_seed=args.random_seed
        )
        
        # Get features for training
        features = dataset_manager.get_feature_columns()
        
        # Create rolling time series splits for robust training
        rolling_splits = dataset_manager.create_rolling_time_series_splits()
        
        # 3. Model Training
        logger.info("Step 3: Model Training")
        model = AdvancedEmbeddingModel()
        trainer = ModelTrainer(model)
        training_results = trainer.train(
            dataset_manager.data,
            features,
            rolling_splits
        )
        
        # 4. Test Evaluation
        logger.info("Step 4: Test Evaluation")
        evaluator = TestEvaluator(
            output_dir=args.results_dir,
            split_strategy=args.split_strategy
        )
        
        # Get train and test data for evaluation (use the last rolling split)
        if rolling_splits:
            train_data, test_data, _ = rolling_splits[-1]  # Use last split for evaluation
        else:
            # Fallback to simple split if no rolling splits available
            train_dataset, test_dataset = dataset_manager.create_datasets()
            train_data = train_dataset.data
            test_data = test_dataset.data
        
        # Run evaluation
        evaluation_results = evaluator.evaluate(
            train_data,
            test_data,
            expected_model_mape=training_results.get('best_val_mape', 5.0)
        )
        
        # 5. Save final results
        logger.info("Step 5: Saving Results")
        results_summary = {
            'timestamp': timestamp,
            'split_strategy': args.split_strategy,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_config': model.get_config()
        }
        
        # Save results to file
        results_file = Path(args.results_dir) / f'pipeline_results_{timestamp}.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Pipeline completed successfully. Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 