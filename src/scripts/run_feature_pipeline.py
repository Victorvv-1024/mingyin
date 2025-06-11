import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_pipeline import SalesFeaturePipeline

def main():
    print("=" * 80)
    print("RUNNING FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = SalesFeaturePipeline(output_dir="data/engineered")
    
    # Run pipeline
    engineered_data_path, modeling_features, rolling_splits, quality_metrics = pipeline.run_complete_pipeline(
        raw_data_dir="data/raw",
        years=[2021, 2022, 2023]
    )
    
    print("\nPipeline completed successfully!")
    print(f"Engineered data saved to: {engineered_data_path}")
    print(f"Number of modeling features: {len(modeling_features)}")
    print(f"Number of rolling splits: {len(rolling_splits)}")

if __name__ == "__main__":
    main() 