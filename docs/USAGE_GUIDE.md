# Sales Forecasting Pipeline - Complete Usage Guide

This guide provides comprehensive documentation for using the refactored sales forecasting pipeline that replaces the `full_data_prediction.ipynb` notebook with production-ready, modular Python code.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Component Overview](#component-overview)
4. [Detailed Usage](#detailed-usage)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Quick Start

### 1. Run Complete Pipeline (Recommended)

The easiest way to get started is using the complete pipeline script:

```bash
# Basic usage - process 2021 and 2022 data
python scripts/run_complete_pipeline.py --data-dir data/raw --output-dir outputs

# Advanced usage with custom parameters
python scripts/run_complete_pipeline.py \
    --data-dir data/raw \
    --output-dir outputs \
    --years 2021 2022 2023 \
    --epochs 100 \
    --batch-size 512 \
    --experiment-name "production_model_v1"
```

### 2. Step-by-Step Usage

For more control, you can run each component separately:

```python
from src.data.feature_pipeline import SalesFeaturePipeline
from src.models.trainer import ModelTrainer

# Step 1: Feature Engineering
pipeline = SalesFeaturePipeline(output_dir="data/engineered")
engineered_data_path, features, rolling_splits, metadata = pipeline.run_complete_pipeline(
    raw_data_dir="data/raw",
    years=[2021, 2022]
)

# Step 2: Model Training
trainer = ModelTrainer(output_dir="outputs", random_seed=42)
df_final, _, _, _ = pipeline.load_engineered_dataset(engineered_data_path)

results = trainer.train_complete_pipeline(
    df_final=df_final,
    features=features,
    rolling_splits=rolling_splits,
    epochs=100,
    batch_size=512
)
```

## Pipeline Architecture

The refactored pipeline consists of two main phases:

### Phase 1: Feature Engineering Pipeline
```
Raw Excel Files → Data Processing → Feature Engineering → Engineered Dataset
     ↓                    ↓                  ↓                    ↓
  2021.xlsx         Data Cleaning      80+ Features        Pickle + CSV
  2022.xlsx         Platform Mapping   Rolling Splits      Metadata
  2023.xlsx         Quality Checks     Validation          Reports
```

### Phase 2: Model Training Pipeline
```
Engineered Dataset → Feature Processing → Model Training → Results & Analysis
       ↓                     ↓                 ↓                ↓
   Pickle File         Multi-input Prep    Neural Network    Predictions
   Features List       Embedding Prep      Rolling Splits    Performance
   Rolling Splits      Scaling/Encoding    Callbacks         Reports
```

## Component Overview

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `SalesFeaturePipeline` | Orchestrates feature engineering | 6-step process, 80+ features, validation |
| `TemporalFeatureEngineer` | Time-based features | Chinese calendar, cyclical, lag, rolling |
| `CustomerBehaviorFeatureEngineer` | Behavioral analytics | Store metrics, loyalty, market share |
| `StoreCategorization` | Store classification | Chinese patterns, quality scoring |
| `PlatformDynamicsEngineer` | Cross-platform analysis | Competition, loyalty, seasonal effects |
| `FeatureProcessor` | Deep learning prep | Multi-input categorization, embeddings |
| `AdvancedEmbeddingModel` | Neural network | Multi-head attention, embeddings, residual |
| `ModelTrainer` | Training orchestration | Experiment tracking, comprehensive analysis |

### Feature Categories Created

1. **Temporal Features (20+ features)**
   - Basic: month, quarter, year, day_of_year
   - Cyclical: sin/cos transformations
   - Chinese promotional calendar integration
   - Year-over-year comparisons

2. **Historical Features (25+ features)**
   - Lag features: [1, 2, 3, 6, 12] months
   - Rolling statistics: [3, 6, 12] month windows
   - Momentum and acceleration indicators

3. **Customer Behavior Features (15+ features)**
   - Store consistency and volatility metrics
   - Brand diversity and market share
   - Customer loyalty indicators

4. **Store & Platform Features (20+ features)**
   - Store type classification (Chinese patterns)
   - Quality and trust indicators
   - Cross-platform competitive dynamics

5. **Interaction Features (10+ features)**
   - Seasonal-brand interactions
   - Platform-brand combinations
   - Spike detection and anomaly features

## Detailed Usage

### 1. Feature Engineering Only

When you only need to process raw data and create features:

```python
from src.data.feature_pipeline import SalesFeaturePipeline

# Initialize pipeline
pipeline = SalesFeaturePipeline(output_dir="data/engineered")

# Run feature engineering
engineered_data_path, modeling_features, rolling_splits, metadata = pipeline.run_complete_pipeline(
    raw_data_dir="data/raw",
    years=[2021, 2022, 2023]
)

print(f"Features created: {len(modeling_features)}")
print(f"Dataset saved: {engineered_data_path}")
```

### 2. Using Existing Engineered Data

When you have previously processed data:

```python
from src.data.feature_pipeline import SalesFeaturePipeline
from src.models.trainer import ModelTrainer

# Load existing engineered dataset
pipeline = SalesFeaturePipeline()
df_final, features, rolling_splits, metadata = pipeline.load_engineered_dataset(
    "data/engineered/sales_forecast_engineered_dataset_20250611_120000.pkl"
)

# Train model
trainer = ModelTrainer(output_dir="outputs")
results = trainer.train_complete_pipeline(
    df_final=df_final,
    features=features,
    rolling_splits=rolling_splits
)
```

### 3. Custom Model Training

For advanced model training with custom parameters:

```python
from src.models.advanced_embedding import AdvancedEmbeddingModel

# Initialize model with custom settings
model = AdvancedEmbeddingModel(random_seed=123)

# Train on rolling splits with custom parameters
results = model.train_on_rolling_splits(
    df_final=df_final,
    features=features,
    rolling_splits=rolling_splits,
    epochs=150,
    batch_size=256
)
```

### 4. Feature Analysis and Exploration

To analyze the engineered features:

```python
import pandas as pd
from src.data.feature_pipeline import SalesFeaturePipeline

# Load engineered data
pipeline = SalesFeaturePipeline()
df_final, features, _, _ = pipeline.load_engineered_dataset("path_to_dataset.pkl")

# Analyze feature importance (correlation with target)
target = 'sales_quantity_log'
feature_importance = {}
for feature in features:
    corr = df_final[feature].corr(df_final[target])
    if not pd.isna(corr):
        feature_importance[feature] = abs(corr)

# Sort by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("Top 10 features:")
for feature, importance in sorted_features[:10]:
    print(f"  {feature}: {importance:.4f}")
```

## Configuration

### Environment Setup

1. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate sales_forecasting
```

2. **Install package:**
```bash
pip install -e .
```

### Data Directory Structure

Organize your data as follows:
```
project_root/
├── data/
│   ├── raw/                    # Raw Excel files
│   │   ├── 2021.xlsx
│   │   ├── 2022.xlsx
│   │   └── 2023.xlsx
│   ├── processed/              # Intermediate processed data
│   └── engineered/             # Feature-engineered datasets
├── outputs/                    # Model outputs
│   ├── models/                 # Trained model files
│   ├── predictions/            # Prediction results
│   └── reports/                # Analysis reports
└── src/                        # Source code
```

### Feature Engineering Configuration

Customize feature engineering in `src/config/feature_config.yaml`:

```yaml
temporal_features:
  lag_features:
    enable: true
    lag_periods: [1, 2, 3, 6, 12]
  rolling_features:
    enable: true
    windows: [3, 6, 12]
    statistics: [mean, std, min, max]

customer_behavior:
  store_behavior:
    enable: true
    consistency_metrics: true
    volatility_analysis: true
```

## Best Practices

### 1. Data Preparation

- **Ensure data quality**: Run data validation before feature engineering
- **Consistent naming**: Use standardized Excel file names (YYYY.xlsx)
- **Complete data**: Include all required columns (sales_month, store_name, brand_name, etc.)

### 2. Feature Engineering

- **Incremental approach**: Start with basic features, then add advanced ones
- **Validation**: Always validate features after engineering
- **Documentation**: Keep track of feature engineering decisions

### 3. Model Training

- **Reproducibility**: Always set random seeds
- **Monitoring**: Track training metrics and early stopping
- **Validation**: Use proper time series splits (no data leakage)

### 4. Performance Optimization

- **Batch size**: Start with 512, adjust based on memory
- **Epochs**: Use early stopping, typically 50-150 epochs sufficient
- **Features**: Remove low-importance features if overfitting

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
**Problem**: Out of memory during training
**Solutions**:
- Reduce batch size: `--batch-size 256`
- Use fewer features (remove low-importance ones)
- Process fewer years at once

#### 2. Data Loading Errors
**Problem**: Excel files not found or corrupted
**Solutions**:
- Check file paths and names (must be YYYY.xlsx)
- Verify Excel file format and structure
- Check for required columns

#### 3. Feature Engineering Failures
**Problem**: Feature engineering pipeline fails
**Solutions**:
- Check data quality and missing values
- Ensure proper datetime formats
- Validate platform names (should be JD, Tmall, Douyin)

#### 4. Poor Model Performance
**Problem**: MAPE > 30%
**Solutions**:
- Check for data leakage in time series splits
- Increase training epochs
- Add more relevant features
- Check data quality and outliers

#### 5. Inconsistent Results
**Problem**: Different results across runs
**Solutions**:
- Set random seed: `--random-seed 42`
- Ensure deterministic operations
- Check for data shuffling issues

### Performance Expectations

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Validation MAPE | ≤15% | 15-20% | 20-30% | >30% |
| Consistency (Std) | ≤3% | 3-5% | 5-8% | >8% |
| Training Time | <30min | 30-60min | 1-2h | >2h |

## API Reference

### SalesFeaturePipeline

```python
class SalesFeaturePipeline:
    def __init__(self, output_dir: str = "data/engineered")
    
    def run_complete_pipeline(self, 
                            raw_data_dir: str,
                            years: List[int] = [2021, 2022, 2023]) -> Tuple[str, List[str], List[Tuple], Dict]
    
    def load_engineered_dataset(self, pickle_filepath: str) -> Tuple[pd.DataFrame, List[str], List[Tuple], Dict]
```

### ModelTrainer

```python
class ModelTrainer:
    def __init__(self, output_dir: str = "outputs", random_seed: int = 42)
    
    def train_complete_pipeline(self, 
                               df_final: pd.DataFrame,
                               features: List[str],
                               rolling_splits: List[Tuple],
                               epochs: int = 100,
                               batch_size: int = 512,
                               experiment_name: Optional[str] = None) -> Dict[str, any]
```

### AdvancedEmbeddingModel

```python
class AdvancedEmbeddingModel:
    def __init__(self, random_seed: int = 42)
    
    def train_on_rolling_splits(self, 
                               df_final: pd.DataFrame,
                               features: List[str],
                               rolling_splits: List[Tuple],
                               epochs: int = 100,
                               batch_size: int = 512) -> Dict[int, Dict]
```

## Command Line Interface

### Complete Pipeline Script

```bash
python scripts/run_complete_pipeline.py [OPTIONS]

Options:
  --data-dir TEXT              Directory containing raw Excel files
  --output-dir TEXT            Base directory for saving outputs
  --years INTEGER [INTEGER...]  Years to process
  --epochs INTEGER             Number of training epochs
  --batch-size INTEGER         Training batch size
  --experiment-name TEXT       Name for the experiment
  --random-seed INTEGER        Random seed for reproducibility
  --log-level [DEBUG|INFO|WARNING|ERROR]  Logging level
  --skip-feature-engineering   Skip feature engineering step
  --engineered-dataset TEXT    Path to existing engineered dataset
  --help                       Show help message
```

### Usage Examples Script

```bash
python examples/usage_examples.py [OPTIONS]

Options:
  --example {1,2,3,4,5}        Run specific example
  --all                        Run all examples
  --help                       Show help message
```

## Output Files

The pipeline generates comprehensive outputs:

### Feature Engineering Outputs
- `sales_forecast_engineered_dataset_YYYYMMDD_HHMMSS.pkl` - Main dataset
- `sales_forecast_engineered_data_YYYYMMDD_HHMMSS.csv` - CSV version
- `modeling_features_YYYYMMDD_HHMMSS.txt` - Feature list
- `dataset_metadata_YYYYMMDD_HHMMSS.txt` - Metadata summary

### Model Training Outputs
- `best_model_split_N_YYYYMMDD_HHMMSS.h5` - Trained models
- `detailed_predictions_split_N_YYYYMMDD_HHMMSS.csv` - Detailed predictions
- `experiment_report_YYYYMMDD_HHMMSS.txt` - Comprehensive report
- `performance_summary_YYYYMMDD_HHMMSS.csv` - Performance metrics
- `experiment_metadata_YYYYMMDD_HHMMSS.json` - Experiment metadata

## Next Steps

1. **Start with examples**: Run `python examples/usage_examples.py --example 1`
2. **Customize for your data**: Modify paths and parameters
3. **Monitor performance**: Check validation MAPE and consistency
4. **Iterate and improve**: Add features or tune hyperparameters based on results
5. **Deploy to production**: Use trained models for forecasting

For additional support, refer to the source code documentation and example scripts.