# Data Directory

This directory contains the datasets used for the MingYin sales forecasting project.

## Directory Structure

```
data/
├── raw/           # Original datasets (not in git due to size)
├── processed/     # Cleaned and preprocessed data
└── engineered/    # Feature-engineered datasets ready for training
```

## Expected Files

### `raw/`
- `sales_data.csv` - Historical sales data
- `store_info.csv` - Store metadata and characteristics
- `product_catalog.csv` - Product information

### `processed/`
- `cleaned_sales.csv` - Preprocessed sales data
- `store_features.csv` - Processed store information

### `engineered/`
- `train_features.csv` - Training dataset with 80+ engineered features
- `validation_features.csv` - Validation dataset
- `test_features.csv` - Test dataset for 2023 evaluation

## Usage

1. Place your raw data files in `raw/`
2. Run Phase 1 feature engineering: `python src/scripts/phase1_feature_engineering.py`
3. Engineered features will be saved to `engineered/`

## Note

Data files are not included in git due to size constraints. Please ensure you have the required datasets before running the pipeline. 