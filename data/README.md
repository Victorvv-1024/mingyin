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

### `raw/` - Historical sales data in Excel form
- `2021.xlsx` 
- `2022.xlsx`
- `2023.xlsx` 

### `processed/`
- `processed_data_{date}.pkl` - Processed data
- `processing_metadata_{date}.pkl` - Metadata

### `engineered/`
- `sales_forecast_engineered_dataset_{date}.pkl` - Entire dataset with 300+ engineered features with time splits

## Usage

1. Place your raw data files in `raw/`
2. Run Phase 1 feature engineering: `python src/scripts/phase1_feature_engineering.py`
3. Engineered features will be saved to `engineered/`

## Note

Data files are not included in git due to size constraints. Please ensure you have the required datasets before running the pipeline. 