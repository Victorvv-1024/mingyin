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
- `sales_forecast_engineered_dataset_{date}.pkl` - Complete dataset with 316 engineered features and time splits
- `modeling_features_{date}.txt` - Detailed feature list and categorization (316 features across 9 categories)
- `feature_metadata_{date}.json` - Feature engineering metadata and statistics
- `rolling_splits_{date}.pkl` - Time-based validation splits for cross-validation

## 🎯 Feature Engineering Overview

The pipeline transforms raw Excel data into **316 sophisticated features** across 9 major categories:

### **Feature Categories Summary**
- **🕒 Temporal Features (24)**: Calendar patterns, Chinese market events, cyclical encodings
- **🎯 Promotional Features (12)**: Holiday seasons, promotional events, platform-specific campaigns  
- **📊 Lag Features (15)**: 1-12 month historical patterns for sales, amount, price
- **📈 Rolling Statistics (24)**: 3, 6, 12-month statistical measures (mean, std, min, max, CV, quartiles)
- **⚡ Momentum Features (16)**: YoY trends, sales acceleration, momentum classification
- **🏪 Customer Behavior (124)**: Store analytics, brand performance, platform dynamics, Chinese liquor brands
- **🏆 Store Categorization (4)**: Competitive positioning (leader, major, minor, niche)
- **🔄 Platform Dynamics (19)**: Cross-platform competition, loyalty patterns, experience levels
- **🚨 Spike Detection (5)**: Anomaly detection, deviation analysis, spike propensity

### **Chinese Market Specialization**
- **Chinese Calendar Events**: Spring Festival, Singles' Day, Mid-year shopping festivals
- **Platform Integration**: Douyin, JD, Tmall-specific features and interactions
- **Chinese Liquor Brands**: Specialized tracking for major brands (茅台, 五粮液, 西凤, etc.)
- **Market Dynamics**: Chinese e-commerce competitive landscape analysis

## Usage

1. Place your raw data files in `raw/`
2. Run Phase 1 feature engineering: `python src/scripts/phase1_feature_engineering.py`
3. Engineered features will be saved to `engineered/`
4. Review `modeling_features_{date}.txt` for detailed feature breakdown

## Note

Data files are not included in git due to size constraints. Please ensure you have the required datasets before running the pipeline. 