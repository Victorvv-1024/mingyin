# Sales Forecasting for Chinese E-commerce Platforms

A deep learning project for predicting sales across major Chinese e-commerce platforms (Douyin, JD, Tmall) using advanced feature engineering and neural networks.

## Project Overview

This project implements sophisticated time series forecasting models to predict sales quantities across different:
- **Platforms**: Douyin, JD, Tmall
- **Time periods**: 2021-2023 data
- **Granularity**: Monthly sales by store, brand, and platform

## Key Features

- 🔄 **Advanced Feature Engineering**: 80+ engineered features including temporal, seasonal, behavioral, and competitive dynamics
- 📊 **Rolling Time Series Validation**: 4-split validation across all seasons for robust model evaluation
- 🏪 **Store Categorization**: Automatic classification of Chinese e-commerce store types
- 📈 **Seasonal Intelligence**: Built-in understanding of Chinese e-commerce promotional periods
- 🔍 **Spike Detection**: Automated detection and handling of sales anomalies

## Repository Structure

```
sales_forecasting/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── preprocessing.py      # Data cleaning and initial processing
│   │   ├── feature_engineering.py  # Advanced feature creation
│   │   └── loader.py            # Data loading utilities
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utility functions
│   └── config/                   # Configuration management
├── data/                         # Data storage
│   ├── raw/                      # Original Excel/CSV files
│   ├── processed/                # Cleaned data
│   └── engineered/               # Feature-engineered datasets
├── notebooks/                    # Jupyter notebooks for exploration
├── scripts/                      # Standalone scripts
├── tests/                        # Unit tests
├── outputs/                      # Model outputs and results
│   ├── models/                   # Trained model artifacts
│   ├── predictions/              # Prediction files
│   └── reports/                  # Analysis reports
└── docs/                         # Documentation
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sales_forecasting
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate sales_forecasting
```

3. Install the package:
```bash
pip install -e .
```

### Usage

#### Data Processing
```python
from src.data.loader import SalesDataLoader
from src.data.preprocessing import SalesPreprocessor
from src.data.feature_engineering import FeatureEngineer

# Load and process data
loader = SalesDataLoader()
data = loader.load_raw_data()

# Preprocess
preprocessor = SalesPreprocessor()
clean_data = preprocessor.process(data)

# Feature engineering
engineer = FeatureEngineer()
engineered_data = engineer.create_all_features(clean_data)
```

#### Model Training
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(config_path='src/config/model_config.yaml')
model = trainer.train(engineered_data)
```

## Data Description

### Raw Data Sources
- **2021.xlsx, 2022.xlsx, 2023.xlsx**: Original sales data
- **Platform-specific CSVs**: Processed data by platform

### Key Variables
- `sales_quantity`: Target variable for prediction
- `sales_amount`: Revenue in RMB
- `sales_month`: Monthly timestamp
- `primary_platform`: E-commerce platform
- `store_name`: Store identifier
- `brand_name`: Product brand
- `product_code`: Product identifier

### Engineered Features (80+ features)
- **Temporal**: Month, quarter, seasonal cycles, promotional periods
- **Lag Features**: 1, 2, 3, 6, 12-month historical values
- **Rolling Statistics**: 3, 6, 12-month windows (mean, std, min, max)
- **Momentum**: Growth rates, acceleration, trend indicators
- **Store Behavior**: Consistency, volatility, diversity metrics
- **Cross-Platform**: Competitive dynamics, market share
- **Seasonal Interactions**: Brand-season, platform-season effects
- **Spike Detection**: Outlier identification and propensity scoring

## Model Performance

Current models achieve:
- **Validation MAPE**: <20% across all seasonal splits
- **Cross-validation**: 4-split rolling time series validation
- **Robustness**: Consistent performance across all platforms and seasons

## Key Technical Innovations

1. **Chinese E-commerce Calendar Integration**: Built-in knowledge of promotional periods (Singles Day, Chinese New Year, etc.)
2. **Store Type Classification**: Automatic categorization based on Chinese naming conventions
3. **Multi-Platform Dynamics**: Features capturing competitive effects across platforms
4. **Robust Temporal Validation**: Rolling splits that test all seasonal patterns

## Development

### Adding New Features
1. Extend `FeatureEngineer` class in `src/data/feature_engineering.py`
2. Add tests in `tests/test_feature_engineering.py`
3. Update configuration in `src/config/feature_config.yaml`

### Adding New Models
1. Create model class in `src/models/`
2. Follow the base model interface
3. Add model configuration
4. Include in model registry

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

[License information]

## Citation

If you use this project in your research, please cite:
```
[Citation format]
``` 