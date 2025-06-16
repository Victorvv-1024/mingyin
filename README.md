# Sales Forecasting Pipeline - Complete Guide

This repository contains a **production-ready, modular sales forecasting pipeline** with advanced deep learning models for accurate sales prediction. The system supports both baseline and enhanced models with comprehensive evaluation capabilities.

## 🏗️ Architecture Overview

### **Three-Phase Pipeline**

```
📊 PHASE 1: Feature Engineering
├── Raw data processing (Excel files)
├── 80+ engineered features
├── Chinese market-specific features
├── Rolling time splits preparation
└── Engineered dataset output

🤖 PHASE 2: Model Training  
├── Vanilla Model (Baseline)
├── Enhanced Model (Advanced)
├── Rolling splits validation
├── Model comparison
└── Best models saved

🎯 PHASE 3: Model Evaluation
├── 2023 data testing
├── Performance analysis
├── Visualization generation
└── Business recommendations
```

### **Model Architecture**

We provide **two complementary models**:

| Model | Description | Use Case |
|-------|-------------|----------|
| **VanillaEmbeddingModel** | Original baseline implementation | Comparison benchmark, stable baseline |
| **EnhancedEmbeddingModel** | Advanced architecture with improvements | Production deployment, best performance |

## 🚀 Quick Start

### **Option 1: Complete Pipeline (Recommended)**

Run all three phases with enhanced model:

```bash
python src/scripts/run_complete_pipeline.py \
    --data-dir data/raw \
    --output-dir outputs \
    --years 2021 2022 \
    --epochs 100 \
    --model-type enhanced \
    --run-phase3
```

### **Option 2: Model Comparison**

Train both models and compare performance:

```bash
python src/scripts/run_complete_pipeline.py \
    --data-dir data/raw \
    --output-dir outputs \
    --years 2021 2022 \
    --epochs 100 \
    --model-type both \
    --run-phase3
```

### **Option 3: Individual Phases**

```bash
# Phase 1: Feature Engineering
python src/scripts/phase1_feature_engineering.py --data-dir data/raw --output-dir outputs/engineered

# Phase 2: Model Training  
python src/scripts/phase2_model_training.py --engineered-dataset outputs/engineered/dataset.pkl --output-dir outputs/models

# Phase 3: Model Evaluation
python src/scripts/phase3_test_model.py --models-dir outputs/models --engineered-dataset outputs/engineered/dataset.pkl
```

## 📁 Project Structure

```
📁 src/
├── 📁 data/                          # Data processing pipeline
│   ├── 📄 feature_pipeline.py        # Master feature orchestration
│   ├── 📄 preprocessing.py           # Data cleaning and preparation
│   ├── 📄 utils.py                   # Data utility functions
│   └── 📁 features/                  # Specialized feature engineering
│       ├── 📄 temporal.py            # Time-based + Chinese calendar
│       ├── 📄 customer_behavior.py   # Store analytics
│       ├── 📄 store_categorization.py # Chinese store classification
│       ├── 📄 platform_dynamics.py   # Cross-platform competition
│       └── 📄 promotional_calendar.py # E-commerce events
├── 📁 models/                        # Deep learning models
│   ├── 📄 vanilla_embedding_model.py # Baseline model
│   ├── 📄 enhanced_embedding_model.py # Advanced model
│   ├── 📄 feature_processor.py       # Multi-input data preparation
│   ├── 📄 trainer.py                 # Training orchestration
│   └── 📄 custom_objects.py          # Shared TensorFlow components
├── 📁 scripts/                       # Execution scripts
│   ├── 📄 run_complete_pipeline.py   # Main pipeline (all phases)
│   ├── 📄 phase1_feature_engineering.py # Feature engineering only
│   ├── 📄 phase2_model_training.py   # Model training only
│   └── 📄 phase3_test_model.py       # Model evaluation only
├── 📁 utils/                         # Utilities
│   ├── 📄 logging_utils.py           # Standardized logging
│   ├── 📄 args_utils.py              # Argument parsing utilities
│   └── 📄 helpers.py                 # General utilities
└── 📁 config/                        # Configuration
    └── 📄 feature_config.yaml        # Feature engineering settings

📁 docs/                              # Documentation
└── 📄 USAGE_GUIDE.md                 # Detailed usage guide

📁 outputs/                           # Generated outputs
├── 📁 engineered/                    # Phase 1 outputs
├── 📁 {experiment_name}/              # Phase 2 & 3 outputs
│   ├── 📁 vanilla_models/            # Vanilla model files
│   ├── 📁 enhanced_models/           # Enhanced model files
│   ├── 📁 predictions/               # Model predictions
│   ├── 📁 reports/                   # Performance reports
│   │   ├── 📄 training_summary.txt     # Training results
│   │   ├── 📄 model_comparison.json    # Model comparison metrics
│   │   └── 📄 experiment_metadata.json # Experiment configuration
│   └── 📁 2023_evaluation/           # Phase 3 evaluation results
```

## 🎯 Feature Engineering

Our pipeline creates **80+ sophisticated features** specifically designed for Chinese e-commerce sales forecasting:

### **Temporal Features**
- **Chinese Calendar Integration**: Spring Festival, Golden Week, Singles' Day
- **Seasonal Patterns**: Monthly, quarterly, yearly cycles
- **Lag Features**: 1, 2, 3, 6, 12-month lags
- **Rolling Statistics**: 3, 6, 12-month moving averages

### **Business Intelligence Features**
- **Store Categorization**: Tier-1/2/3 city classification
- **Platform Dynamics**: Cross-platform competition analysis
- **Customer Behavior**: Purchase patterns and loyalty metrics
- **Promotional Calendar**: Major e-commerce events impact

### **Advanced Analytics**
- **Year-over-Year Growth**: Trend analysis
- **Interaction Features**: Store × Brand × Platform combinations
- **Market Share**: Competitive positioning metrics
- **Seasonality Decomposition**: Trend, seasonal, residual components

## 🤖 Model Architecture

### **VanillaEmbeddingModel (Baseline)**
- **Purpose**: Stable baseline for comparison
- **Architecture**: Standard embedding + attention mechanism
- **Features**: Original feature processing pipeline
- **Use Case**: Benchmark, fallback option

### **EnhancedEmbeddingModel (Advanced)**
- **Purpose**: Production-ready enhanced performance
- **Architecture**: Advanced embedding + multi-head attention + residual connections
- **Features**: Enhanced feature processing + regularization
- **Improvements**: 
  - Better numerical stability
  - Advanced optimizer (AdamW)
  - Enhanced callbacks
  - Improved model saving (Keras 3 compatible)

### **Key Technical Features**
- **Multi-Input Architecture**: Handles categorical + continuous features
- **Embedding Layers**: Efficient categorical feature processing
- **Attention Mechanism**: Focus on important features
- **Regularization**: Dropout, batch normalization, early stopping
- **Custom Metrics**: MAPE and RMSE in original scale

## 📊 Model Performance Analysis

### **Expected Performance Benchmarks**

| Metric | Vanilla Model | Enhanced Model | Target |
|--------|---------------|----------------|---------|
| Validation MAPE | ~10-22% | ~14-18% | <20% |
| 2023 Test MAPE | ~30-45% | ~16-20% | <20% |
| Training Time | ~70 min | On Mac ~24 hr | - |

### **🔍 Vanilla Model Performance on 2023 Unseen Test Data**

The vanilla embedding model was evaluated on 2023 data to assess its generalization capability on completely unseen data. Here's the comprehensive performance analysis:

![Vanilla Model 2023 Evaluation](outputs/vanilla-model/reports/vanilla_model_2023_evaluation_plots_20250613_172409.png)

#### **📈 Performance Summary**
- **✅ Models Evaluated**: 5 cross-validation folds
- **📊 Best MAPE**: 32.73% (Split 5)
- **📊 Worst MAPE**: 43.35% (Split 1) 
- **📊 Average MAPE**: 39.95%
- **🏆 Overall Grade**: POOR
- **💡 Assessment**: ❌ Poor performance - significant improvements needed

#### **🔬 Detailed Analysis**

**1. MAPE Performance by Split:**
- **Split 1**: 43.4% - Highest error, indicating potential overfitting to training period
- **Split 2**: 42.8% - Consistently high error across validation folds
- **Split 3**: 38.3% - Best performing split, showing some generalization capability
- **Split 4**: 42.6% - Similar to splits 1-2, suggesting systematic issues
- **Split 5**: 32.7% - **Best performance**, indicating model potential under optimal conditions

**2. RMSE Consistency:**
- All splits show similar RMSE values (~4500), indicating consistent error magnitude
- Uniform RMSE suggests the model maintains prediction scale consistency across folds

**3. R² Score Analysis:**
- **Critical Issue**: All R² scores are negative (-0.001 to -0.0025)
- **Interpretation**: Model performs worse than simply predicting the mean
- **Root Cause**: Indicates fundamental model architecture or feature engineering issues

**4. Actual vs Predicted Scatter Plot (Best Model - Split 5):**
- **Severe Underprediction**: Model consistently predicts much lower values than actual sales
- **Limited Range**: Predictions clustered in 0-2000 range while actual sales reach 15000+
- **Poor Correlation**: Scatter shows no clear linear relationship between actual and predicted values

#### **🚨 Key Issues Identified**

1. **Scale Mismatch**: Model fails to capture the full range of sales values
2. **Feature Inadequacy**: Current features may not capture essential sales drivers
3. **Architecture Limitations**: Vanilla embedding approach insufficient for complex patterns
4. **Temporal Generalization**: Poor performance on 2023 suggests overfitting to 2021-2022 patterns

#### **💡 Improvement Recommendations**

1. **Enhanced Model Architecture**: 
   - Implement multi-head attention mechanisms
   - Add residual connections for better gradient flow
   - Increase model capacity with deeper networks

2. **Feature Engineering Enhancements**:
   - Add more sophisticated temporal features
   - Include external economic indicators
   - Implement advanced lag and rolling window features

3. **Training Strategy Improvements**:
   - Implement more robust cross-validation
   - Add regularization techniques
   - Use advanced optimizers (AdamW, RMSprop)

4. **Data Quality Assessment**:
   - Investigate data distribution shifts between training and test periods
   - Analyze outliers and anomalies in 2023 data
   - Consider domain adaptation techniques

This analysis clearly demonstrates why the **EnhancedEmbeddingModel** was developed - to address these fundamental limitations and achieve production-ready performance.

## 🔧 Configuration Options

### **Model Selection**
```bash
--model-type vanilla    # Train baseline model only
--model-type enhanced   # Train enhanced model only (default)
--model-type both       # Train both models for comparison
```

### **Pipeline Control**
```bash
--run-phase3                    # Include 2023 evaluation
--skip-feature-engineering      # Use existing engineered dataset
--engineered-dataset path.pkl   # Specify existing dataset
```

### **Training Parameters**
```bash
--epochs 150              # Training epochs (default: 100)
--batch-size 256          # Batch size (default: 512)
--random-seed 42          # Reproducibility seed
--experiment-name "prod"  # Custom experiment name
```

## 📈 Output Structure

After running the complete pipeline, you'll get:

```
📁 outputs/
└── 📁 {experiment_name}_YYYYMMDD_HHMMSS/
    ├── 📁 vanilla_models/              # Vanilla model files (.keras)
    ├── 📁 enhanced_models/             # Enhanced model files (.keras)
    ├── 📁 predictions/                 # CSV prediction files
    ├── 📁 reports/                     # Performance analysis
    │   ├── 📄 training_summary.txt     # Training results
    │   ├── 📄 model_comparison.json    # Model comparison metrics
    │   └── 📄 experiment_metadata.json # Experiment configuration
    └── 📁 2023_evaluation/             # Phase 3 results
        ├── 📄 evaluation_results.json  # 2023 test performance
        ├── 📄 performance_plots.png    # Visualization
        └── 📄 business_recommendations.txt # Actionable insights
```

## 🔍 Model Comparison

When training both models (`--model-type both`), you'll get:

```
MODEL COMPARISON:
  Vanilla (Baseline): 19.45%
  Enhanced: 15.23%
  Improvement: 4.22 percentage points
  ✅ Enhanced model outperforms vanilla baseline
```

## 🛠️ Installation & Setup

### **Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd sales-forecasting-pipeline

# Create conda environment
conda create -n sales_forecasting python=3.9
conda activate sales_forecasting

# Install dependencies
pip install -r requirements.txt
```

### **Data Preparation**
```bash
# Place Excel files in data/raw/
data/raw/
├── 2021.xlsx
├── 2022.xlsx
└── 2023.xlsx  # Optional, for Phase 3 evaluation
```

### **Verify Installation**
```bash
python src/scripts/run_complete_pipeline.py --help
```

## 🔧 Customization

### **Feature Engineering**
Modify `src/config/feature_config.yaml`:
```yaml
temporal_features:
  lag_periods: [1, 2, 3, 6, 12]
  rolling_windows: [3, 6, 12]
  
business_features:
  enable_store_categorization: true
  enable_platform_dynamics: true
```

### **Model Architecture**
Customize models in:
- `src/models/vanilla_embedding_model.py` - Baseline model
- `src/models/enhanced_embedding_model.py` - Advanced model

### **Training Parameters**
```python
# In your custom script
from models.enhanced_embedding_model import EnhancedEmbeddingModel

model = EnhancedEmbeddingModel(random_seed=42)
results = model.train_on_rolling_splits(
    df_final=df,
    features=features,
    rolling_splits=splits,
    epochs=150,           # Custom epochs
    batch_size=256,       # Custom batch size
    models_dir="custom/path"
)
```

## 🚨 Troubleshooting

### **Common Issues**

#### **Issue 1: CUDA/GPU Problems**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python src/scripts/run_complete_pipeline.py ...
```

#### **Issue 2: Memory Issues**
```bash
# Reduce batch size
python src/scripts/run_complete_pipeline.py --batch-size 128
```

#### **Issue 3: Different Results**
```bash
# Ensure consistent random seed
python src/scripts/run_complete_pipeline.py --random-seed 42
```

#### **Issue 4: Missing 2023 Data**
```bash
# Skip Phase 3 if no 2023 data
python src/scripts/run_complete_pipeline.py  # Don't use --run-phase3
```

## 📚 Documentation

- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Comprehensive usage guide
- **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)** - File organization
- **Model Documentation** - In-code documentation for each model

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Scikit-learn for machine learning utilities
- Chinese e-commerce domain expertise contributors

---

**Ready to forecast sales with state-of-the-art deep learning?** 🚀

```bash
python src/scripts/run_complete_pipeline.py --data-dir data/raw --model-type enhanced --run-phase3
```