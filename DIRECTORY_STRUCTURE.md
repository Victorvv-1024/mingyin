# Sales Forecasting Pipeline Directory Structure

This document outlines the consistent directory structure used across all phases of the sales forecasting pipeline.

## ✅ Consistent Structure (After Fix)

### Phase 1: Feature Engineering
```
outputs/
└── engineered/
    ├── sales_forecast_engineered_dataset_20250113_143025.pkl
    ├── sales_forecast_engineered_data_20250113_143025.csv
    ├── modeling_features_20250113_143025.txt
    └── dataset_metadata_20250113_143025.txt
```

### Phase 2: Model Training
```
outputs/
└── phase2_model_training_20250113_143025/     # Experiment directory
    ├── models/
    │   ├── best_model_split_1_epoch_045_mape_12.34_20250113_143025.h5
    │   ├── best_model_split_2_epoch_038_mape_15.67_20250113_143025.h5
    │   └── best_model_split_3_epoch_042_mape_14.23_20250113_143025.h5
    ├── predictions/
    │   ├── detailed_predictions_split_1_20250113_143025.csv
    │   ├── detailed_predictions_split_2_20250113_143025.csv
    │   └── detailed_predictions_split_3_20250113_143025.csv
    └── reports/
        ├── experiment_report_20250113_143025.txt
        ├── performance_summary_20250113_143025.csv
        └── experiment_metadata_20250113_143025.json
```

### Phase 3: Model Evaluation (✅ FIXED)
```
outputs/
└── phase2_model_training_20250113_143025/     # Same experiment directory
    ├── models/                                 # ← Input: Models from Phase 2
    │   └── [model files]
    ├── predictions/                            # ← From Phase 2
    ├── reports/                                # ← From Phase 2
    └── 2023_evaluation/                        # ← NEW: Phase 3 results
        ├── 2023_evaluation_results_20250113_150030.json
        ├── 2023_evaluation_summary_20250113_150030.csv
        ├── 2023_predictions_sample_20250113_150030.csv
        └── 2023_evaluation_plots_20250113_150030.png
```

## Commands and Directory Flow

### Phase 1: Feature Engineering
```bash
python src/scripts/phase1_feature_engineering.py \
    --data-dir data/raw \
    --output-dir outputs/engineered
```
**Output**: `outputs/engineered/`

### Phase 2: Model Training
```bash
python src/scripts/phase2_model_training.py \
    --engineered-dataset outputs/engineered/dataset.pkl \
    --output-dir outputs \
    --experiment-name phase2_model_training_20250113_143025
```
**Output**: `outputs/phase2_model_training_20250113_143025/`

### Phase 3: Model Evaluation (✅ IMPROVED)
```bash
# Before (inconsistent):
python src/scripts/phase3_test_model.py \
    --models-dir outputs/phase2_model_training_20250113_143025/models \
    --output-dir outputs/2023_evaluation_fixed \  # ❌ Different location
    --engineered-dataset outputs/engineered/dataset.pkl

# After (consistent):
python src/scripts/phase3_test_model.py \
    --models-dir outputs/phase2_model_training_20250113_143025/models \
    --engineered-dataset outputs/engineered/dataset.pkl
    # ✅ Output automatically goes to: outputs/phase2_model_training_20250113_143025/2023_evaluation/
```

## Benefits of the Fixed Structure

### 1. **Consistency**
- All outputs from the same experiment are in one directory
- Easy to find related files (models, predictions, evaluations)

### 2. **Traceability** 
- Clear relationship between training and evaluation results
- Timestamps link all phases of the same experiment

### 3. **Organization**
- No scattered output directories
- Self-documenting structure

### 4. **Simplified Commands**
- Fewer arguments needed for Phase 3
- Automatic output directory detection

## Migration from Old Structure

If you have existing results in the old structure:

```bash
# Old structure:
outputs/2023_evaluation_fixed/
└── [evaluation results]

# Move to new structure:
mkdir -p outputs/your_experiment_name/2023_evaluation/
mv outputs/2023_evaluation_fixed/* outputs/your_experiment_name/2023_evaluation/
```

## Directory Structure Patterns

| Phase | Input | Output Location | Pattern |
|-------|-------|-----------------|---------|
| Phase 1 | `data/raw/` | `outputs/engineered/` | Fixed location |
| Phase 2 | Engineered data | `outputs/{experiment_name}/` | Experiment-based |
| Phase 3 | Models directory | `{models_dir}/../2023_evaluation/` | Same experiment dir |

This structure ensures that all outputs from a single training experiment stay together, making it easier to manage, compare, and archive results. 