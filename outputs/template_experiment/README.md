# Experiment Template Structure

This template shows the standard directory structure for each experiment run.

## Folder Organization

```
{experiment_name}/                  # e.g., 20250616_120000
├── vanilla_models/                 # Vanilla embedding model outputs
├── enhanced_models/                # Enhanced embedding model outputs
├── predictions/                    # All model predictions and metrics
├── reports/                        # Experiment reports and analysis
└── feature_engineering/            # Feature engineering outputs
```

## Subfolder Contents

### `vanilla_models/`
- Model checkpoints (.keras files)
- Architecture configuration
- Training history and metrics
- Best model selection

### `enhanced_models/`
- Model checkpoints (.keras files)
- Architecture configuration  
- Training history and metrics
- Best model selection

### `predictions/`
- Fold-wise predictions
- Performance metrics (MAPE, MAE, RMSE)
- Model comparison results
- Phase 3 evaluation (if run)

### `reports/`
- Experiment summary report
- Configuration metadata
- Validation results analysis
- Business insights (Phase 3)

### `feature_engineering/`
- Feature importance analysis
- Engineering statistics
- Correlation analysis
- Processing logs

## Usage

This template structure is automatically created when running:
```bash
python run_complete_pipeline.py --model-type [vanilla|enhanced|both]
```

Each run creates a new timestamped experiment folder following this structure. 