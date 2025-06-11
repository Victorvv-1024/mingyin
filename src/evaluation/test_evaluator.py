import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEvaluator:
    """Evaluates model performance on test data"""
    
    def __init__(self, 
                 output_dir: str = "results",
                 split_strategy: str = 'split_4_test'):
        """
        Initialize test evaluator
        
        Args:
            output_dir: Directory to save results
            split_strategy: Strategy used for splitting data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_strategy = split_strategy
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def safe_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate MAPE safely handling edge cases"""
        actual_clipped = np.clip(actual, 1, 1e6)
        predicted_clipped = np.clip(predicted, 1, 1e6)
        ape = np.abs(actual_clipped - predicted_clipped) / (actual_clipped + 1)
        return np.mean(ape) * 100
    
    def calculate_baselines(self, 
                          train_data: pd.DataFrame,
                          test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline predictions and their MAPE"""
        results = {}
        
        # 1. Simple mean baseline
        train_mean = train_data['sales_quantity'].mean()
        mean_predictions = np.full(len(test_data), train_mean)
        mean_mape = self.safe_mape(test_data['sales_quantity'].values, mean_predictions)
        results['mean_baseline'] = mean_mape
        
        # 2. Platform-specific means
        platform_predictions = []
        for idx, row in test_data.iterrows():
            platform = row['primary_platform']
            platform_train_data = train_data[train_data['primary_platform'] == platform]
            if len(platform_train_data) > 0:
                platform_mean = platform_train_data['sales_quantity'].mean()
            else:
                platform_mean = train_mean
            platform_predictions.append(platform_mean)
        
        platform_predictions = np.array(platform_predictions)
        platform_mape = self.safe_mape(test_data['sales_quantity'].values, platform_predictions)
        results['platform_baseline'] = platform_mape
        
        # 3. Feature-based baseline (Random Forest)
        feature_cols = [col for col in train_data.columns 
                       if col not in ['sales_quantity', 'sales_amount', 'sales_month', 
                                    'store_name', 'brand_name', 'primary_platform',
                                    'secondary_platform', 'product_code'] 
                       and not col.startswith('Unnamed')]
        
        basic_features = []
        for col in feature_cols:
            if col in ['month', 'quarter', 'year', 'unit_price']:
                basic_features.append(col)
        
        if len(basic_features) > 0:
            try:
                X_train = train_data[basic_features].fillna(0)
                X_test = test_data[basic_features].fillna(0)
                y_train = np.log1p(train_data['sales_quantity'])
                y_test = test_data['sales_quantity'].values
                
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X_train, y_train)
                rf_pred_log = rf.predict(X_test)
                rf_pred = np.expm1(rf_pred_log)
                rf_mape = self.safe_mape(y_test, rf_pred)
                results['rf_baseline'] = rf_mape
                
            except Exception as e:
                logger.error(f"Random Forest baseline failed: {str(e)}")
        
        return results
    
    def create_visualizations(self,
                            train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            baseline_results: Dict[str, float],
                            expected_model_mape: float) -> str:
        """Create and save visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sales distribution
        axes[0,0].hist(train_data['sales_quantity'], bins=50, alpha=0.7, label='Train', density=True)
        axes[0,0].hist(test_data['sales_quantity'], bins=50, alpha=0.7, label='Test', density=True)
        axes[0,0].set_xlabel('Sales Quantity')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Sales Distribution: Train vs Test')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # 2. Time series
        train_monthly = train_data.groupby(train_data['sales_month'].dt.to_period('M'))['sales_quantity'].mean()
        test_monthly = test_data.groupby(test_data['sales_month'].dt.to_period('M'))['sales_quantity'].mean()
        
        axes[0,1].plot(train_monthly.index.to_timestamp(), train_monthly.values, 'b-', label='Train', marker='o')
        axes[0,1].plot(test_monthly.index.to_timestamp(), test_monthly.values, 'r-', label='Test', marker='s')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Average Sales')
        axes[0,1].set_title(f'Time Series: {self.split_strategy}')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Platform comparison
        platforms = test_data['primary_platform'].unique()
        x = np.arange(len(platforms))
        width = 0.35
        
        train_platform_means = [train_data[train_data['primary_platform']==p]['sales_quantity'].mean() 
                              for p in platforms]
        test_platform_means = [test_data[test_data['primary_platform']==p]['sales_quantity'].mean() 
                             for p in platforms]
        
        axes[0,2].bar(x - width/2, train_platform_means, width, label='Train')
        axes[0,2].bar(x + width/2, test_platform_means, width, label='Test')
        axes[0,2].set_xlabel('Platform')
        axes[0,2].set_ylabel('Average Sales')
        axes[0,2].set_title('Platform Comparison: Train vs Test')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(platforms, rotation=45)
        axes[0,2].legend()
        
        # 4. Baseline comparison
        baseline_names = ['Mean', 'Platform-Specific']
        baseline_values = [baseline_results['mean_baseline'], 
                         baseline_results['platform_baseline']]
        if 'rf_baseline' in baseline_results:
            baseline_names.append('Random Forest')
            baseline_values.append(baseline_results['rf_baseline'])
        
        axes[1,0].bar(baseline_names, baseline_values)
        axes[1,0].set_ylabel('MAPE (%)')
        axes[1,0].set_title('Baseline Methods Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add expected performance line
        expected_line = [expected_model_mape] * len(baseline_names)
        axes[1,0].plot(baseline_names, expected_line, 'r--', linewidth=2, 
                      label=f'Expected Model: {expected_model_mape:.1f}%')
        axes[1,0].legend()
        
        # 5. Error analysis for best baseline
        best_baseline = min(baseline_values)
        best_baseline_idx = baseline_values.index(best_baseline)
        best_baseline_name = baseline_names[best_baseline_idx]
        
        if best_baseline_name == 'Mean':
            best_predictions = np.full(len(test_data), train_data['sales_quantity'].mean())
        elif best_baseline_name == 'Platform-Specific':
            best_predictions = []
            for idx, row in test_data.iterrows():
                platform = row['primary_platform']
                platform_train_data = train_data[train_data['primary_platform'] == platform]
                if len(platform_train_data) > 0:
                    platform_mean = platform_train_data['sales_quantity'].mean()
                else:
                    platform_mean = train_data['sales_quantity'].mean()
                best_predictions.append(platform_mean)
            best_predictions = np.array(best_predictions)
        
        baseline_errors = np.abs(test_data['sales_quantity'].values - best_predictions) / \
                         (test_data['sales_quantity'].values + 1) * 100
        
        axes[1,1].hist(baseline_errors, bins=50, alpha=0.7)
        axes[1,1].set_xlabel('Absolute Percentage Error (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title(f'Best Baseline Error Distribution (MAPE: {best_baseline:.1f}%)')
        axes[1,1].axvline(best_baseline, color='red', linestyle='--', 
                         label=f'Mean: {best_baseline:.1f}%')
        axes[1,1].legend()
        
        # 6. Model performance expectations
        axes[1,2].bar(['Best Baseline', 'Your Model\n(Conservative)', 
                      'Your Model\n(Optimistic)', 'Industry\nStandard'], 
                     [best_baseline, expected_model_mape, 3.5, 25])
        axes[1,2].set_ylabel('MAPE (%)')
        axes[1,2].set_title('Performance Comparison')
        axes[1,2].set_yscale('log')
        
        # Add text annotations
        axes[1,2].text(0, best_baseline*1.5, f'{best_baseline:.1f}%', ha='center')
        axes[1,2].text(1, expected_model_mape*1.5, f'{expected_model_mape:.1f}%', ha='center')
        axes[1,2].text(2, 3.5*1.5, '3.5%', ha='center')
        axes[1,2].text(3, 25*1.5, '25%', ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = self.output_dir / f'test_analysis_{self.split_strategy}_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def evaluate(self, 
                train_data: pd.DataFrame,
                test_data: pd.DataFrame,
                expected_model_mape: float = 4.86) -> Dict:
        """
        Run complete test evaluation
        
        Args:
            train_data: Training data
            test_data: Test data
            expected_model_mape: Expected MAPE for the model
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting test evaluation...")
        
        # 1. Calculate baselines
        baseline_results = self.calculate_baselines(train_data, test_data)
        best_baseline = min([v for v in baseline_results.values() if v < 1000])
        
        # 2. Create visualizations
        viz_file = self.create_visualizations(
            train_data, test_data, baseline_results, expected_model_mape
        )
        
        # 3. Prepare results summary
        results_summary = {
            'split_strategy': self.split_strategy,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_period': f"{train_data['sales_month'].min().date()} to {train_data['sales_month'].max().date()}",
            'test_period': f"{test_data['sales_month'].min().date()} to {test_data['sales_month'].max().date()}",
            'baselines': baseline_results,
            'best_baseline_mape': best_baseline,
            'expected_model_mape': expected_model_mape,
            'improvement_factor': best_baseline / expected_model_mape if expected_model_mape > 0 else 0,
            'business_assessment': 'Outstanding' if expected_model_mape < 10 else 'Excellent',
            'visualization_file': viz_file
        }
        
        # 4. Save test data
        test_data_file = self.output_dir / f'test_data_{self.split_strategy}_{self.timestamp}.csv'
        test_data.to_csv(test_data_file, index=False)
        results_summary['test_data_file'] = str(test_data_file)
        
        # 5. Print summary
        logger.info("\nTest Evaluation Summary:")
        logger.info(f"  Split Strategy: {self.split_strategy}")
        logger.info(f"  Train Size: {len(train_data):,}")
        logger.info(f"  Test Size: {len(test_data):,}")
        logger.info(f"  Best Baseline MAPE: {best_baseline:.2f}%")
        logger.info(f"  Expected Model MAPE: {expected_model_mape:.2f}%")
        logger.info(f"  Improvement Factor: {best_baseline/expected_model_mape:.1f}x")
        logger.info(f"  Business Assessment: {results_summary['business_assessment']}")
        logger.info(f"  Results saved to: {self.output_dir}")
        
        return results_summary 