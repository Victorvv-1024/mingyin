import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    """Handles model evaluation and analysis"""
    
    def __init__(self, model):
        self.model = model
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_detailed_predictions(self, val_data, val_pred_orig, val_true_orig, split_num, description, model_type):
        """Save detailed predictions with comprehensive analysis"""
        # Create results DataFrame
        results_df = val_data.copy()
        results_df['predicted_sales'] = val_pred_orig
        results_df['actual_sales'] = val_true_orig
        results_df['absolute_error'] = np.abs(val_pred_orig - val_true_orig)
        results_df['absolute_percentage_error'] = np.abs(val_pred_orig - val_true_orig) / (val_true_orig + 1) * 100
        results_df['is_perfect_prediction'] = results_df['absolute_error'] < 1
        
        # Add error categories
        def categorize_error(ape):
            if ape < 5: return "Excellent (<5%)"
            elif ape < 10: return "Very Good (5-10%)"
            elif ape < 20: return "Good (10-20%)"
            elif ape < 50: return "Fair (20-50%)"
            else: return "Poor (>50%)"
        
        results_df['error_category'] = results_df['absolute_percentage_error'].apply(categorize_error)
        
        # Save detailed predictions
        predictions_filename = f"detailed_predictions_split_{split_num}_{model_type}_{self.timestamp}.csv"
        results_df.to_csv(predictions_filename, index=False)
        
        # Save summary predictions (top-level metrics only)
        summary_cols = ['sales_month', 'primary_platform', 'store_name', 'brand_name', 
                       'actual_sales', 'predicted_sales', 'absolute_percentage_error', 'error_category']
        available_cols = [col for col in summary_cols if col in results_df.columns]
        
        summary_filename = f"summary_predictions_split_{split_num}_{model_type}_{self.timestamp}.csv"
        results_df[available_cols].to_csv(summary_filename, index=False)
        
        print(f"  📊 Predictions saved:")
        print(f"    Detailed: {predictions_filename}")
        print(f"    Summary: {summary_filename}")
        
        return results_df, {'detailed_predictions': predictions_filename, 'summary_predictions': summary_filename}
    
    def save_split_analysis_report(self, results_df, split_num, description, model_type):
        """Save comprehensive analysis report for a training split"""
        filename = f"training_analysis_report_split_{split_num}_{model_type}_{self.timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"TRAINING SPLIT ANALYSIS REPORT\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Split: {split_num} - {description}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Total Predictions: {len(results_df):,}\n\n")
            
            # Overall metrics
            f.write(f"OVERALL PERFORMANCE METRICS\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Mean Absolute Percentage Error: {results_df['absolute_percentage_error'].mean():.2f}%\n")
            f.write(f"Median Absolute Percentage Error: {results_df['absolute_percentage_error'].median():.2f}%\n")
            f.write(f"Standard Deviation of APE: {results_df['absolute_percentage_error'].std():.2f}%\n")
            f.write(f"Mean Absolute Error: {results_df['absolute_error'].mean():.0f} units\n")
            f.write(f"Root Mean Square Error: {np.sqrt(np.mean(results_df['absolute_error']**2)):.0f} units\n\n")
            
            # Error distribution
            f.write(f"ERROR DISTRIBUTION BY CATEGORY\n")
            f.write(f"-" * 30 + "\n")
            error_dist = results_df['error_category'].value_counts()
            for category in error_dist.index:
                count = error_dist[category]
                percentage = count / len(results_df) * 100
                f.write(f"{category}: {count:,} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Percentile analysis
            f.write(f"ERROR PERCENTILES\n")
            f.write(f"-" * 18 + "\n")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            ape_percentiles = np.percentile(results_df['absolute_percentage_error'], percentiles)
            for p, value in zip(percentiles, ape_percentiles):
                f.write(f"{p:2d}th percentile: {value:.2f}%\n")
            f.write("\n")
            
            # Platform analysis (if available)
            if 'primary_platform' in results_df.columns:
                f.write(f"PERFORMANCE BY PLATFORM\n")
                f.write(f"-" * 25 + "\n")
                platform_stats = results_df.groupby('primary_platform').agg({
                    'absolute_percentage_error': ['mean', 'median', 'std', 'count'],
                    'is_perfect_prediction': 'sum'
                }).round(2)
                
                for platform in platform_stats.index:
                    f.write(f"{platform}:\n")
                    f.write(f"  Mean APE: {platform_stats.loc[platform, ('absolute_percentage_error', 'mean')]:.2f}%\n")
                    f.write(f"  Median APE: {platform_stats.loc[platform, ('absolute_percentage_error', 'median')]:.2f}%\n")
                    f.write(f"  Samples: {platform_stats.loc[platform, ('absolute_percentage_error', 'count')]:,}\n")
                    f.write(f"  Perfect predictions: {platform_stats.loc[platform, ('is_perfect_prediction', 'sum')]:,}\n\n")
            
            # Time-based analysis (if available)
            if 'sales_month' in results_df.columns:
                f.write(f"PERFORMANCE BY MONTH\n")
                f.write(f"-" * 20 + "\n")
                monthly_stats = results_df.groupby(results_df['sales_month'].dt.month).agg({
                    'absolute_percentage_error': ['mean', 'count'],
                    'is_perfect_prediction': 'sum'
                }).round(2)
                
                for month in sorted(monthly_stats.index):
                    f.write(f"Month {month:2d}: ")
                    f.write(f"{monthly_stats.loc[month, ('absolute_percentage_error', 'mean')]:.2f}% MAPE ")
                    f.write(f"({monthly_stats.loc[month, ('absolute_percentage_error', 'count')]:,} samples)\n")
            
            # Brand analysis (if available)
            if 'brand_name' in results_df.columns:
                f.write(f"\nTOP 10 BRANDS BY SAMPLE COUNT\n")
                f.write(f"-" * 30 + "\n")
                brand_stats = results_df.groupby('brand_name').agg({
                    'absolute_percentage_error': ['mean', 'count'],
                    'is_perfect_prediction': 'sum'
                }).round(2)
                
                top_brands = brand_stats.nlargest(10, ('absolute_percentage_error', 'count'))
                for brand in top_brands.index:
                    mean_ape = top_brands.loc[brand, ('absolute_percentage_error', 'mean')]
                    count = top_brands.loc[brand, ('absolute_percentage_error', 'count')]
                    perfect = top_brands.loc[brand, ('is_perfect_prediction', 'sum')]
                    f.write(f"{brand}: {mean_ape:.2f}% MAPE ({count:,} samples, {perfect:,} perfect)\n")
            
            # Store analysis (if available)  
            if 'store_name' in results_df.columns:
                f.write(f"\nTOP 10 STORES BY SAMPLE COUNT\n")
                f.write(f"-" * 30 + "\n")
                store_stats = results_df.groupby('store_name').agg({
                    'absolute_percentage_error': ['mean', 'count'],
                    'is_perfect_prediction': 'sum'
                }).round(2)
                
                top_stores = store_stats.nlargest(10, ('absolute_percentage_error', 'count'))
                for store in top_stores.index:
                    mean_ape = top_stores.loc[store, ('absolute_percentage_error', 'mean')]
                    count = top_stores.loc[store, ('absolute_percentage_error', 'count')]
                    perfect = top_stores.loc[store, ('is_perfect_prediction', 'sum')]
                    f.write(f"{store}: {mean_ape:.2f}% MAPE ({count:,} samples, {perfect:,} perfect)\n")
            
            # Suspicious patterns
            f.write(f"\nSUSPICION ANALYSIS\n")
            f.write(f"-" * 18 + "\n")
            perfect_count = results_df['is_perfect_prediction'].sum()
            perfect_pct = perfect_count / len(results_df) * 100
            f.write(f"Perfect predictions (<1 unit error): {perfect_count:,} ({perfect_pct:.1f}%)\n")
            
            if perfect_pct > 5:
                f.write(f"⚠️ WARNING: High percentage of perfect predictions may indicate data leakage\n")
            
            mean_ape = results_df['absolute_percentage_error'].mean()
            if mean_ape < 5:
                f.write(f"⚠️ WARNING: Very low MAPE ({mean_ape:.2f}%) may indicate technical issues\n")
            
            # Prediction range analysis
            pred_min = results_df['predicted_sales'].min()
            pred_max = results_df['predicted_sales'].max()
            actual_min = results_df['actual_sales'].min()
            actual_max = results_df['actual_sales'].max()
            
            f.write(f"\nPREDICTION RANGE ANALYSIS\n")
            f.write(f"-" * 25 + "\n")
            f.write(f"Predicted sales range: [{pred_min:.0f}, {pred_max:.0f}]\n")
            f.write(f"Actual sales range: [{actual_min:.0f}, {actual_max:.0f}]\n")
            f.write(f"Range coverage ratio: {(pred_max - pred_min) / (actual_max - actual_min):.2f}\n")
            
            # Worst predictions
            f.write(f"\nWORST PREDICTIONS (Top 10)\n")
            f.write(f"-" * 27 + "\n")
            worst_predictions = results_df.nlargest(10, 'absolute_percentage_error')
            for idx, row in worst_predictions.iterrows():
                f.write(f"Actual: {row['actual_sales']:8.0f}, Predicted: {row['predicted_sales']:8.0f}, ")
                f.write(f"APE: {row['absolute_percentage_error']:6.1f}%")
                if 'store_name' in row and 'brand_name' in row:
                    f.write(f" ({row['store_name']}, {row['brand_name']})")
                f.write("\n")
            
            # Best predictions
            f.write(f"\nBEST PREDICTIONS (Top 10)\n")
            f.write(f"-" * 26 + "\n")
            best_predictions = results_df.nsmallest(10, 'absolute_percentage_error')
            for idx, row in best_predictions.iterrows():
                f.write(f"Actual: {row['actual_sales']:8.0f}, Predicted: {row['predicted_sales']:8.0f}, ")
                f.write(f"APE: {row['absolute_percentage_error']:6.1f}%")
                if 'store_name' in row and 'brand_name' in row:
                    f.write(f" ({row['store_name']}, {row['brand_name']})")
                f.write("\n")
        
        print(f"  📊 Analysis report saved: {filename}")
        return filename
    
    def enhanced_sanity_check_results(self, results):
        """Enhanced sanity checks on results"""
        print("\n" + "=" * 50)
        print("ENHANCED SANITY CHECKS ON RESULTS")
        print("=" * 50)
        
        if not results:
            print("❌ No results to check")
            return False
        
        mapes = [result['mape'] for result in results.values()]
        avg_mape = np.mean(mapes)
        
        # Check 1: Too good to be true?
        if avg_mape < 5:
            print(f"🚨 SUSPICIOUS: Average MAPE ({avg_mape:.2f}%) is suspiciously low")
            print("   This may indicate data leakage or incorrect calculation")
            return False
        
        # Check 2: All splits performing similarly well?
        mape_std = np.std(mapes)
        if mape_std < 2 and avg_mape < 15:
            print(f"🚨 SUSPICIOUS: All splits perform very similarly ({mape_std:.2f}% std)")
            print("   This may indicate overfitting or data leakage")
            return False
        
        # Check 3: Check for perfect predictions across splits
        total_perfect = sum([result.get('perfect_predictions', 0) for result in results.values()])
        total_predictions = sum([result.get('total_predictions', 1) for result in results.values()])
        perfect_ratio = total_perfect / total_predictions
        
        if perfect_ratio > 0.1:  # More than 10% perfect predictions
            print(f"🚨 SUSPICIOUS: {perfect_ratio*100:.1f}% perfect predictions across all splits")
            print("   This strongly suggests data leakage")
            return False
        
        # Check 4: Dramatic improvement from baseline?
        if avg_mape < 10:
            print(f"📊 BUSINESS REALITY CHECK:")
            print(f"   Average MAPE: {avg_mape:.2f}%")
            print(f"   This means predictions are typically within {avg_mape:.1f}% of actual sales")
            print(f"   For a product selling 1000 units, predictions would be ±{avg_mape*10:.0f} units")
            print(f"   Please validate this level of accuracy with business stakeholders")
        
        print(f"✅ Results pass enhanced sanity checks")
        return True
    
    def print_final_results(self, results):
        """Print comprehensive results with saved files information"""
        print("\n" + "=" * 70)
        print("ADVANCED EMBEDDING MODEL RESULTS")
        print("=" * 70)
        
        if not results:
            print("❌ No successful training completed")
            return
        
        mapes = [result['mape'] for result in results.values()]
        avg_mape = np.mean(mapes)
        
        print(f"Results by split:")
        for split_name, result in results.items():
            train_mape = result.get('train_mape', 'N/A')
            perfect_preds = result.get('perfect_predictions', 0)
            total_preds = result.get('total_predictions', 0)
            perfect_pct = (perfect_preds / total_preds * 100) if total_preds > 0 else 0
            
            print(f"  {result['description']}:")
            train_display = f"{train_mape:.2f}%" if train_mape != 'N/A' and isinstance(train_mape, (int, float)) else 'N/A'
            print(f"    MAPE: {result['mape']:.2f}% (train: {train_display})")
            print(f"    Perfect predictions: {perfect_preds:,}/{total_preds:,} ({perfect_pct:.1f}%)")
            
            # Show saved files
            if 'saved_files' in result:
                files = result['saved_files']
                print(f"    📁 Saved files:")
                print(f"      Model: {files.get('model', 'N/A')}")
                print(f"      Predictions: {files.get('summary_predictions', 'N/A')}")
                print(f"      Analysis: {files.get('analysis_report', 'N/A')}")
        
        print(f"\nOverall Performance:")
        print(f"  Average MAPE: {avg_mape:.2f}%")
        print(f"  Best MAPE: {min(mapes):.2f}%")
        print(f"  Worst MAPE: {max(mapes):.2f}%")
        print(f"  Standard Deviation: {np.std(mapes):.2f}%")
        
        # Enhanced sanity checks
        total_perfect = sum([result.get('perfect_predictions', 0) for result in results.values()])
        total_predictions = sum([result.get('total_predictions', 1) for result in results.values()])
        overall_perfect_pct = (total_perfect / total_predictions * 100) if total_predictions > 0 else 0
        
        print(f"\nOVERALL QUALITY ANALYSIS:")
        print(f"  Total predictions across all splits: {total_predictions:,}")
        print(f"  Total perfect predictions: {total_perfect:,} ({overall_perfect_pct:.1f}%)")
        
        # Perform enhanced sanity checks
        is_sane = self.enhanced_sanity_check_results(results)
        
        if is_sane:
            if avg_mape <= 20:
                print(f"\n🎉 BREAKTHROUGH! Average MAPE ({avg_mape:.2f}%) broke 20% barrier!")
                if avg_mape <= 10:
                    print(f"🌟 EXCELLENT! Achieved business-usable accuracy!")
                    print(f"📋 RECOMMENDATION: Validate these results with business stakeholders")
                    print(f"📋 Review saved prediction files for detailed analysis")
            else:
                print(f"\n⚠️ Still above 20% threshold ({avg_mape:.2f}%)")
        else:
            print(f"\n🚨 RESULTS FAILED SANITY CHECKS - INVESTIGATE POTENTIAL ISSUES")
            print(f"   📋 RECOMMENDATION: Review saved prediction files to identify issues")
            print(f"   📋 Check for data leakage in detailed prediction analysis")
        
        # Summary of all saved files
        print(f"\n📁 ALL SAVED FILES SUMMARY:")
        all_files = []
        for result in results.values():
            if 'saved_files' in result:
                files = result['saved_files']
                all_files.extend([
                    files.get('model', ''),
                    files.get('detailed_predictions', ''),
                    files.get('summary_predictions', ''),
                    files.get('analysis_report', '')
                ])
        
        valid_files = [f for f in all_files if f and f != 'N/A']
        if valid_files:
            print(f"  Total files saved: {len(valid_files)}")
            print(f"  File types: Models, Predictions (detailed & summary), Analysis reports")
            print(f"  Use these files for:")
            print(f"    - Business validation of results")
            print(f"    - Identifying data leakage patterns")
            print(f"    - Understanding model performance by platform/brand/store")
            print(f"    - Generating business insights and recommendations")
        else:
            print(f"  No files were saved") 