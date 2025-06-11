"""
Streamlined data preprocessing for sales forecasting.

This module provides a clean, efficient preprocessing pipeline that:
1. Loads raw data from Excel files
2. Translates columns and maps platforms
3. Handles basic data cleaning
4. Prepares data for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from .utils import (
    setup_logging, translate_columns, map_platforms, 
    clean_infinite_values, validate_data_integrity
)

logger = setup_logging()

class SalesDataProcessor:
    """Streamlined processor for raw sales data."""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Initialize the processor.
        
        Args:
            raw_data_dir: Directory containing raw Excel files
            output_dir: Directory for saving processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, years: List[int] = [2021, 2022, 2023]) -> pd.DataFrame:
        """
        Load and combine raw Excel data files.
        
        Args:
            years: List of years to load
            
        Returns:
            Combined DataFrame with raw data
        """
        logger.info(f"Loading raw data for years: {years}")
        
        dfs = []
        for year in years:
            file_path = self.raw_data_dir / f"{year}.xlsx"
            
            if file_path.exists():
                df = pd.read_excel(file_path)
                dfs.append(df)
                logger.info(f"Loaded {year}.xlsx: {df.shape}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not dfs:
            raise ValueError("No data files found!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned and standardized DataFrame
        """
        logger.info("Cleaning and standardizing data...")
        
        # Translate columns to English
        df = translate_columns(df)
        
        # Map platform names to English
        df = map_platforms(df)
        
        # Convert sales_month to datetime
        df['sales_month'] = pd.to_datetime(df['sales_month'])
        
        # Calculate unit price
        df['unit_price'] = df['sales_amount'] / df['sales_quantity']
        
        # Clean infinite values
        df = clean_infinite_values(df)
        
        # Handle missing values strategically
        df['store_name'] = df['store_name'].fillna('Unknown_Store')
        df['brand_name'] = df['brand_name'].fillna('Unknown_Brand')
        df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        
        logger.info("✓ Data cleaning completed")
        return df
    
    def perform_quality_checks(self, df: pd.DataFrame) -> Dict:
        """
        Perform data quality validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Performing data quality checks...")
        
        quality_metrics = validate_data_integrity(df)
        
        # Additional business logic checks
        quality_metrics.update({
            'platforms': df['primary_platform'].unique().tolist(),
            'date_coverage_months': len(df['sales_month'].dt.to_period('M').unique()),
            'avg_sales_per_record': df['sales_quantity'].mean(),
            'total_sales_value': df['sales_amount'].sum()
        })
        
        # Flag potential issues
        issues = []
        if quality_metrics['negative_sales'] > 0:
            issues.append(f"{quality_metrics['negative_sales']} records with negative sales")
        if quality_metrics['zero_sales'] > df.shape[0] * 0.1:  # More than 10% zero sales
            issues.append(f"High proportion of zero sales: {quality_metrics['zero_sales']}")
        
        quality_metrics['issues'] = issues
        
        if issues:
            logger.warning(f"Data quality issues found: {issues}")
        else:
            logger.info("✓ Data quality checks passed")
        
        return quality_metrics
    
    def save_processed_data(self, df: pd.DataFrame, quality_metrics: Dict) -> str:
        """
        Save processed data and metadata.
        
        Args:
            df: Processed DataFrame
            quality_metrics: Quality check results
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed data
        output_file = self.output_dir / f"processed_data_{timestamp}.pkl"
        df.to_pickle(output_file)
        
        # Save metadata
        metadata_file = self.output_dir / f"processing_metadata_{timestamp}.txt"
        with open(metadata_file, 'w') as f:
            f.write("SALES DATA PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing timestamp: {timestamp}\n")
            f.write(f"Total records: {len(df):,}\n")
            f.write(f"Date range: {df['sales_month'].min()} to {df['sales_month'].max()}\n")
            f.write(f"Platforms: {', '.join(quality_metrics['platforms'])}\n\n")
            
            f.write("Quality Metrics:\n")
            for key, value in quality_metrics.items():
                if key != 'issues':
                    f.write(f"  {key}: {value}\n")
            
            if quality_metrics['issues']:
                f.write(f"\nIssues Found:\n")
                for issue in quality_metrics['issues']:
                    f.write(f"  - {issue}\n")
        
        logger.info(f"Processed data saved to: {output_file}")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return str(output_file)
    
    def process(self, years: List[int] = [2021, 2022, 2023]) -> Tuple[str, Dict]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            years: Years to process
            
        Returns:
            Tuple of (output_file_path, quality_metrics)
        """
        logger.info("=" * 60)
        logger.info("STARTING SALES DATA PREPROCESSING")
        logger.info("=" * 60)
        
        try:
            # 1. Load raw data
            raw_df = self.load_raw_data(years)
            
            # 2. Clean and standardize
            processed_df = self.clean_and_standardize(raw_df)
            
            # 3. Quality checks
            quality_metrics = self.perform_quality_checks(processed_df)
            
            # 4. Save results
            output_file = self.save_processed_data(processed_df, quality_metrics)
            
            logger.info("=" * 60)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return output_file, quality_metrics
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

def main():
    """Main function for standalone execution."""
    processor = SalesDataProcessor("data/raw", "data/processed")
    output_file, quality_metrics = processor.process()
    print(f"Processing completed. Output: {output_file}")

if __name__ == "__main__":
    main() 