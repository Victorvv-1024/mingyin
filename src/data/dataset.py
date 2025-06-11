import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecastDataset(Dataset):
    """Dataset class for sales forecasting"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 features: List[str],
                 target_col: str = 'sales_quantity',
                 is_training: bool = True):
        """
        Initialize the dataset
        
        Args:
            data: DataFrame containing the data
            features: List of feature column names
            target_col: Name of the target column
            is_training: Whether this is for training (affects data augmentation)
        """
        self.data = data
        self.features = features
        self.target_col = target_col
        self.is_training = is_training
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Prepare feature data
        self.X = data[features].values
        self.y = data[target_col].values
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(features),
            'feature_names': features,
            'date_range': (data['sales_month'].min(), data['sales_month'].max())
        }
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (features, target)
        """
        # Get features and target
        X = self.X[idx]
        y = self.y[idx]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor([y])
        
        return X_tensor, y_tensor
    
    def get_metadata(self) -> Dict:
        """Get dataset metadata"""
        return self.metadata

class DatasetManager:
    """Manages dataset creation and splitting"""
    
    def __init__(self, 
                 engineered_data_path: str,
                 split_strategy: str = 'split_4_test',
                 random_seed: int = 42):
        """
        Initialize dataset manager
        
        Args:
            engineered_data_path: Path to engineered data pickle file
            split_strategy: Strategy for splitting data
            random_seed: Random seed for reproducibility
        """
        self.engineered_data_path = Path(engineered_data_path)
        self.split_strategy = split_strategy
        self.random_seed = random_seed
        
        # Load engineered data
        self.data = pd.read_pickle(self.engineered_data_path)
        logger.info(f"Loaded engineered data from: {engineered_data_path}")
        
        # Set random seed
        np.random.seed(random_seed)
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = ['sales_quantity', 'sales_amount', 'sales_month', 
                       'store_name', 'brand_name', 'primary_platform',
                       'secondary_platform', 'product_code']
        
        return [col for col in self.data.columns 
                if col not in exclude_cols 
                and not col.startswith('Unnamed')]
    
    def create_split_4_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split for strategy 4"""
        # Train: 2021 + 2022H1
        train_end = '2022-07-01'
        # Test: 2022 Q4
        test_start = '2022-10-01'
        test_end = '2023-01-01'
        
        train_mask = self.data['sales_month'] < train_end
        test_mask = (self.data['sales_month'] >= test_start) & \
                   (self.data['sales_month'] < test_end)
        
        train_data = self.data[train_mask].copy()
        test_data = self.data[test_mask].copy()
        
        return train_data, test_data
    
    def create_split_3_replication(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split for strategy 3"""
        train_end = '2022-07-01'
        test_start = '2022-07-01'
        test_end = '2022-10-01'
        
        train_mask = self.data['sales_month'] < train_end
        test_mask = (self.data['sales_month'] >= test_start) & \
                   (self.data['sales_month'] < test_end)
        
        train_data = self.data[train_mask].copy()
        test_data = self.data[test_mask].copy()
        
        return train_data, test_data
    
    def create_datasets(self) -> Tuple[SalesForecastDataset, 
                                     SalesForecastDataset]:
        """
        Create train and test datasets based on split strategy
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Get feature columns
        features = self.get_feature_columns()
        
        # Create splits based on strategy
        if self.split_strategy == 'split_4_test':
            train_data, test_data = self.create_split_4_test()
        elif self.split_strategy == 'split_3_replication':
            train_data, test_data = self.create_split_3_replication()
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        
        # Create datasets
        train_dataset = SalesForecastDataset(
            train_data,
            features,
            is_training=True
        )
        
        test_dataset = SalesForecastDataset(
            test_data,
            features,
            is_training=False
        )
        
        logger.info(f"Created datasets:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, 
                          batch_size: int = 32,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test dataloaders
        
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_dataloader, test_dataloader)
        """
        train_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, test_loader
    
    def create_rolling_time_series_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """
        Create rolling time series splits for robust model training and validation.
        
        Returns:
            List of tuples (train_data, val_data, description)
        """
        logger.info("Creating rolling time series splits...")
        
        df_sorted = self.data.sort_values('sales_month').copy()
        rolling_splits = []
        
        # Split 1: 2021 full → 2022 Q1 (3 months validation)
        train_1 = df_sorted[df_sorted['sales_month'].dt.year == 2021].copy()
        val_1 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([1, 2, 3]))
        ].copy()
        
        if len(train_1) > 0 and len(val_1) > 0:
            rolling_splits.append((train_1, val_1, "2021_full → 2022_Q1"))
            logger.info(f"Split 1: Train {len(train_1):,} samples, Val {len(val_1):,} samples")
        
        # Split 2: 2021 full + 2022 Q1 → 2022 Q2 (3 months validation)
        train_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3])))
        ].copy()
        val_2 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([4, 5, 6]))
        ].copy()
        
        if len(train_2) > 0 and len(val_2) > 0:
            rolling_splits.append((train_2, val_2, "2021_full+2022_Q1 → 2022_Q2"))
            logger.info(f"Split 2: Train {len(train_2):,} samples, Val {len(val_2):,} samples")
        
        # Split 3: 2021 full + 2022 H1 → 2022 Q3 (3 months validation)
        train_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3, 4, 5, 6])))
        ].copy()
        val_3 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([7, 8, 9]))
        ].copy()
        
        if len(train_3) > 0 and len(val_3) > 0:
            rolling_splits.append((train_3, val_3, "2021_full+2022_H1 → 2022_Q3"))
            logger.info(f"Split 3: Train {len(train_3):,} samples, Val {len(val_3):,} samples")
        
        # Split 4: 2021 full + 2022 Q1-Q3 → 2022 Q4 (3 months validation)
        train_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2021) |
            ((df_sorted['sales_month'].dt.year == 2022) & 
             (df_sorted['sales_month'].dt.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9])))
        ].copy()
        val_4 = df_sorted[
            (df_sorted['sales_month'].dt.year == 2022) & 
            (df_sorted['sales_month'].dt.month.isin([10, 11, 12]))
        ].copy()
        
        if len(train_4) > 0 and len(val_4) > 0:
            rolling_splits.append((train_4, val_4, "2021_full+2022_Q1Q2Q3 → 2022_Q4"))
            logger.info(f"Split 4: Train {len(train_4):,} samples, Val {len(val_4):,} samples")
        
        # Optional Split 5: Include some 2023 data if available
        available_2023 = df_sorted[df_sorted['sales_month'].dt.year == 2023]
        if len(available_2023) > 0:
            train_5 = df_sorted[
                (df_sorted['sales_month'].dt.year == 2021) |
                (df_sorted['sales_month'].dt.year == 2022)
            ].copy()
            val_5 = df_sorted[
                (df_sorted['sales_month'].dt.year == 2023) & 
                (df_sorted['sales_month'].dt.month.isin([1, 2, 3]))
            ].copy()
            
            if len(val_5) > 0:
                rolling_splits.append((train_5, val_5, "2021_full+2022_full → 2023_Q1"))
                logger.info(f"Split 5: Train {len(train_5):,} samples, Val {len(val_5):,} samples")
        
        logger.info(f"Created {len(rolling_splits)} rolling time series splits")
        
        # Validate splits for temporal integrity
        for i, (train_data, val_data, description) in enumerate(rolling_splits):
            train_max_date = train_data['sales_month'].max()
            val_min_date = val_data['sales_month'].min()
            
            if train_max_date >= val_min_date:
                logger.warning(f"Split {i+1} has temporal overlap! Train max: {train_max_date}, Val min: {val_min_date}")
            else:
                logger.info(f"Split {i+1} temporal integrity: ✓ (gap: {(val_min_date - train_max_date).days} days)")
        
        return rolling_splits 