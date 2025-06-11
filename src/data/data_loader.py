import os
from pathlib import Path
import pandas as pd
from .dataset import DatasetManager

def load_processed_data(data_path: str = None) -> pd.DataFrame:
    """
    Load processed data from the engineered data file
    
    Args:
        data_path: Optional path to the engineered data file.
                  If None, uses default path in data directory.
    
    Returns:
        DataFrame containing the processed data
    """
    if data_path is None:
        # Use default path in data directory
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data',
            'processed',
            'engineered_data.pkl'
        )
    
    # Load data directly
    return pd.read_pickle(data_path) 