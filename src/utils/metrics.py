import numpy as np

def safe_mape_calculation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) safely
    
    Args:
        y_true: True values (log scale)
        y_pred: Predicted values (log scale)
    
    Returns:
        MAPE value (percentage)
    """
    # Convert to original scale
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    
    # Clip predictions to avoid division by zero
    y_pred_orig = np.clip(y_pred_orig, 0.1, 1e6)
    y_true_orig = np.clip(y_true_orig, 0.1, 1e6)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1.0
    
    # Calculate MAPE
    ape = np.abs(y_true_orig - y_pred_orig) / (y_true_orig + epsilon)
    mape = np.mean(ape) * 100
    
    # Cap MAPE at 1000% to avoid extreme values
    return min(mape, 1000.0) 