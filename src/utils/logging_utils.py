"""
Simplified logging utilities to reduce redundant print statements.
"""

import logging
from typing import Optional, Dict, Any

def log_model_info(logger: logging.Logger, model_info: Dict[str, Any], level: str = "INFO") -> None:
    """Log model information in a concise format"""
    log_func = getattr(logger, level.lower())
    log_func(f"Model: {model_info.get('parameters', 'N/A'):,} parameters")
    log_func(f"Performance: MAPE {model_info.get('mape', 0):.2f}%, R² {model_info.get('r2', 0):.3f}")

def log_progress(logger: logging.Logger, current: int, total: int, description: str = "") -> None:
    """Log progress in a concise format"""
    percentage = (current / total) * 100
    logger.info(f"{description} Progress: {current}/{total} ({percentage:.1f}%)")

def log_experiment_summary(logger: logging.Logger, summary: Dict[str, Any]) -> None:
    """Log experiment summary in a standardized format"""
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    
    for key, value in summary.items():
        if isinstance(value, float):
            if 'mape' in key.lower():
                logger.info(f"{key}: {value:.2f}%")
            else:
                logger.info(f"{key}: {value:.3f}")
        else:
            logger.info(f"{key}: {value}")

def suppress_verbose_logs():
    """Suppress verbose third-party logging"""
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING) 