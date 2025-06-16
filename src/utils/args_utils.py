"""
Shared argument parsing utilities to reduce redundant argument definitions.
"""

import argparse
from typing import Optional

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments used across multiple scripts"""
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )

def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments"""
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=512,
        help="Training batch size"
    )

def add_output_args(parser: argparse.ArgumentParser, default_output: str = "outputs") -> None:
    """Add output-related arguments"""
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=default_output,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--experiment-name", 
        type=str,
        help="Name for the experiment (auto-generated if not provided)"
    ) 