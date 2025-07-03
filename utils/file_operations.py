"""
File operation utilities for AutoML Agent.

This module provides functions for file operations including code extraction,
file saving, and dataset conversion to CSV format.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .data_processing import validate_dataset
from .types import ValidationError


def extract_code_block(code: str) -> str:
    """
    Extract Python code between markdown code fences.

    Args:
        code: String containing markdown code blocks

    Returns:
        Extracted Python code or original string if no code blocks found
    """
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1).strip() if match else code


def save_code_to_file(code: str, filename: str, directory: Optional[str] = None) -> Path:
    """
    Save generated code to file.

    Args:
        code: Code string to save
        filename: Name of the file
        directory: Directory to save file in (optional)

    Returns:
        Path to saved file

    Raises:
        OSError: If file cannot be written
    """
    if directory:
        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    file_path = output_path / filename

    try:
        # if file exists, delete it
        if file_path.exists():
            file_path.unlink()
        file_path.write_text(code)
        return file_path
    except OSError as e:
        raise OSError(f"Failed to save code to {file_path}: {e}")


def convert_dataset_to_csv(dataset: Dict[str, Any], output_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Convert dataset to CSV files.

    Args:
        dataset: Dataset dictionary
        output_dir: Output directory (optional)

    Returns:
        Tuple of paths to features and target CSV files

    Raises:
        ValidationError: If dataset format is unsupported
        OSError: If files cannot be written
    """
    try:
        validate_dataset(dataset)
        X, y = dataset["X"], dataset["y"]

        output_path = Path(output_dir) if output_dir else Path.cwd()
        output_path.mkdir(parents=True, exist_ok=True)

        features_path = output_path / "dataset.csv"
        target_path = output_path / "target.csv"

        pd.DataFrame(X).to_csv(features_path, index=False)
        pd.Series(y).to_csv(target_path, index=False, header=True)  # Ensure header for series

        return features_path, target_path

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise OSError(f"Failed to convert dataset to CSV: {e}")
