"""
File operation utilities for AutoML Agent.

This module provides functions for file operations including code extraction,
file saving, and dataset conversion to CSV format.
"""

import openml
import os
import numpy as np
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

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


def save_code_to_file(
    code: str, filename: str, directory: Optional[str] = None
) -> Path:
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


def convert_dataset_to_csv(
    dataset: Dict[str, Any], output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
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
        pd.Series(y).to_csv(
            target_path, index=False, header=True
        )  # Ensure header for series

        return features_path, target_path

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise OSError(f"Failed to convert dataset to CSV: {e}")


def load_dataset(
    dataset_origin: str, dataset_id: str | int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset from different origins. Returns a tuple of (X, y) where X is a pandas DataFrame and y is a pandas Series.

    Args:
        dataset_origin: Origin of the dataset (e.g. "openml", "kaggle", "csv")
        dataset_id: ID of the dataset

    Returns:
        Tuple of (X, y) where X is a pandas DataFrame and y is a pandas Series.
    """
    if dataset_origin == "openml":
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        X = pd.DataFrame(X, columns=pd.Index(attribute_names))
        y = pd.DataFrame({"target": y})
    elif dataset_origin == "kaggle":
        raise NotImplementedError("Kaggle datasets are not supported yet")
    elif dataset_origin == "csv":
        raise NotImplementedError("CSV datasets are not supported yet")
    else:
        raise ValueError(f"Unsupported dataset origin: {dataset_origin}")

    return X, y


def load_image_dataset(
    dataset_origin: str, dataset_id: str | int, overwrite: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load image dataset from different origins. Returns a tuple of (X, y) where X is a pandas DataFrame and y is a pandas Series.
    """
    if dataset_origin == "openml":
        path = get_imagedata_path(dataset_origin, dataset_id)
        if os.path.exists(path) and overwrite:
            dataset = openml.datasets.get_dataset(dataset_id)
            X_openml, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )
            X_openml = pd.DataFrame(X_openml, columns=pd.Index(attribute_names))
            X_openml = X_openml.values.reshape(-1, 3, 32, 32)
            X_openml = np.transpose(X_openml, (0, 2, 3, 1))

            X = []
            # Convert to a list of jgps
            from PIL import Image

            images = [Image.fromarray(image) for image in X_openml]
            # Save jgps to folder
            for i, image in enumerate(images):
                image.save(f"{path}/{str(i)}.jpg")
                X.append(f"{path}/{str(i)}.jpg")
            X = pd.DataFrame(X, columns=pd.Index(["image_path"]))
            y = pd.DataFrame({"target": y})
            y = y.astype({"target": "int64"})
            X.to_csv(f"{path}/X.csv", index=False)
            y.to_csv(f"{path}/y.csv", index=False)
        else:
            X = pd.read_csv(f"{path}/X.csv")
            y = pd.read_csv(f"{path}/y.csv")
    else:
        raise ValueError(f"Unsupported dataset origin: {dataset_origin}")

    return X, y


def get_imagedata_path(dataset_origin: str, dataset_id: str | int) -> str:
    """
    Get the path to the dataset.
    """
    return f"data/images/{dataset_origin}/{dataset_id}"


def load_kaggle_dataset(
    dataset_id: str | int, file_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load kaggle dataset.
    """
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
