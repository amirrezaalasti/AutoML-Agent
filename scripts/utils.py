"""
Utility functions for AutoML Agent.

This module provides utility functions for dataset handling, description generation,
hyperparameter optimization, and various data processing tasks.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_SMAC_BUDGET = 100
DEFAULT_SMAC_TRIALS = 10
DEFAULT_SMAC_METRIC = "accuracy"

# Supported dataset types
SUPPORTED_DATASET_TYPES = {"tabular", "time_series", "image", "categorical", "text"}

# Task types mapping
DATASET_TASK_MAPPING = {
    "tabular": ["classification", "regression", "clustering"],
    "time_series": ["clustering", "regression", "classification"],
    "image": ["classification", "clustering"],
    "categorical": ["classification", "clustering"],
    "text": ["clustering", "classification"],
}


@dataclass
class DatasetInfo:
    """Data class to hold dataset information."""

    n_samples: int
    n_features: int
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None
    dataset_type: str = "tabular"

    def __post_init__(self):
        """Validate dataset info after initialization."""
        if self.dataset_type not in SUPPORTED_DATASET_TYPES:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")


class DatasetError(Exception):
    """Custom exception for dataset-related errors."""

    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_dataset(dataset: Any) -> None:
    """
    Validate dataset format and structure.

    Args:
        dataset: Dataset to validate

    Raises:
        ValidationError: If dataset format is invalid
    """
    if dataset is None:
        raise ValidationError("Dataset cannot be None")

    if isinstance(dataset, dict):
        required_keys = {"X", "y"}
        if not all(key in dataset for key in required_keys):
            raise ValidationError(f"Dataset dict must contain keys: {required_keys}")

        if dataset["X"] is None or dataset["y"] is None:
            raise ValidationError("Dataset X and y cannot be None")

    elif isinstance(dataset, pd.DataFrame):
        if "target" not in dataset.columns:
            raise ValidationError("DataFrame must include a 'target' column")

    elif isinstance(dataset, np.ndarray):
        if dataset.size == 0:
            raise ValidationError("NumPy array cannot be empty")

    else:
        raise ValidationError(f"Unsupported dataset type: {type(dataset)}")


def validate_dataset_type(dataset_type: str) -> None:
    """
    Validate dataset type.

    Args:
        dataset_type: Type of dataset to validate

    Raises:
        ValidationError: If dataset type is not supported
    """
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ValidationError(f"Unsupported dataset type: {dataset_type}. " f"Supported types: {SUPPORTED_DATASET_TYPES}")


def get_dataset_info(dataset: Dict[str, Any]) -> DatasetInfo:
    """
    Extract basic information from dataset.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys

    Returns:
        DatasetInfo: Dataset information object

    Raises:
        ValidationError: If dataset is invalid
    """
    validate_dataset(dataset)

    X, y = dataset["X"], dataset["y"]

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        target_name = y.name if hasattr(y, "name") else None
    else:
        feature_names = None
        target_name = None

    return DatasetInfo(
        n_samples=X.shape[0],
        n_features=X.shape[1] if X.ndim > 1 else 1,
        feature_names=feature_names,
        target_name=target_name,
    )


def format_dataset(dataset: Any) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Format dataset into standardized dictionary format.

    Args:
        dataset: Input dataset in various formats

    Returns:
        Dict containing 'X' (features) and 'y' (target) as pandas objects

    Raises:
        ValidationError: If dataset format is unsupported
    """
    try:
        if isinstance(dataset, pd.DataFrame):
            if "target" not in dataset.columns:
                raise ValidationError("DataFrame must include a 'target' column")
            X = dataset.drop(columns=["target"])
            y = dataset["target"]

        elif isinstance(dataset, dict):
            X = dataset["X"]
            y = dataset["y"]

            # Convert NumPy arrays to pandas
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

        elif isinstance(dataset, np.ndarray):
            if dataset.ndim == 1:
                X = pd.DataFrame(dataset.reshape(-1, 1))
            else:
                X = pd.DataFrame(dataset)
            y = None  # No labels provided

        else:
            raise ValidationError(f"Unsupported dataset format: {type(dataset)}")

        return {"X": X, "y": y}

    except Exception as e:
        raise ValidationError(f"Failed to format dataset: {e}")


def split_dataset(
    dataset: Dict[str, Any],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Dict[str, Union[pd.DataFrame, pd.Series]], Dict[str, Union[pd.DataFrame, pd.Series]]]:
    """
    Split dataset into training and testing sets.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility

    Returns:
        Tuple containing train and test data

    Raises:
        ValidationError: If dataset is invalid or split parameters are invalid
    """
    try:
        validate_dataset(dataset)

        if not 0 < test_size < 1:
            raise ValidationError(f"test_size must be between 0 and 1, got {test_size}")

        X, y = dataset["X"], dataset["y"]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Reset indices for DataFrames/Series
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

        if isinstance(y_train, pd.Series):
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

        train_data = {"X": X_train, "y": y_train}
        test_data = {"X": X_test, "y": y_test}

        return train_data, test_data

    except Exception as e:
        raise ValidationError(f"Failed to split dataset: {e}")


def describe_tabular_dataset(dataset: Dict[str, Any]) -> str:
    """
    Generate description for tabular dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        String description of the dataset
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = [
        "This is a tabular dataset.",
        f"It has {info.n_samples} samples and {info.n_features} features.",
    ]

    if hasattr(X, "dtypes"):
        description.append("Feature columns and types:")
        for col, dtype in X.dtypes.items():
            description.append(f"- {col}: {dtype}")

    if hasattr(X, "describe"):
        description.append("\nFeature statistical summary:")
        description.append(str(X.describe(include="all")))

    if hasattr(y, "value_counts"):
        description.append("\nLabel distribution:")
        description.append(str(y.value_counts()))

    return "\n".join(description)


def describe_time_series_dataset(dataset: Dict[str, Any]) -> str:
    """
    Generate description for time series dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        String description of the dataset
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = [
        "This is a time series dataset.",
        f"Number of samples: {info.n_samples}",
    ]

    if hasattr(X, "index"):
        description.extend(
            [
                f"Time index type: {type(X.index)}",
                f"Time range: {X.index.min()} to {X.index.max()}",
            ]
        )

    description.append("Features:")
    description.extend([f"- {col}" for col in X.columns])

    # Add handling requirements
    requirements = [
        "\n### Time Series Handling Requirements",
        "- Assume `dataset['X']` is a 3D array or tensor with shape `(num_samples, sequence_length, num_features)`.",
        "- If `dataset['X']` is 2D, raise a `ValueError` if the model is RNN-based (`LSTM`, `GRU`, `RNN`).",
        "- Do **not** flatten the input when using RNN-based models.",
        "- Use `batch_first=True` in all recurrent models to maintain `(batch, seq_len, features)` format.",
        "- Dynamically infer sequence length as `X.shape[1]` and feature dimension as `X.shape[2]`.",
        "- If `X.ndim != 3` and a sequential model is selected, raise a clear error with shape info.",
        "- Example input validation check:",
        "  ```python",
        "  if model_type in ['LSTM', 'GRU', 'RNN'] and X_tensor.ndim != 3:",
        '      raise ValueError(f"Expected 3D input (batch, seq_len, features) for {model_type}, got {X_tensor.shape}")',
        "  ```",
        "- Time index or datetime values can be logged but should not be used in the model unless specified.",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_image_dataset(dataset: Dict[str, Any]) -> str:
    """
    Generate description for image dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        String description of the dataset
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = ["This is an image dataset."]

    if isinstance(X, np.ndarray):
        if X.ndim == 4:
            n, c, h, w = X.shape if X.shape[1] <= 3 else (X.shape[0], X.shape[3], X.shape[1], X.shape[2])
            description.extend(
                [
                    f"Image format: (N={n}, C={c}, H={h}, W={w})",
                    "Note: Images are in (batch, channels, height, width) format.",
                ]
            )
        elif X.ndim == 3:
            n, h, w = X.shape
            description.extend(
                [
                    f"Image format: (N={n}, H={h}, W={w})",
                    "Note: Images are single-channel in (batch, height, width) format.",
                ]
            )
        elif X.ndim == 2:
            n, pixels = X.shape
            height = int(np.sqrt(pixels))
            if height * height == pixels:
                description.extend(
                    [
                        f"Image format: Flattened {height}x{height} images",
                        f"Shape: (N={n}, pixels={pixels})",
                        "Note: Images are flattened. Must be reshaped for processing.",
                    ]
                )
            else:
                description.extend(
                    [
                        f"Shape: (N={n}, features={pixels})",
                        "Warning: Feature count is not a perfect square.",
                    ]
                )

    if hasattr(y, "nunique"):
        description.extend(
            [
                f"\nNumber of classes: {y.nunique()}",
                "Class distribution:",
                str(y.value_counts()),
            ]
        )

    # Add handling requirements
    requirements = [
        "\nImage Data Handling Requirements:",
        "1. Input Format Requirements:",
        "   - For CNN models: Input must be in (batch, channels, height, width) format",
        "   - For dense/linear layers: Input should be flattened",
        "",
        "2. Data Processing Steps:",
        "   a) For flattened input (2D):",
        "      - Calculate dimensions: height = width = int(sqrt(n_features))",
        "      - Verify square dimensions: height * height == n_features",
        "      - Reshape to (N, 1, H, W) for CNNs",
        "   b) For 3D input (N, H, W):",
        "      - Add channel dimension: reshape to (N, 1, H, W)",
        "   c) For 4D input:",
        "      - Verify channel order matches framework requirements",
        "",
        "3. Framework-Specific Format:",
        "   - PyTorch: (N, C, H, W)",
        "   - TensorFlow: (N, H, W, C)",
        "   - Convert between formats if necessary",
        "",
        "4. Normalization:",
        "   - Scale pixel values to [0, 1] by dividing by 255.0",
        "   - Or standardize to mean=0, std=1",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_categorical_dataset(dataset: Dict[str, Any]) -> str:
    """
    Generate description for categorical dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        String description of the dataset
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = [
        "This is a categorical dataset.",
        f"It has {info.n_samples} samples.",
        "Categorical feature summary:",
    ]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        description.append(f"- {col}: {X[col].nunique()} unique values")

    if hasattr(y, "nunique"):
        description.append(f"\nTarget variable has {y.nunique()} unique classes.")

    # Add handling requirements
    requirements = [
        "\nCategorical Data Handling Requirements:",
        "1. Preprocessing Steps:",
        "   - Encode categorical variables (one-hot or label encoding)",
        "   - Handle missing values",
        "   - Check for cardinality of categorical variables",
        "",
        "2. Model Considerations:",
        "   - Use appropriate encoding for model type",
        "   - Handle high cardinality features appropriately",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_text_dataset(dataset: Dict[str, Any]) -> str:
    """
    Generate description for text dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        String description of the dataset
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = [
        "This is a text dataset.",
        f"Number of text entries: {info.n_samples}",
    ]

    if hasattr(X, "apply"):
        avg_length = X.apply(lambda x: len(str(x))).mean()
        description.append(f"Average text length: {avg_length:.2f} characters")

    if hasattr(y, "nunique"):
        description.append(f"\nTarget variable has {y.nunique()} unique classes.")

    # Add handling requirements
    requirements = [
        "\nText Data Handling Requirements:",
        "1. Preprocessing Steps:",
        "   - Tokenization",
        "   - Convert to lowercase",
        "   - Remove special characters if needed",
        "   - Handle missing values",
        "",
        "2. Feature Extraction:",
        "   - Use TF-IDF or word embeddings",
        "   - Consider max sequence length",
        "   - Handle vocabulary size",
        "",
        "3. Model-Specific Requirements:",
        "   - For neural networks: Use appropriate padding/truncation",
        "   - For traditional ML: Convert to fixed-size features",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_dataset(dataset: Dict[str, Any], dataset_type: str = "tabular") -> str:
    """
    Generate detailed description of dataset based on its type.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        dataset_type: Type of dataset ('tabular', 'time_series', 'image', 'categorical', 'text')

    Returns:
        String description of the dataset

    Raises:
        ValidationError: If dataset type is not supported
    """
    try:
        validate_dataset(dataset)
        validate_dataset_type(dataset_type)

        description_functions = {
            "tabular": describe_tabular_dataset,
            "time_series": describe_time_series_dataset,
            "image": describe_image_dataset,
            "categorical": describe_categorical_dataset,
            "text": describe_text_dataset,
        }

        if dataset_type not in description_functions:
            return "Dataset type not recognized. Please provide a valid dataset type."

        return description_functions[dataset_type](dataset)

    except Exception as e:
        raise ValidationError(f"Failed to describe dataset: {e}")


def generate_dataset_task_types(dataset_type: str) -> List[str]:
    """
    Generate list of task types for given dataset type.

    Args:
        dataset_type: Type of dataset

    Returns:
        List of supported task types

    Raises:
        ValidationError: If dataset type is not supported
    """
    try:
        validate_dataset_type(dataset_type)
        return DATASET_TASK_MAPPING.get(dataset_type, [])
    except Exception as e:
        raise ValidationError(f"Failed to generate task types: {e}")


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
    return match.group(1) if match else code


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
    # if file exists, delete it
    if Path(filename).exists():
        Path(filename).unlink()
    with open(filename, "w") as file:
        file.write(code)


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

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd()

        features_path = output_path / "dataset.csv"
        target_path = output_path / "target.csv"

        if isinstance(X, pd.DataFrame):
            X.to_csv(features_path, index=False)
        elif isinstance(X, np.ndarray):
            pd.DataFrame(X).to_csv(features_path, index=False)
        else:
            raise ValidationError(f"Unsupported X format: {type(X)}")

        if isinstance(y, pd.Series):
            y.to_csv(target_path, index=False)
        elif isinstance(y, np.ndarray):
            pd.Series(y).to_csv(target_path, index=False)
        else:
            raise ValidationError(f"Unsupported y format: {type(y)}")

        return features_path, target_path

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise OSError(f"Failed to convert dataset to CSV: {e}")


save_code = save_code_to_file
generate_general_dataset_task_types = generate_dataset_task_types
convert_to_csv = convert_dataset_to_csv
