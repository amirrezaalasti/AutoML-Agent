"""
Dataset description utilities for AutoML Agent.

This module provides functions for generating detailed descriptions of datasets
based on their type and the machine learning task.
"""

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .constants import DATASET_TASK_MAPPING
from .data_processing import format_dataset, get_dataset_info, validate_dataset_type
from .types import ValidationError


def describe_tabular_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a description for a tabular dataset, tailored to the ML task.

    Args:
        dataset: A preprocessed dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        A string description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    description = [
        "This is a tabular dataset.",
        "NOTE: Categorical features have been pre-processed using one-hot encoding.",
        f"It has {info.n_samples} samples and {info.n_features} features after encoding.",
    ]

    # Task-specific description of the target variable
    if task_type == "classification":
        description.append("The target variable has been label-encoded to be numeric for classification.")
        description.append("\nEncoded label distribution:")
        description.append(str(y.value_counts()))
    elif task_type == "regression":
        description.append("The target variable is continuous for this regression task.")
        # for regression sometimes the target value is not normalized properly, so we need to normalize it
        if y.min() < 0 or y.max() > 1:
            description.append("The target variable is not normalized properly. Please normalize it.")
        description.append("\nTarget variable statistical summary:")
        description.append(str(y.describe()))
    else:  # Clustering or other tasks
        description.append("The target variable is present for evaluation purposes.")

    if hasattr(X, "dtypes"):
        description.append("\nFeature columns and their final data types:")
        description.append(X.dtypes.to_string())

    if hasattr(X, "describe"):
        description.append("\nFeature statistical summary (post-encoding):")
        description.append(str(X.describe(include="all")))

    return "\n".join(description)


def describe_time_series_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a description for a time series dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.
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

    # --- Target Variable Analysis (Task-Specific) ---
    description.append("\n### Target Variable (y) Analysis")
    if task_type == "classification":
        description.append(f"* **Task**: Time series classification.")
        description.append(f"* **Number of Classes**: {y.nunique()}")
        description.append(f"* **Class Distribution**: {y.value_counts().to_dict()}")
    elif task_type == "regression":
        description.append(f"* **Task**: Time series forecasting/regression.")
        description.append(f"* **Target Summary**: \n{y.describe().to_string()}")

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


def describe_image_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a detailed description for an image dataset, tailored to the ML task.

    Args:
        dataset: A dictionary containing 'X' (features) and 'y' (labels).
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        A string description of the dataset with preprocessing advice.
    """
    X, y = dataset["X"], dataset["y"]
    description_parts = []

    # --- Section 1: Data Shape, Type, and Analysis ---
    description_parts.append("### Dataset Analysis")
    if isinstance(X, (pd.DataFrame, pd.Series)):
        description_parts.append(f"* **Input Type**: Pandas {type(X).__name__} with shape {X.shape}.")
        X_np = X.values
    elif isinstance(X, np.ndarray):
        description_parts.append(f"* **Input Type**: NumPy Array with shape {X.shape}.")
        X_np = X
    else:
        return f"Unsupported input type for image dataset: {type(X).__name__}."

    n_samples = X_np.shape[0]

    # --- Section 2: Preprocessing Recommendations ---
    description_parts.append("\n### Preprocessing Recommendations for CNNs")

    if X_np.ndim == 2:  # Flattened images
        n_features = X_np.shape[1]
        description_parts.append(f"* **Data Format**: Flattened images, with {n_samples} samples and {n_features} features each.")
        h = int(math.sqrt(n_features))
        if h * h == n_features:
            description_parts.append(f"* **Action**: Reshape to `(N, 1, {h}, {h})` for grayscale CNNs.")
        else:
            next_h = int(math.ceil(math.sqrt(n_features)))
            pad_needed = next_h * next_h - n_features
            description_parts.append(
                f"* **Action Required**: Features ({n_features}) are not a perfect square. "
                f"Pad with {pad_needed} zeros and reshape to `(N, 1, {next_h}, {next_h})`."
            )
    elif X_np.ndim == 3:  # Grayscale images
        description_parts.append("* **Data Format**: 3D array `(N, H, W)`. Add a channel dimension.")
        description_parts.append("* **Action**: Reshape to `(N, 1, H, W)`.")
    elif X_np.ndim == 4:  # Color images
        description_parts.append("* **Data Format**: 4D array. Verify channel order (`(N, C, H, W)` vs `(N, H, W, C)`).")
    else:
        description_parts.append(f"* **Data Format**: Unexpected {X_np.ndim}D array. Manual inspection required.")

    description_parts.append("\n* **Normalization**: Remember to scale pixel values (e.g., divide by 255.0).")

    # --- Section 3: Target Variable Analysis (Task-Specific) ---
    description_parts.append("\n### Target Variable (y) Analysis")
    if task_type == "classification":
        num_classes = y.nunique() if hasattr(y, "nunique") else len(np.unique(y))
        class_dist = str(y.value_counts().to_dict()) if hasattr(y, "value_counts") else str(dict(zip(*np.unique(y, return_counts=True))))
        description_parts.append(f"* **Task**: Image Classification.")
        description_parts.append(f"* **Number of Classes**: {num_classes}")
        description_parts.append(f"* **Class Distribution**: {class_dist}")
        description_parts.append(
            f"""* **For classification tasks**
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
            metrics = {{
                'loss': loss,
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_test, predictions)
            }}
            """
        )
    elif task_type == "regression":
        description_parts.append(f"* **Task**: Image Regression (e.g., predicting age, angle).")
        description_parts.append(f"* **Target Summary**: \n{y.describe().to_string()}")
        description_parts.append(
            f"""* **For regression tasks**
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {{
                'loss': loss,
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }}
            """
        )

    return "\n".join(description_parts)


def describe_categorical_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate description for categorical dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'clustering', etc.).

    Returns:
        String description of the dataset.
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

    # Task-specific description of the target variable
    if task_type == "classification":
        description.append(f"\nTarget variable has {y.nunique()} unique classes for classification.")
        description.append("Label distribution:")
        description.append(str(y.value_counts()))
    elif task_type == "clustering":
        description.append("\nTarget variable is provided for clustering evaluation.")

    # Add handling requirements
    requirements = [
        "\n### Categorical Data Handling Requirements",
        "- **Encoding**: Ensure categorical features are appropriately encoded (e.g., one-hot, target encoding).",
        "- **High Cardinality**: For features with many unique values, consider target encoding or feature hashing.",
        "- **Model Choice**: Tree-based models (like CatBoost, LightGBM) can handle categorical features natively.",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_text_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate description for text dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset)

    # Assuming X is a Series of text for this description
    if not isinstance(X, pd.Series):
        X = pd.Series(X.squeeze(), name="text_feature")

    description = [
        "This is a text dataset.",
        f"Number of text entries: {info.n_samples}",
        f"Average text length: {X.str.len().mean():.2f} characters",
    ]

    # Task-specific description of the target variable
    description.append("\n### Target Variable (y) Analysis")
    if task_type == "classification":
        description.append(f"* **Task**: Text Classification.")
        description.append(f"* **Number of Classes**: {y.nunique()}")
        description.append(f"* **Class Distribution**: {y.value_counts().to_dict()}")
    elif task_type == "regression":
        description.append(f"* **Task**: Text Regression (e.g., predicting a sentiment score).")
        description.append(f"* **Target Summary**: \n{y.describe().to_string()}")

    # Add handling requirements
    requirements = [
        "\n### Text Data Handling Requirements",
        "1. **Preprocessing**: Tokenization, stop-word removal, and stemming/lemmatization " "are standard.",
        "2. **Feature Extraction**: Convert text to numerical vectors using TF-IDF, word "
        "embeddings (Word2Vec, GloVe), or transformer-based embeddings (BERT).",
        "3. **Model-Specific**: For RNNs/Transformers, ensure inputs are padded/truncated " "to a consistent sequence length.",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_dataset(
    dataset: Dict[str, Any],
    dataset_type: str = "tabular",
    task_type: str = "classification",
) -> str:
    """
    Generate a detailed, task-aware description of a dataset.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys.
        dataset_type: Type of dataset ('tabular', 'time_series', 'image', 'categorical', 'text').
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.

    Raises:
        ValidationError: If dataset type is not supported or processing fails.
    """
    try:
        # First, format the dataset which includes preprocessing
        processed_dataset = format_dataset(dataset)
        validate_dataset_type(dataset_type)

        description_functions = {
            "tabular": describe_tabular_dataset,
            "time_series": describe_time_series_dataset,
            "image": describe_image_dataset,
            "categorical": describe_categorical_dataset,
            "text": describe_text_dataset,
        }

        # Describe the *processed* dataset using the appropriate task-aware function
        description = description_functions[dataset_type](processed_dataset, task_type)
        description = add_general_description(description, processed_dataset)
        return description

    except Exception as e:
        # Re-raise as a validation error for consistent error handling
        raise ValidationError(f"Failed to describe dataset: {e}")


def add_general_description(description: str, dataset: Dict[str, Any]) -> str:
    """
    Add a preview of the processed feature data to the description.
    """
    description += f"\n\nThe dataset features (X) start with the following values:\n{dataset['X'].head().to_string()}"
    return description


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
