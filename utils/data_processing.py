"""
Data processing utilities for AutoML Agent.

This module provides functions for dataset validation, formatting, and splitting
operations used throughout the AutoML Agent system.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
import openml
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from numpy.random import RandomState

from .constants import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from .types import DatasetInfo, ValidationError


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
    from .constants import SUPPORTED_DATASET_TYPES

    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ValidationError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types: {SUPPORTED_DATASET_TYPES}"
        )


def get_dataset_info(dataset: Dict[str, Any], task_type: str) -> DatasetInfo:
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
        target_name = y.name if hasattr(y, "name") else "target"
    else:
        feature_names = (
            [f"feature_{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["feature_0"]
        )
        target_name = "target"

    if isinstance(y, pd.Series):
        # add description of the target variable
        if task_type == "classification":
            target_description = (
                f"This is a {task_type} problem with {y.nunique()} classes."
            )
        elif task_type == "regression":
            target_description = f"This is a {task_type} problem."
        else:
            raise ValidationError(f"Unsupported task type: {task_type}")

    return DatasetInfo(
        n_samples=X.shape[0],
        n_features=X.shape[1] if X.ndim > 1 else 1,
        feature_names=feature_names,
        target_name=target_name,
        target_type=task_type,
        target_description=target_description,
    )


def format_dataset(dataset: Any) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Format dataset into a standardized, preprocessed dictionary.

    This function now handles one-hot encoding for categorical features and
    label encoding for the target variable to prevent downstream errors.

    Args:
        dataset: Input dataset in various formats.

    Returns:
        A dictionary containing preprocessed 'X' (features) and 'y' (target).

    Raises:
        ValidationError: If the dataset format is unsupported or invalid.
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
        else:
            raise ValidationError(f"Unsupported dataset format: {type(dataset)}")

        # Ensure X is a DataFrame for consistent processing
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure y is a Series for consistent processing
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")

        # --- Data Preprocessing Step ---
        # 1. One-hot encode categorical features in X
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(
                X, columns=categorical_cols, drop_first=True, dtype=float
            )

        # 2. Label encode the target variable y if it's a non-numeric type (for classification)
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)

        return {"X": X, "y": y}

    except Exception as e:
        raise ValidationError(f"Failed to format dataset: {e}")


def split_dataset_kfold(
    dataset: Dict[str, Any],
    n_folds: int,
    fold: int,
    rng: RandomState,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split dataset into training and testing sets using K-fold cross-validation.

    Automatically detects task type and uses appropriate cross-validation strategy:
    - For classification: Uses StratifiedKFold to maintain class distribution
    - For regression: Uses regular KFold for continuous targets

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        n_folds: Number of folds for cross-validation
        fold: Current fold to use for train-test split (0-indexed)
        rng: Random state for reproducibility

    Returns:
        Tuple containing train and test data

    Raises:
        ValidationError: If dataset is invalid or split parameters are invalid
    """
    try:
        validate_dataset(dataset)

        X, y = dataset["X"], dataset["y"]

        # Detect task type: classification vs regression
        # For regression: target is continuous (many unique values relative to dataset size)
        # For classification: target is discrete (few unique values)
        unique_values = y.nunique()
        n_samples = len(y)

        # Use heuristic: if unique values > 10 and > 5% of total samples, likely regression
        is_regression = (unique_values > 10 and unique_values > 0.05 * n_samples) or (
            pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y)
        )

        if is_regression:
            # Use regular K-fold for regression
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
            fold_indices = list(kf.split(X, y))
        else:
            # Use stratified K-fold for classification
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng)
            fold_indices = list(skf.split(X, y))

        train_idx, test_idx = fold_indices[fold]

        # Split the data using the fold indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Reset indices for DataFrames/Series to ensure clean alignment
        train_data = {
            "X": X_train.reset_index(drop=True),
            "y": y_train.reset_index(drop=True),
        }
        test_data = {
            "X": X_test.reset_index(drop=True),
            "y": y_test.reset_index(drop=True),
        }

        return train_data, test_data

    except Exception as e:
        raise ValidationError(f"Failed to split dataset: {e}")


def split_dataset_kfold_classification(
    dataset: Dict[str, Any],
    n_folds: int,
    fold: int,
    rng: RandomState,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split dataset into training and testing sets using stratified K-fold cross-validation for classification.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        n_folds: Number of folds for cross-validation
        fold: Current fold to use for train-test split (0-indexed)
        rng: Random state for reproducibility

    Returns:
        Tuple containing train and test data

    Raises:
        ValidationError: If dataset is invalid or split parameters are invalid
    """
    try:
        validate_dataset(dataset)

        X, y = dataset["X"], dataset["y"]

        # Create stratified K-fold splitter for classification
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng)
        fold_indices = list(skf.split(X, y))
        train_idx, test_idx = fold_indices[fold]

        # Split the data using the fold indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Reset indices for DataFrames/Series to ensure clean alignment
        train_data = {
            "X": X_train.reset_index(drop=True),
            "y": y_train.reset_index(drop=True),
        }
        test_data = {
            "X": X_test.reset_index(drop=True),
            "y": y_test.reset_index(drop=True),
        }

        return train_data, test_data

    except Exception as e:
        raise ValidationError(f"Failed to split dataset for classification: {e}")


def split_dataset_kfold_regression(
    dataset: Dict[str, Any],
    n_folds: int,
    fold: int,
    rng: RandomState,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split dataset into training and testing sets using regular K-fold cross-validation for regression.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys
        n_folds: Number of folds for cross-validation
        fold: Current fold to use for train-test split (0-indexed)
        rng: Random state for reproducibility

    Returns:
        Tuple containing train and test data

    Raises:
        ValidationError: If dataset is invalid or split parameters are invalid
    """
    try:
        validate_dataset(dataset)

        X, y = dataset["X"], dataset["y"]

        # Create regular K-fold splitter for regression
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
        fold_indices = list(kf.split(X, y))
        train_idx, test_idx = fold_indices[fold]

        # Split the data using the fold indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Reset indices for DataFrames/Series to ensure clean alignment
        train_data = {
            "X": X_train.reset_index(drop=True),
            "y": y_train.reset_index(drop=True),
        }
        test_data = {
            "X": X_test.reset_index(drop=True),
            "y": y_test.reset_index(drop=True),
        }

        return train_data, test_data

    except Exception as e:
        raise ValidationError(f"Failed to split dataset for regression: {e}")
