"""
Utility functions for AutoML Agent.

This package provides utility functions for dataset handling, description generation,
hyperparameter optimization, and various data processing tasks.
"""

# Import all utility functions to make them available at the package level
from .constants import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SMAC_BUDGET,
    DEFAULT_SMAC_TRIALS,
    DEFAULT_SMAC_METRIC,
    SUPPORTED_DATASET_TYPES,
    DATASET_TASK_MAPPING,
    FACADE_PARAMETER_REQUIREMENTS,
)

from .types import DatasetInfo, DatasetError, ValidationError

from .data_processing import (
    validate_dataset,
    validate_dataset_type,
    get_dataset_info,
    format_dataset,
    split_dataset_kfold,
)

from .dataset_description import (
    describe_dataset,
    describe_tabular_dataset,
    describe_time_series_dataset,
    describe_image_dataset,
    describe_categorical_dataset,
    describe_text_dataset,
    add_general_description,
    generate_dataset_task_types,
)

from .file_operations import (
    extract_code_block,
    save_code_to_file,
    convert_dataset_to_csv,
)

# Version
__version__ = "1.0.0"

# Package level exports
__all__ = [
    # Constants
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_SMAC_BUDGET",
    "DEFAULT_SMAC_TRIALS",
    "DEFAULT_SMAC_METRIC",
    "SUPPORTED_DATASET_TYPES",
    "DATASET_TASK_MAPPING",
    "FACADE_PARAMETER_REQUIREMENTS",
    # Types
    "DatasetInfo",
    "DatasetError",
    "ValidationError",
    # Data Processing
    "validate_dataset",
    "validate_dataset_type",
    "get_dataset_info",
    "format_dataset",
    "split_dataset_kfold",
    # Dataset Description
    "describe_dataset",
    "describe_tabular_dataset",
    "describe_time_series_dataset",
    "describe_image_dataset",
    "describe_categorical_dataset",
    "describe_text_dataset",
    "add_general_description",
    "generate_dataset_task_types",
    # File Operations
    "extract_code_block",
    "save_code_to_file",
    "convert_dataset_to_csv",
]
