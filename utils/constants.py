"""
Constants and configuration values for AutoML Agent.

This module contains all constant values, default parameters, and mappings
used throughout the AutoML Agent system.
"""

# Default parameters
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
    "image": ["classification", "clustering", "regression"],
    "categorical": ["classification", "clustering"],
    "text": ["clustering", "classification", "regression"],
}
