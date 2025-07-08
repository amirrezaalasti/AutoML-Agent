"""
Type definitions and custom exceptions for AutoML Agent.

This module contains dataclasses, custom exceptions, and type definitions
used throughout the AutoML Agent system.
"""

from dataclasses import dataclass
from typing import List, Optional

from .constants import SUPPORTED_DATASET_TYPES


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
