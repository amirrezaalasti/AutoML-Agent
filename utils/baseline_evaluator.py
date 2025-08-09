"""
AutoGluon baseline evaluator for setting minimum performance targets
"""

import pandas as pd
from pathlib import Path
import os
import shutil
import uuid
import atexit
from typing import Dict, Optional
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor


class AutoGluonBaseline:
    """AutoGluon baseline evaluator to set minimum performance targets"""

    def __init__(self, dataset_type: str, time_budget: int = 300):
        """
        Initialize AutoGluon baseline evaluator

        Args:
            dataset_type: Type of dataset ('tabular' or 'image')
            time_budget: Time budget in seconds for AutoGluon training
        """
        self.dataset_type = dataset_type
        self.time_budget = time_budget
        self.baseline_accuracy: Optional[float] = None
        self.baseline_metrics: Optional[Dict] = None

    def evaluate_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """
        Evaluate baseline performance using AutoGluon

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary with baseline metrics
        """
        if self.dataset_type == "tabular":
            return self._evaluate_tabular_baseline(X_train, y_train, X_test, y_test)
        elif self.dataset_type == "image":
            return self._evaluate_image_baseline(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _evaluate_tabular_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Evaluate baseline for tabular data using AutoGluon TabularPredictor"""

        # Prepare data and ensure label column has a valid name
        label_name = y_train.name if y_train.name else "target"
        train_data = pd.concat([X_train, y_train], axis=1).copy()
        test_data = pd.concat([X_test, y_test], axis=1).copy()
        if y_train.name != label_name:
            # y had no name; rename the last column to a stable label name
            train_data.columns = list(train_data.columns[:-1]) + [label_name]
            test_data.columns = list(test_data.columns[:-1]) + [label_name]

        # Create a unique, absolute temporary directory for AutoGluon to avoid races
        temp_parent = Path(os.path.abspath("./temp_autogluon_baseline"))
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_parent / f"run_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize predictor
            predictor = TabularPredictor(
                label=label_name, path=str(temp_dir), verbosity=0
            )

            # Train with limited time budget
            predictor.fit(
                train_data=train_data,
                presets="good_quality",
                time_limit=self.time_budget,
                num_cpus=min(4, os.cpu_count() or 1),  # Limit CPU usage
                verbosity=0,
            )

            # Evaluate
            results = predictor.evaluate(test_data, silent=True)

            return results

        finally:
            # Defer cleanup to process exit to prevent races with any background file access
            atexit.register(
                lambda p=str(temp_dir): shutil.rmtree(p, ignore_errors=True)
            )

    def _evaluate_image_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Evaluate baseline for image data using AutoGluon MultiModalPredictor"""

        # Prepare data
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Create a unique, absolute temporary directory
        temp_parent = Path(os.path.abspath("./temp_autogluon_multimodal_baseline"))
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_parent / f"run_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize predictor
            predictor = MultiModalPredictor(
                label=y_train.name, path=str(temp_dir), verbosity=0
            )

            # Train with limited time budget
            predictor.fit(
                train_data=train_data,
                presets="best_quality",
                time_limit=self.time_budget,
                verbosity=0,
            )

            # Evaluate
            results = predictor.evaluate(test_data, silent=True)
            return results

        finally:
            # Defer cleanup to process exit
            atexit.register(
                lambda p=str(temp_dir): shutil.rmtree(p, ignore_errors=True)
            )


def create_baseline_evaluator(dataset_type: str, time_budget: int = 300):
    """
    Factory function to create appropriate baseline evaluator

    Args:
        dataset_type: Type of dataset ('tabular' or 'image')
        time_budget: Time budget for evaluation

    Returns:
        Baseline evaluator instance
    """

    # If imports succeed, use AutoGluon baseline
    return AutoGluonBaseline(dataset_type=dataset_type, time_budget=time_budget)
