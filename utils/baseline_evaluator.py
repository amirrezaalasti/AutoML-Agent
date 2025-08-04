"""
AutoGluon baseline evaluator for setting minimum performance targets
"""

import pandas as pd
from pathlib import Path
import os
import shutil
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

        # Prepare data
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Create temporary directory for AutoGluon
        temp_dir = Path("./temp_autogluon_baseline")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Initialize predictor
            predictor = TabularPredictor(label=y_train.name, path=str(temp_dir), verbosity=0)

            # Train with limited time budget
            predictor.fit(
                train_data=train_data,
                presets="best_quality",
                time_limit=self.time_budget,
                num_cpus=min(4, os.cpu_count() or 1),  # Limit CPU usage
                verbosity=0,
            )

            # Evaluate
            results = predictor.evaluate(test_data, silent=True)

            return results

        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)

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

        # Create temporary directory
        temp_dir = Path("./temp_autogluon_multimodal_baseline")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Initialize predictor
            predictor = MultiModalPredictor(label=y_train.name, path=str(temp_dir), verbosity=0)

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
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)


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
