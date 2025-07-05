"""
Model evaluation utilities for AutoML Agent.

This module provides functions for evaluating trained models,
calculating metrics, and handling predictions safely.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from typing import Dict, Any, Tuple, Optional
import logging


def safe_metric_calculation(metric_func, y_true, y_pred, **kwargs) -> float:
    """
    Safely calculate a metric with error handling.

    Args:
        metric_func: The metric function to call
        y_true: True labels
        y_pred: Predicted labels
        **kwargs: Additional arguments for the metric function

    Returns:
        float: The calculated metric value, or 0.0 if calculation fails
    """
    try:
        return float(metric_func(y_true, y_pred, **kwargs))
    except Exception as e:
        logging.warning(f"Failed to calculate {metric_func.__name__}: {str(e)}")
        return 0.0


def calculate_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict[str, float]: Dictionary containing all classification metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = safe_metric_calculation(accuracy_score, y_true, y_pred)
    metrics["balanced_accuracy"] = safe_metric_calculation(balanced_accuracy_score, y_true, y_pred)

    # Precision, Recall, F1 with weighted average
    metrics["precision"] = safe_metric_calculation(precision_score, y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall"] = safe_metric_calculation(recall_score, y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1"] = safe_metric_calculation(f1_score, y_true, y_pred, average="weighted", zero_division=0)

    # ROC AUC (only for binary classification or if it works for multiclass)
    metrics["roc_auc"] = safe_metric_calculation(roc_auc_score, y_true, y_pred)

    return metrics


def calculate_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dict[str, float]: Dictionary containing all regression metrics
    """
    metrics = {}

    # Basic regression metrics
    mse = safe_metric_calculation(mean_squared_error, y_true, y_pred)
    metrics["mse"] = mse
    metrics["rmse"] = np.sqrt(mse) if mse >= 0 else 0.0
    metrics["mae"] = safe_metric_calculation(mean_absolute_error, y_true, y_pred)
    metrics["r2"] = safe_metric_calculation(r2_score, y_true, y_pred)

    return metrics


def evaluate_model_predictions(model: Any, X_test, y_test, task_type: str) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test targets
        task_type: Type of ML task ("classification" or "regression")

    Returns:
        Dict[str, Any]: Dictionary containing metrics and any errors
    """
    result = {"success": False, "metrics": {}, "error": None}

    # Check if model exists
    if model is None:
        result["error"] = "No model provided for evaluation"
        return result

    # Check if model has predict method
    if not (hasattr(model, "predict") and callable(getattr(model, "predict"))):
        result["error"] = f"Model type '{type(model).__name__}' does not have a predict method"
        return result

    try:
        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics based on task type
        if task_type == "classification":
            metrics = calculate_classification_metrics(y_test, predictions)
        elif task_type == "regression":
            metrics = calculate_regression_metrics(y_test, predictions)
        else:
            result["error"] = f"Unsupported task type: '{task_type}'"
            return result

        result["success"] = True
        result["metrics"] = metrics

    except Exception as e:
        result["error"] = f"Error during model evaluation: {str(e)}"

    return result


def get_default_metrics(task_type: str) -> Dict[str, float]:
    """
    Get default metrics dictionary for when evaluation fails.

    Args:
        task_type: Type of ML task

    Returns:
        Dict[str, float]: Dictionary with default metric values
    """
    if task_type == "classification":
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "balanced_accuracy": 0.0,
        }
    elif task_type == "regression":
        return {
            "mse": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": 0.0,
        }
    else:
        return {"error": f"Unknown task type: {task_type}"}
