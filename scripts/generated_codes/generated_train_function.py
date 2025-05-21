from typing import Any
from ConfigSpace import Configuration
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a RandomForestClassifier model on the given dataset using the provided
    configuration and returns the average training loss over 10 epochs (cross-validation folds).

    Args:
        cfg (Configuration): A ConfigSpace Configuration object containing the hyperparameters
            for the RandomForestClassifier.
        dataset (Any): A dictionary containing the training data, with keys 'X' for the
            feature matrix and 'y' for the label vector.
        seed (int): The random seed to use for training.

    Returns:
        float: The average training loss (negative accuracy) over 10 cross-validation folds.
    """

    X = dataset["X"]
    y = dataset["y"]

    # Instantiate the RandomForestClassifier with hyperparameters from the configuration
    model = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators"),
        max_depth=cfg.get("max_depth"),
        min_samples_split=cfg.get("min_samples_split"),
        min_samples_leaf=cfg.get("min_samples_leaf"),
        criterion=cfg.get("criterion"),
        random_state=seed,
        n_jobs=-1,  # Use all available cores
    )

    # Perform cross-validation and calculate the average negative accuracy (loss)
    try:
        scores = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error", n_jobs=-1)
        loss = -np.mean(scores)  # Convert negative accuracy to loss
    except Exception as e:
        # Handle potential errors during cross-validation (e.g., due to invalid hyperparameter combinations)
        print(f"Error during cross-validation: {e}")
        loss = float("inf")  # Assign a high loss value to indicate failure

    return loss
