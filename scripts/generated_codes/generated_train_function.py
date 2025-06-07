from typing import Any
from ConfigSpace import Configuration
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a RandomForestClassifier on the given dataset using the provided configuration.

    Args:
        cfg (Configuration): A ConfigSpace Configuration object containing hyperparameters for the RandomForestClassifier.
        dataset (Any): A dictionary containing the training data, with 'X' for features and 'y' for labels.
        seed (int): Random seed for reproducibility.

    Returns:
        float: The average training loss (negative cross-validation score) over 10 epochs.
    """
    X = dataset["X"]
    y = dataset["y"]

    # Data Preprocessing: StandardScaler
    scaler = StandardScaler()

    # Model: RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators"),
        max_depth=cfg.get("max_depth"),
        min_samples_split=cfg.get("min_samples_split"),
        min_samples_leaf=cfg.get("min_samples_leaf"),
        random_state=seed,
        n_jobs=1,  # Explicitly set n_jobs to 1 for consistency
    )
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    # Cross-validation (10-fold)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring="neg_log_loss", n_jobs=1)

    # Return average loss
    loss = -np.mean(cv_scores)  # Convert negative log-loss to positive loss

    return loss
