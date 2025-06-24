import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a Gradient Boosting Classifier model using the provided configuration and dataset.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): The dataset dictionary containing 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: Negative accuracy score on the test set.
    """

    # Validate dataset
    if not isinstance(dataset, dict):
        raise TypeError("Dataset must be a dictionary.")
    if "X" not in dataset or "y" not in dataset:
        raise ValueError("Dataset must contain 'X' and 'y' keys.")

    X = dataset["X"]
    y = dataset["y"]

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a pandas DataFrame and y must be a pandas Series.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # Validate configuration
    if not isinstance(cfg, Configuration):
        raise TypeError("cfg must be a ConfigSpace Configuration object.")

    # Extract hyperparameters with validation and defaults
    num_estimators = int(cfg.get("num_estimators"))
    if not isinstance(num_estimators, int) or num_estimators < 50 or num_estimators > 200:
        raise ValueError("num_estimators must be an integer between 50 and 200.")

    max_depth = int(cfg.get("max_depth"))
    if not isinstance(max_depth, int) or max_depth < 2 or max_depth > 6:
        raise ValueError("max_depth must be an integer between 2 and 6.")

    learning_rate = float(cfg.get("learning_rate")) if "learning_rate" in cfg else 0.1  # Default learning rate
    if "learning_rate" in cfg and (not isinstance(learning_rate, float) or learning_rate < 1e-5 or learning_rate > 1e-2):
        raise ValueError("learning_rate must be a float between 1e-5 and 1e-2")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Initialize and train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=num_estimators, max_depth=max_depth, random_state=seed, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Return negative accuracy for SMAC optimization
    return -accuracy
