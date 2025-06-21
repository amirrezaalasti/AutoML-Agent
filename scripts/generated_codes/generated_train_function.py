import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a classification model (k-NN or SVM) on the provided dataset using the given configuration.

    Args:
        cfg (Configuration): The configuration object containing hyperparameter settings.
        dataset (Any): A dictionary containing 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: The negative cross-validation accuracy.
    """

    # Input validation
    assert isinstance(dataset, dict), "Dataset must be a dictionary."
    assert "X" in dataset and "y" in dataset, "Dataset must contain 'X' and 'y'."
    X = dataset["X"]
    y = dataset["y"]
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y must be a pandas Series."
    assert len(X) == len(y), "X and y must have the same length."
    assert X.shape[0] > 0 and X.shape[1] > 0, "X must have at least one sample and one feature."
    assert y.nunique() > 1, "y must have more than one class."

    # Hyperparameter extraction and validation
    n_neighbors = cfg.get("n_neighbors")
    C = cfg.get("C")
    kernel = cfg.get("kernel")
    degree = cfg.get("degree")
    gamma = cfg.get("gamma")

    assert isinstance(n_neighbors, int) and 1 <= n_neighbors <= 10, "n_neighbors must be an integer between 1 and 10."
    assert isinstance(C, float) and C > 0, "C must be a positive float."
    assert kernel in ["linear", "rbf", "poly", "sigmoid"], "kernel must be one of 'linear', 'rbf', 'poly', 'sigmoid'."
    assert isinstance(degree, int) and 2 <= degree <= 5, "degree must be an integer between 2 and 5."
    assert isinstance(gamma, float) and gamma > 0, "gamma must be a positive float."

    # Model selection and initialization based on hyperparameters
    if kernel is None:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=seed, probability=False)  # probability=False for speed

    # Cross-validation setup
    n_splits = 5  # Fixed number of splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Cross-validation loop
    accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Model training
        model.fit(X_train, y_train)

        # Prediction and evaluation
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    # Calculate mean accuracy
    mean_accuracy = np.mean(accuracies)

    # Return negative mean accuracy (SMAC minimizes)
    return -mean_accuracy
