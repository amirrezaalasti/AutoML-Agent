import sklearn.svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ConfigSpace import Configuration
from typing import Any
import numpy as np


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains an SVM model on the given dataset using the provided configuration.

    Args:
        cfg (Configuration): The configuration for the SVM model.
        dataset (Any): A dictionary containing the feature matrix 'X' and label vector 'y'.
        seed (int): The random seed for reproducibility.

    Returns:
        float: The average cross-validation error (1 - accuracy).
    """
    try:
        X = dataset["X"]
        y = dataset["y"]

        # Handle missing values with imputation
        imputer = SimpleImputer(strategy="mean")  # or 'median', 'most_frequent', 'constant'
        X = imputer.fit_transform(X)

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Extract hyperparameters from configuration
        kernel = cfg.get("svm_kernel")
        C = cfg.get("svm_C")
        gamma = cfg.get("svm_gamma") if "svm_gamma" in cfg else "scale"  # Handle conditional gamma
        degree = cfg.get("svm_degree") if "svm_degree" in cfg else 3  # Handle conditional degree

        # Create SVM model with configuration
        model = sklearn.svm.SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            random_state=seed,
            probability=False,  # probability=True makes training slower
        )

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=1)  # n_jobs=1 for reproducibility in SMAC
        avg_accuracy = np.mean(cv_scores)

        # Return 1 - accuracy as SMAC minimizes the objective
        return -(avg_accuracy)

    except Exception as e:
        print(f"Error during training: {e}")
        # Return a large error value in case of failure
        return 1.0
