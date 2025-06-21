import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.preprocessing
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a RandomForestClassifier on the given dataset using the provided configuration.

    Args:
        cfg (Configuration): Configuration object containing hyperparameters.
        dataset (Any): Dictionary containing the dataset with 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: Negative accuracy of the model on the test set.
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Extract data
        X = dataset["X"]
        y = dataset["y"]

        # Data Preprocessing
        scaling_method = cfg.get("scaling")
        if scaling_method == "standard":
            scaler = sklearn.preprocessing.StandardScaler()
            X = scaler.fit_transform(X)
        elif scaling_method == "minmax":
            scaler = sklearn.preprocessing.MinMaxScaler()
            X = scaler.fit_transform(X)
        # else scaling_method == "none" - Do nothing

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

        # Initialize RandomForestClassifier with hyperparameters from the configuration
        rf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=cfg.get("rf_n_estimators"),
            max_features=cfg.get("rf_max_features"),
            min_samples_split=cfg.get("rf_min_samples_split"),
            min_samples_leaf=cfg.get("rf_min_samples_leaf"),
            random_state=seed,
            n_jobs=1,  # Ensure single core usage,
            oob_score=True,  # Enable OOB score calculation
        )

        # Train the model
        rf.fit(X_train, y_train)

        # Calculate the accuracy on the test set
        accuracy = rf.score(X_test, y_test)

        # Return negative accuracy, as SMAC minimizes the target function
        return -accuracy
    except Exception as e:
        print(f"An error occurred: {e}")
        return -0.0  # Return a very low score in case of error.
