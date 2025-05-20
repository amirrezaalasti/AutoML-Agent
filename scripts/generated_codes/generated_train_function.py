from typing import Any
from ConfigSpace import Configuration
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss  # Use log loss for multi-class classification


def train(cfg: Configuration, dataset: Any) -> float:
    """
    Trains a machine learning model based on the given configuration and dataset.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): A dictionary containing the training data with keys 'X' (features) and 'y' (labels).

    Returns:
        float: The average training loss over 10 epochs.
    """

    X = dataset["X"]
    y = dataset["y"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Using a validation set is important

    classifier_name = cfg.get("classifier")

    if classifier_name == "knn":
        model = KNeighborsClassifier(n_neighbors=cfg.get("n_neighbors"), weights=cfg.get("weights"), p=cfg.get("p"))
    elif classifier_name == "svm":
        model = SVC(
            C=cfg.get("C"), kernel=cfg.get("kernel"), degree=cfg.get("degree"), gamma=cfg.get("gamma"), probability=True  # Required for log_loss
        )
    elif classifier_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators"),
            max_depth=cfg.get("max_depth"),
            min_samples_split=cfg.get("min_samples_split"),
            min_samples_leaf=cfg.get("min_samples_leaf"),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    losses = []
    for _ in range(10):
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)
        loss = log_loss(y_val, y_pred_proba)
        losses.append(loss)

    return np.mean(losses)
