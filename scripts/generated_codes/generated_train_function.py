from typing import Any
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ConfigSpace import Configuration


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a Support Vector Machine (SVM) model on the given dataset
    using the provided configuration and returns the average training loss
    over 10 epochs.

    Args:
        cfg (Configuration): Configuration object containing hyperparameters for the SVM.
        dataset (Any): Dictionary containing the training data, with keys 'X' for features
            and 'y' for labels.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Average training loss over 10 epochs.
    """
    X = dataset["X"]
    y = dataset["y"]

    # Infer input and output dimensions dynamically
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Initialize the SVM model with hyperparameters from the configuration
    kernel = cfg.get("kernel")
    C = cfg.get("C")
    gamma = cfg.get("gamma")
    degree = cfg.get("degree")
    coef0 = cfg.get("coef0")

    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, random_state=seed)

    # Train the model for 10 epochs (in this case, we're just fitting the model 10 times and averaging the loss,
    #  as there's no actual epoch-based training for sklearn's SVC directly).
    total_loss = 0.0
    for _ in range(10):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        loss = mean_squared_error(y_train, y_pred)  # Use MSE for consistent loss metric
        total_loss += loss

    avg_loss = total_loss / 10.0

    return avg_loss
