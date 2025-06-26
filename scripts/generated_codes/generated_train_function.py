import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from ConfigSpace import Configuration
from typing import Any
import xgboost as xgb


def train(cfg: Configuration, dataset: Any, seed: int, model_dir: str = None) -> float:
    """
    Trains a machine learning model based on the provided configuration and dataset.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): The dataset containing features (X) and target variable (y).
        seed (int): The random seed for reproducibility.
        model_dir (str, optional): The directory to save the trained model. Defaults to None.

    Returns:
        float: The Root Mean Squared Error (RMSE) on the validation set.
    """

    # Validate inputs
    assert isinstance(cfg, Configuration), "cfg must be a Configuration object"
    assert isinstance(dataset, dict), "dataset must be a dictionary"
    assert "X" in dataset and "y" in dataset, "dataset must contain 'X' and 'y' keys"
    assert isinstance(seed, int), "seed must be an integer"
    if model_dir is not None:
        assert isinstance(model_dir, str), "model_dir must be a string"

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Extract data
    X = dataset["X"]
    y = dataset["y"]

    # Validate data types and shapes
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    assert isinstance(X, np.ndarray), "X must be a NumPy array"
    assert isinstance(y, np.ndarray), "y must be a NumPy array"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
    assert len(y.shape) == 1, "y must be a 1D array"

    # Split the dataset into training and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Determine the model type from the configuration
    model_type = cfg.get("model_type")
    assert model_type in ["random_forest", "gradient_boosting", "xgboost"], "Invalid model_type"

    # Initialize the model based on the model type
    if model_type == "random_forest":
        model = RandomForestRegressor(random_state=seed)
        model.n_estimators = int(cfg.get("n_estimators", 100))
        model.max_depth = int(cfg.get("max_depth", 10))
        model.min_samples_split = int(cfg.get("min_samples_split", 2))
        model.min_samples_leaf = int(cfg.get("min_samples_leaf", 1))

    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=seed)
        model.n_estimators = int(cfg.get("n_estimators", 100))
        model.max_depth = int(cfg.get("max_depth", 3))
        model.learning_rate = float(cfg.get("learning_rate", 0.1))

    elif model_type == "xgboost":
        model = xgb.XGBRegressor(random_state=seed)
        model.n_estimators = int(cfg.get("n_estimators", 100))
        model.max_depth = int(cfg.get("max_depth", 3))
        model.reg_alpha = float(cfg.get("reg_alpha", 0.0))
        model.reg_lambda = float(cfg.get("reg_lambda", 1.0))

    else:
        raise ValueError("Invalid model type")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Save the model if model_dir is provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{seed}.joblib")
        dump(model, model_path)

    return rmse
