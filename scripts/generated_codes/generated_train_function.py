import sklearn
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a machine learning model based on the provided configuration and dataset.

    Args:
        cfg (Configuration): The configuration object containing the hyperparameters.
        dataset (Any): The dataset containing features (X) and labels (y).
        seed (int): The random seed for reproducibility.

    Returns:
        float: The negative accuracy score on the test set.
    """
    np.random.seed(seed)

    X = dataset["X"]
    y = dataset["y"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # 1. Data Preprocessing
    scaler_name = cfg.get("scaler")
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 2. Feature Engineering
    use_feature_engineering = cfg.get("use_feature_engineering")
    if use_feature_engineering:
        poly_degree = cfg.get("poly_degree")
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

    # 3. Model Selection
    model_type = cfg.get("model_type")

    # 4. SVC
    if model_type == "SVC":
        svc_kernel = cfg.get("svc_kernel")
        svc_C = cfg.get("svc_C")
        svc_gamma = cfg.get("svc_gamma")
        svc_degree = cfg.get("svc_degree")

        model = SVC(kernel=svc_kernel, C=svc_C, gamma=svc_gamma, degree=svc_degree, random_state=seed)

    # 5. RandomForest
    elif model_type == "RandomForest":
        rf_n_estimators = cfg.get("rf_n_estimators")
        rf_max_depth = cfg.get("rf_max_depth")
        rf_min_samples_leaf = cfg.get("rf_min_samples_leaf")

        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_leaf=rf_min_samples_leaf, random_state=seed)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Return the negative accuracy score for minimization
    return -accuracy
