import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a machine learning model based on the provided configuration and dataset.

    Args:
        cfg (Configuration): A Configuration object containing the hyperparameters.
        dataset (Any): A dictionary containing the training data ('X' and 'y').
        seed (int): Random seed for reproducibility.

    Returns:
        float: Negative validation accuracy.
    """
    np.random.seed(seed)

    X = dataset["X"].copy()
    y = dataset["y"].copy()

    # Preprocessing: Handle non-numeric values and scale
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]):
            # Attempt conversion to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors="coerce")
            # Fill NaN values resulting from failed conversion with the median
            X[col] = X[col].fillna(X[col].median())
            # If there are still NaN values after filling with median, use Label Encoding
            if X[col].isnull().any():
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())  # Fill NaN with median
        else:
            # Handle cases where the column is neither object nor numeric
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model_type = cfg.get("model_type")

    if model_type == "logistic_regression":
        C = cfg.get("logistic_regression:C")
        penalty = cfg.get("logistic_regression:penalty")
        model = LogisticRegression(C=C, penalty=penalty, solver="liblinear", random_state=seed)
    elif model_type == "random_forest":
        n_estimators = cfg.get("random_forest:n_estimators")
        max_depth = cfg.get("random_forest:max_depth")
        min_samples_split = cfg.get("random_forest:min_samples_split")
        min_samples_leaf = cfg.get("random_forest:min_samples_leaf")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "svm":
        C = cfg.get("svm:C")
        kernel = cfg.get("svm:kernel")
        gamma = cfg.get("svm:gamma")
        degree = cfg.get("svm:degree")
        model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return -accuracy
