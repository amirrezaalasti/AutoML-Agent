from typing import Any
from ConfigSpace import Configuration
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a RandomForestClassifier on the given dataset using the provided configuration and returns the average cross-validation loss.

    Args:
        cfg (Configuration): A configuration object containing hyperparameters for the RandomForestClassifier.
        dataset (Any): A dictionary containing the dataset with keys 'X' (feature matrix) and 'y' (target vector).
        seed (int): A random seed for reproducibility.

    Returns:
        float: The average cross-validation loss across 10 epochs (splits).
    """

    X = dataset["X"]
    y = dataset["y"]

    # Convert to pandas DataFrame
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Preprocessing steps
    numeric_features = X.select_dtypes(include=np.number).columns
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]  # Handle missing values  # Scale numerical features
    )

    # Apply preprocessing to numerical features
    X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])

    # Model definition
    model = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators"),
        max_depth=cfg.get("max_depth"),
        min_samples_split=cfg.get("min_samples_split"),
        min_samples_leaf=cfg.get("min_samples_leaf"),
        random_state=seed,
        n_jobs=-1,  # Use all available cores
    )

    # Cross-validation
    n_splits = 10  # Number of cross-validation splits (epochs)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)
        loss = log_loss(y_val, y_pred)
        losses.append(loss)

    return np.mean(losses)
