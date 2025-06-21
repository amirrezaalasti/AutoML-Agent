import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a machine learning model (KNN or SVM) based on the provided configuration
    and dataset, and returns the negative accuracy.

    Args:
        cfg (Configuration): A ConfigSpace Configuration object containing
            hyperparameter settings for the model.
        dataset (Any): A dictionary containing the dataset with keys 'X'
            (feature matrix) and 'y' (label vector).
        seed (int): A random seed for reproducibility.

    Returns:
        float: The negative mean cross-validation accuracy.
    """

    X = dataset["X"]
    y = dataset["y"]

    # Correct way to detect numerical features:
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessing steps
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numerical_features), ("cat", categorical_transformer, categorical_features)]
    )

    # Determine model type and set up pipeline
    if "knn__n_neighbors" in cfg:
        model = KNeighborsClassifier(n_neighbors=cfg.get("knn__n_neighbors"), weights=cfg.get("knn__weights"))
    elif "svm__C" in cfg:
        model = SVC(
            C=cfg.get("svm__C"),
            kernel=cfg.get("svm__kernel"),
            gamma=cfg.get("svm__gamma"),
            degree=cfg.get("svm__degree"),
            random_state=seed,
            probability=True,  # Required for cross_val_predict with decision_function
        )

    else:
        raise ValueError("Invalid configuration: Model type not specified.")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    mean_accuracy = np.mean(scores)

    # SMAC minimizes, so return negative accuracy
    return -mean_accuracy
