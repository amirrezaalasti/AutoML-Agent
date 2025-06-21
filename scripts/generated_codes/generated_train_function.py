import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a Gradient Boosting Classifier on the adult dataset using the provided
    hyperparameters and returns the validation error.
    """
    try:
        # 1. Data Loading & Preprocessing
        X = dataset["X"]
        y = dataset["y"]

        # Identify categorical and numerical features (explicitly defined)
        categorical_features = [
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capitalgain",
            "capitalloss",
            "hoursperweek",
            "native-country",
        ]
        numerical_features = ["fnlwgt", "education-num"]

        # Create preprocessor including OneHotEncoding for categorical features
        # and StandardScaler for numerical features.
        numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])  # Handle missing values

        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]  # Handle missing values
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numerical_transformer, numerical_features), ("cat", categorical_transformer, categorical_features)],
            remainder="passthrough",  # Drop remaining columns
        )

        # 2. Data Splitting
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        # 3. Model Instantiation
        model = GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators"),
            learning_rate=cfg.get("learning_rate"),
            max_depth=cfg.get("max_depth"),
            min_samples_split=cfg.get("min_samples_split"),
            min_samples_leaf=cfg.get("min_samples_leaf"),
            subsample=cfg.get("subsample"),
            random_state=seed,
            n_iter_no_change=5,
            validation_fraction=0.1,
            tol=0.01,
            warm_start=False,
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        # 4. Model Training
        pipeline.fit(X_train, y_train)

        # 5. Performance Evaluation
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # 6. Return Value (Validation Error)
        return -accuracy  # SMAC minimizes, return negative accuracy
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1.0  # Return a large error value in case of failure
