from typing import Any
from ConfigSpace import Configuration
import sklearn.svm
import sklearn.tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a model based on the given configuration and dataset.

    Args:
        cfg (Configuration): Configuration object containing hyperparameters.
        dataset (Any): Dictionary containing 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: Cross-validation score (negative mean cross-validation score, as SMAC minimizes).
    """
    X = dataset["X"]
    y = dataset["y"]

    # Infer input and output dimensions dynamically
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # Data preprocessing: StandardScaler
    scaler = StandardScaler()

    # Model selection and configuration
    classifier_type = cfg.get("classifier_type")

    if classifier_type == "SVM":
        # SVM Model
        svm_kernel = cfg.get("svm_kernel")
        svm_C = cfg.get("svm_C")
        svm_gamma = cfg.get("svm_gamma")
        svm_degree = cfg.get("svm_degree")

        model = sklearn.svm.SVC(
            kernel=svm_kernel,
            C=svm_C,
            gamma=svm_gamma,
            degree=svm_degree,
            random_state=seed,  # Add random_state for reproducibility
            probability=True,  # Required for cross_val_predict with decision_function
        )
    elif classifier_type == "DecisionTree":
        # Decision Tree Model
        dt_criterion = cfg.get("dt_criterion")
        dt_max_depth = cfg.get("dt_max_depth")
        dt_min_samples_split = cfg.get("dt_min_samples_split")
        dt_min_samples_leaf = cfg.get("dt_min_samples_leaf")

        model = sklearn.tree.DecisionTreeClassifier(
            criterion=dt_criterion,
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            min_samples_leaf=dt_min_samples_leaf,
            random_state=seed,  # Add random_state for reproducibility
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Create a pipeline with scaling and the model
    pipeline = Pipeline([("scaler", scaler), ("classifier", model)])

    # Cross-validation
    try:
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_log_loss")  # Use neg_log_loss, as SMAC minimizes. Fallback for probabilities.
        loss = np.mean(scores)  # Lower loss is better
    except ValueError as e:
        # Handle cases where probabilities are required but not available (e.g., decision_function not applicable)
        # or when cross-validation fails because of class imbalance in a split
        print(f"Cross-validation failed: {e}. Returning a large loss value.")
        loss = -10.0  # Return a large loss value to penalize the configuration

    return loss
