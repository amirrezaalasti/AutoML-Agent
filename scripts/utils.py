import pandas as pd
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario

from typing import Any


def describe_dataset(dataset: dict, dataset_type: str = "tabular") -> str:
    """
    Generate a detailed description of the dataset based on its type.

    Args:
        dataset (dict): A dictionary with 'X' (features) and 'y' (labels).
        dataset_type (str): One of ['tabular', 'time_series', 'image', 'categorical', 'text', etc.]

    Returns:
        str: A textual description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    description = ""

    if dataset_type == "tabular":
        description += "This is a tabular dataset.\n"
        description += f"It has {X.shape[0]} samples and {X.shape[1]} features.\n"
        description += "Feature columns and types:\n"
        if hasattr(X, "dtypes"):
            description += "\n".join(
                [f"- {col}: {dtype}" for col, dtype in X.dtypes.items()]
            )
        if hasattr(X, "describe"):
            description += "\n\nFeature statistical summary:\n"
            description += str(X.describe(include="all"))
        if hasattr(y, "value_counts"):
            description += "\n\nLabel distribution:\n"
            description += str(y.value_counts())

    elif dataset_type == "time_series":
        description += "This is a time series dataset.\n"
        description += f"Number of samples: {X.shape[0]}\n"
        if hasattr(X, "index"):
            description += f"Time index type: {type(X.index)}\n"
            description += f"Time range: {X.index.min()} to {X.index.max()}\n"
        description += "Features:\n"
        description += "\n".join([f"- {col}" for col in X.columns])

    elif dataset_type == "image":
        description += "This is an image dataset.\n"
        if hasattr(X, "__getitem__"):
            sample = X[0]
            if isinstance(sample, tuple) and len(sample) > 0:
                img = sample[0]
                description += f"Sample image shape: {img.shape}\n"
        description += f"Number of images: {len(X)}\n"
        description += f"Labels available: {len(y)}\n"

        description += f"Raw feature shape: {X.shape}\n"
        if isinstance(X, np.ndarray) and X.ndim == 2:
            description += "Note: images are likely flattened (e.g., 28x28 → 784)."
        elif isinstance(X, np.ndarray) and X.ndim == 4:
            description += "Note: images are likely in (N, C, H, W) format."

    elif dataset_type == "categorical":
        description += "This is a categorical dataset.\n"
        description += f"It has {X.shape[0]} samples.\n"
        description += "Categorical feature summary:\n"
        for col in X.select_dtypes(include=["object", "category"]).columns:
            description += f"- {col}: {X[col].nunique()} unique values\n"
        if hasattr(y, "nunique"):
            description += f"\nTarget variable has {y.nunique()} unique classes."

    elif dataset_type == "text":
        description += "This is a text dataset.\n"
        description += f"Number of text entries: {X.shape[0]}\n"
        if hasattr(X, "apply"):
            description += f"Average text length: {X.apply(lambda x: len(str(x))).mean():.2f} characters\n"
        if hasattr(y, "nunique"):
            description += f"\nTarget variable has {y.nunique()} unique classes."

    else:
        description += (
            "Dataset type not recognized. Please provide a valid dataset type."
        )

    return description


def log_message(message: str, log_file: str = "./logs/logs.txt"):
    """
    Log a message to a specified log file.

    Args:
        message (str): The message to log.
        log_file (str): The path to the log file. Default is "./logs/logs.txt".
    """
    with open(log_file, "a") as log:
        log.write(message + "\n")


def save_code(code: str, filename: str):
    """
    Save the generated code to a file.

    Args:
        code (str): The code to save.
        filename (str): The name of the file to save the code in.
    """
    with open(filename, "w") as file:
        file.write(code)


def run_smac(
    scenario: Scenario,
    train: Any,
    metric: str = "accuracy",
    budget: int = 100,
    n_trials: int = 10,
) -> Any:
    """
    Run the SMAC hyperparameter optimization.
    Args:
        scenario (Scenario): The SMAC scenario object.
        train (Any): The training data.
        metric (str): The evaluation metric. Default is "accuracy".
        budget (int): The budget for the optimization. Default is 100.
        n_trials (int): The number of trials to run. Default is 10.
    Returns:
        Any: The best hyperparameters found by SMAC.
    """
    # Placeholder for actual SMAC implementation
    smac = HyperparameterOptimizationFacade(scenario, train)
    smac.optimize(
        metric=metric,
        budget=budget,
        n_trials=n_trials,
    )
    return smac.get_best_configuration()

def format_dataset(dataset: Any) -> dict:
    """
    Format the dataset into a dictionary with 'X' and 'y' keys,
    ensuring both are pandas DataFrame/Series.

    Args:
        dataset (Any): The dataset to format.

    Returns:
        dict: A dictionary with 'X' and 'y' keys as pandas objects.
    """
    if isinstance(dataset, pd.DataFrame):
        if "target" not in dataset.columns:
            raise ValueError("DataFrame must include a 'target' column.")
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

    elif isinstance(dataset, dict):
        X = dataset["X"]
        y = dataset["y"]

        # Convert to pandas if they're NumPy arrays
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

    elif isinstance(dataset, np.ndarray):
        # Single feature case
        if dataset.ndim == 1:
            X = pd.DataFrame(dataset.reshape(-1, 1))
        else:
            X = pd.DataFrame(dataset)
        y = None  # No labels provided

    else:
        raise ValueError("Unsupported dataset format")

    return {"X": X, "y": y}