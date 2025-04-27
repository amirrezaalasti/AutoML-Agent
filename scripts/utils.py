import pandas as pd
import numpy as np

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
        description += (
            f"It has {X.shape[0]} samples (rows) and {X.shape[1]} features (columns).\n"
        )
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
