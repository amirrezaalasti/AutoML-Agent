import pandas as pd
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
import warnings
from typing import Any
import re


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
            description += "\n".join([f"- {col}: {dtype}" for col, dtype in X.dtypes.items()])
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

        description += "\n\n### Time Series Handling Requirements\n"
        description += "- Assume `dataset['X']` is a 3D array or tensor with shape `(num_samples, sequence_length, num_features)`.\n"
        description += "- If `dataset['X']` is 2D, raise a `ValueError` if the model is RNN-based (`LSTM`, `GRU`, `RNN`).\n"
        description += "- Do **not** flatten the input when using RNN-based models.\n"
        description += "- Use `batch_first=True` in all recurrent models to maintain `(batch, seq_len, features)` format.\n"
        description += "- Dynamically infer sequence length as `X.shape[1]` and feature dimension as `X.shape[2]`.\n"
        description += "- If `X.ndim != 3` and a sequential model is selected, raise a clear error with shape info.\n"
        description += "- Example input validation check:\n"
        description += "  ```python\n"
        description += "  if model_type in ['LSTM', 'GRU', 'RNN'] and X_tensor.ndim != 3:\n"
        description += '      raise ValueError(f"Expected 3D input (batch, seq_len, features) for {model_type}, got {X_tensor.shape}")\n'
        description += "  ```\n"
        description += "- Time index or datetime values can be logged but should not be used in the model unless specified.\n"

    elif dataset_type == "image":
        description += "This is an image dataset.\n"
        if isinstance(X, np.ndarray):
            if X.ndim == 4:
                n, c, h, w = X.shape if X.shape[1] <= 3 else (X.shape[0], X.shape[3], X.shape[1], X.shape[2])
                description += f"Image format: (N={n}, C={c}, H={h}, W={w})\n"
                description += "Note: Images are in (batch, channels, height, width) format.\n"
            elif X.ndim == 3:
                n, h, w = X.shape
                description += f"Image format: (N={n}, H={h}, W={w})\n"
                description += "Note: Images are single-channel in (batch, height, width) format.\n"
            elif X.ndim == 2:
                n, pixels = X.shape
                height = int(np.sqrt(pixels))
                if height * height == pixels:
                    description += f"Image format: Flattened {height}x{height} images\n"
                    description += f"Shape: (N={n}, pixels={pixels})\n"
                    description += "Note: Images are flattened. Must be reshaped for processing.\n"
                else:
                    description += f"Shape: (N={n}, features={pixels})\n"
                    description += "Warning: Feature count is not a perfect square.\n"

        if hasattr(y, "nunique"):
            description += f"\nNumber of classes: {y.nunique()}\n"
            description += "Class distribution:\n"
            description += str(y.value_counts())

        # Add specific handling requirements for image data
        description += "\n\nImage Data Handling Requirements:\n"
        description += "1. Input Format Requirements:\n"
        description += "   - For CNN models: Input must be in (batch, channels, height, width) format\n"
        description += "   - For dense/linear layers: Input should be flattened\n\n"
        description += "2. Data Processing Steps:\n"
        description += "   a) For flattened input (2D):\n"
        description += "      - Calculate dimensions: height = width = int(sqrt(n_features))\n"
        description += "      - Verify square dimensions: height * height == n_features\n"
        description += "      - Reshape to (N, 1, H, W) for CNNs\n"
        description += "   b) For 3D input (N, H, W):\n"
        description += "      - Add channel dimension: reshape to (N, 1, H, W)\n"
        description += "   c) For 4D input:\n"
        description += "      - Verify channel order matches framework requirements\n\n"
        description += "3. Framework-Specific Format:\n"
        description += "   - PyTorch: (N, C, H, W)\n"
        description += "   - TensorFlow: (N, H, W, C)\n"
        description += "   - Convert between formats if necessary\n\n"
        description += "4. Normalization:\n"
        description += "   - Scale pixel values to [0, 1] by dividing by 255.0\n"
        description += "   - Or standardize to mean=0, std=1\n"

    elif dataset_type == "categorical":
        description += "This is a categorical dataset.\n"
        description += f"It has {X.shape[0]} samples.\n"
        description += "Categorical feature summary:\n"
        for col in X.select_dtypes(include=["object", "category"]).columns:
            description += f"- {col}: {X[col].nunique()} unique values\n"
        if hasattr(y, "nunique"):
            description += f"\nTarget variable has {y.nunique()} unique classes."
        description += "\n\nCategorical Data Handling Requirements:\n"
        description += "1. Preprocessing Steps:\n"
        description += "   - Encode categorical variables (one-hot or label encoding)\n"
        description += "   - Handle missing values\n"
        description += "   - Check for cardinality of categorical variables\n\n"
        description += "2. Model Considerations:\n"
        description += "   - Use appropriate encoding for model type\n"
        description += "   - Handle high cardinality features appropriately\n"

    elif dataset_type == "text":
        description += "This is a text dataset.\n"
        description += f"Number of text entries: {X.shape[0]}\n"
        if hasattr(X, "apply"):
            description += f"Average text length: {X.apply(lambda x: len(str(x))).mean():.2f} characters\n"
        if hasattr(y, "nunique"):
            description += f"\nTarget variable has {y.nunique()} unique classes."

        # Add specific handling requirements for text data
        description += "\n\nText Data Handling Requirements:\n"
        description += "1. Preprocessing Steps:\n"
        description += "   - Tokenization\n"
        description += "   - Convert to lowercase\n"
        description += "   - Remove special characters if needed\n"
        description += "   - Handle missing values\n\n"
        description += "2. Feature Extraction:\n"
        description += "   - Use TF-IDF or word embeddings\n"
        description += "   - Consider max sequence length\n"
        description += "   - Handle vocabulary size\n\n"
        description += "3. Model-Specific Requirements:\n"
        description += "   - For neural networks: Use appropriate padding/truncation\n"
        description += "   - For traditional ML: Convert to fixed-size features\n"

    else:
        description += "Dataset type not recognized. Please provide a valid dataset type."

    return description


def generate_general_dataset_task_types(dataset_type: str) -> list[str]:
    """
    Generate a list of task types for a given dataset type.
    """
    if dataset_type == "tabular":
        return ["classification", "regression"]
    elif dataset_type == "time_series":
        return ["clustering"]
    elif dataset_type == "image":
        return ["classification", "clustering"]
    elif dataset_type == "categorical":
        return ["classification", "clustering"]
    elif dataset_type == "text":
        return ["clustering"]


def log_message(message: str, log_file: str = "./logs/logs.txt"):
    """
    DEPRECATED: Use the Logger class instead.
    Log a message to a specified log file.

    Args:
        message (str): The message to log.
        log_file (str): The path to the log file. Default is "./logs/logs.txt".
    """
    warnings.warn(
        "This function is deprecated. Please use the Logger class from logger.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def extract_code_block(code: str) -> str:
    """Extract Python code between markdown code fences"""
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1) if match else code
