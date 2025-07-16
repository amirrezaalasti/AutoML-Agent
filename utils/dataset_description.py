"""
Dataset description utilities for AutoML Agent.

This module provides functions for generating detailed descriptions of datasets
based on their type and the machine learning task.
"""

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .constants import DATASET_TASK_MAPPING
from .data_processing import format_dataset, get_dataset_info, validate_dataset_type
from .types import ValidationError


def describe_tabular_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a description for a tabular dataset, tailored to the ML task.

    Args:
        dataset: A preprocessed dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        A string description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset, task_type)

    description = [
        "This is a tabular dataset.",
        "NOTE: Categorical features have been pre-processed using one-hot encoding.",
        f"It has {info.n_samples} samples and {info.n_features} features after encoding.",
        f"The target variable is {info.target_type} with {info.target_description}.",
    ]

    # Task-specific description of the target variable
    if task_type == "classification":
        description.append(
            "The target variable has been label-encoded to be numeric for classification."
        )
        class_counts = y.value_counts()
        if len(class_counts) <= 10:
            description.append(f"\nClass distribution: {class_counts.to_dict()}")
        else:
            description.append(
                f"\nNumber of classes: {len(class_counts)} (too many to display)"
            )
        description.append(
            f"""* **For classification tasks**
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
            metrics = {{
                'loss': loss,
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_test, predictions)
            }}
            """
        )
    elif task_type == "regression":
        description.append(
            "The target variable is continuous for this regression task."
        )
        # for regression sometimes the target value is not normalized properly, so we need to normalize it
        if y.min() < 0 or y.max() > 1:
            description.append(
                "The target variable is not normalized properly. Please normalize it."
            )
        description.append("\nTarget variable statistical summary:")
        description.append(str(y.describe()))
        description.append(
            f"""* **For regression tasks**
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {{
                'loss': loss,
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }}
            """
        )
    else:  # Clustering or other tasks
        description.append("The target variable is present for evaluation purposes.")

    if hasattr(X, "dtypes"):
        description.append(f"\nFeature data types summary:")
        dtype_counts = X.dtypes.value_counts()
        description.append(f"- {dtype_counts.to_dict()}")
        if len(X.columns) <= 20:
            description.append(f"- Column names: {list(X.columns)}")
        else:
            description.append(
                f"- Total columns: {len(X.columns)} (too many to display)"
            )

    if hasattr(X, "describe"):
        description.append("\nFeature statistical summary (first 5 numeric columns):")
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) > 0:
            description.append(str(X[numeric_cols].describe()))

    return "\n".join(description)


def describe_time_series_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a description for a time series dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset, task_type)

    description = [
        "This is a time series dataset.",
        f"Number of samples: {info.n_samples}",
    ]

    if hasattr(X, "index"):
        description.extend(
            [
                f"Time index type: {type(X.index)}",
                f"Time range: {X.index.min()} to {X.index.max()}",
            ]
        )

    description.append("Features:")
    description.extend([f"- {col}" for col in X.columns])

    # --- Target Variable Analysis (Task-Specific) ---
    description.append("\n### Target Variable (y) Analysis")
    if task_type == "classification":
        description.append(f"* **Task**: Time series classification.")
        description.append(f"* **Number of Classes**: {y.nunique()}")
        description.append(f"* **Class Distribution**: {y.value_counts().to_dict()}")
    elif task_type == "regression":
        description.append(f"* **Task**: Time series forecasting/regression.")
        description.append(f"* **Target Summary**: \n{y.describe().to_string()}")

    # Add handling requirements
    requirements = [
        "\n### Time Series Handling Requirements",
        "- Assume `dataset['X']` is a 3D array or tensor with shape `(num_samples, sequence_length, num_features)`.",
        "- If `dataset['X']` is 2D, raise a `ValueError` if the model is RNN-based (`LSTM`, `GRU`, `RNN`).",
        "- Do **not** flatten the input when using RNN-based models.",
        "- Use `batch_first=True` in all recurrent models to maintain `(batch, seq_len, features)` format.",
        "- Dynamically infer sequence length as `X.shape[1]` and feature dimension as `X.shape[2]`.",
        "- If `X.ndim != 3` and a sequential model is selected, raise a clear error with shape info.",
        "- Example input validation check:",
        "  ```python",
        "  if model_type in ['LSTM', 'GRU', 'RNN'] and X_tensor.ndim != 3:",
        '      raise ValueError(f"Expected 3D input (batch, seq_len, features) for {model_type}, got {X_tensor.shape}")',
        "  ```",
        "- Time index or datetime values can be logged but should not be used in the model unless specified.",
    ]

    description.extend(requirements)
    return "\n".join(description)


import os
import math
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any


def describe_image_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate a description for an image dataset, tailored to the ML task.

    Args:
        dataset: A dictionary containing 'X' (features) and 'y' (labels).
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        A string description of the dataset with preprocessing guidance.
    """
    # Validate inputs
    if "X" not in dataset or "y" not in dataset:
        return "Error: dataset must contain 'X' and 'y'."

    X, y = dataset["X"], dataset["y"]

    # Determine input type and extract numpy array
    if isinstance(X, pd.DataFrame):
        input_type = f"Pandas DataFrame with shape {X.shape}"
        X_np = X.values.astype(np.float32)
    elif isinstance(X, np.ndarray):
        input_type = f"NumPy Array with shape {X.shape}"
        X_np = X.astype(np.float32)
    else:
        return f"Unsupported input type for image dataset: {type(X).__name__}."

    # Basic sample and feature counts
    n_samples = X_np.shape[0]
    n_features = np.prod(X_np.shape[1:]) if X_np.ndim >= 2 else 0

    # Analyze image format and detect channels dynamically
    img_info = ""
    channels = 1  # Default to grayscale
    img_dim = 28  # Default dimension

    if X_np.ndim == 2:
        # Flattened images - try to infer dimensions
        feature_count = X_np.shape[1]
        dim_color = int(math.sqrt(feature_count / 3))
        dim_gray = int(math.sqrt(feature_count))

        if dim_color**2 * 3 == feature_count:
            img_info = f"Likely {dim_color}x{dim_color} color images (3 channels)"
            channels = 3
            img_dim = dim_color
        elif dim_gray**2 == feature_count:
            img_info = f"Likely {dim_gray}x{dim_gray} grayscale images (1 channel)"
            channels = 1
            img_dim = dim_gray
        else:
            img_info = f"Flattened images with {feature_count} features per sample"
            # Try to infer from common image sizes
            if feature_count == 784:  # 28x28 MNIST/Fashion-MNIST
                channels, img_dim = 1, 28
            elif feature_count == 3072:  # 32x32x3 CIFAR-10
                channels, img_dim = 3, 32
            elif feature_count == 1024:  # 32x32 grayscale
                channels, img_dim = 1, 32
    elif X_np.ndim == 4:
        n, c, h, w = X_np.shape
        img_info = f"4D array format: {n} samples, {c} channels, {h}x{w} resolution"
        channels = c
        img_dim = h
    else:
        img_info = f"Unexpected dimensionality: {X_np.ndim}D array"

    # Target variable analysis
    if task_type.lower() == "classification":
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        num_classes = len(np.unique(y_np))
        task_info = f"Classification task with {num_classes} classes"
    elif task_type.lower() == "regression":
        task_info = "Regression task"
    else:
        task_info = f"{task_type} task"

    # Build description
    description = f"""### Image Dataset Analysis
    **Input Type**: {input_type}
    **Samples**: {n_samples}
    **Total Features**: {n_features}
    **Image Format**: {img_info}
    **Task**: {task_info}

    ### Preprocessing Guidelines

    1. **Data Reshaping**: If data is flattened, reshape to (N, C, H, W) format for CNN models
    2. **Normalization**: Scale pixel values to [0,1] range by dividing by 255.0
    3. **Data Splitting**: Split data into training (70%), validation (15%), and test (15%) sets
    4. **Validation**: Ensure input dimensions match model expectations
    
    ### Data Augmentation (Recommended for Better Performance)
    
    Data augmentation can significantly improve model performance and generalization:
    
    ```python
    from torchvision import transforms
    
    # For training data
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # For validation/test data
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

    ### Model Architecture Considerations

    - **CNN Models**: Require 4D input tensors (batch, channels, height, width)
    - **Adaptive Pooling**: Use `nn.AdaptiveAvgPool2d((1, 1))` for robust feature extraction
    - **Dynamic Padding**: Use `padding = kernel_size // 2` to maintain spatial dimensions
    - **Memory Management**: Monitor batch sizes to avoid GPU memory issues

    ### Critical Hyperparameter Insights

    Based on empirical analysis, consider these factors for optimal performance:

    1. **Learning Rate**: Default to 0.01 for better convergence (avoid 0.001 which can lead to poor performance)
    2. **Learning Rate Schedulers**: Use schedulers for better training convergence
    3. **Optimizer**: Adam optimizer typically provides better convergence than SGD
    4. **Tensor Operations**: Use `torch.from_numpy()` for memory efficiency
    5. **Budget Handling**: Use `min(budget, epochs)` to respect both time constraints and default training duration
    6. **Parameter Validation**: Use less restrictive bounds that preserve user intent
    
    **Learning Rate Scheduler Example**:
    ```python
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    
    # Option 1: Step decay
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Option 2: Reduce on plateau (recommended)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # In training loop:
    for epoch in range(epochs):
        # ... training code ...
        scheduler.step()  # For StepLR
        # scheduler.step(val_loss)  # For ReduceLROnPlateau
    ```

    ### Task-Specific Guidance
    """
    # Add task-specific guidance
    if task_type.lower() == "classification":
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        num_classes = len(np.unique(y_np))
        description += f"""
    **Classification Task**:
    - Use `nn.CrossEntropyLoss()` for multi-class classification
    - Evaluate with: accuracy, precision, recall, F1-score (weighted average)
    - Number of classes: {num_classes}
    - Consider class imbalance in evaluation metrics

    **Suggested Architecture Pattern**:
    ```python
    class DynamicCNN(nn.Module):
        def __init__(self, num_classes, num_conv_layers=3, initial_filters=32, 
                     kernel_size=3, dropout_rate=0.2, in_channels={channels}):
            super().__init__()
            layers = []
            current_channels = {channels}
            out_channels = initial_filters
            for _ in range(num_conv_layers):
                padding = kernel_size // 2
                layers.extend([
                    nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                ])
                current_channels = out_channels
                out_channels *= 2
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(current_channels, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)
    ```
    """
    elif task_type.lower() == "regression":
        description += f"""
    **Regression Task**:
    - Use `nn.MSELoss()` or `nn.L1Loss()` depending on requirements
    - Evaluate with: MSE, MAE, R² score
    - Consider target normalization if values are not in reasonable ranges
    - Monitor for overfitting with validation loss

    **Suggested Architecture Pattern**:
    ```python
    class CNNRegressor(nn.Module):
        def __init__(self, num_conv_layers=3, initial_filters=32, 
                     kernel_size=3, dropout_rate=0.2, in_channels={channels}):
            super().__init__()
            layers = []
            current_channels =  {channels}
            out_channels = initial_filters
            for _ in range(num_conv_layers):
                padding = kernel_size // 2
                layers.extend([
                    nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                ])
                current_channels = out_channels
                out_channels *= 2
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.features = nn.Sequential(*layers)
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(current_channels, 1)
            )
        def forward(self, x):
            x = self.features(x)
            return self.regressor(x)
    ```
    """
    description += f"""
    
    ### Transfer Learning (Alternative Approach)
    
    For better performance, especially with limited data, consider using pre-trained models:
    
    ```python
    import torchvision.models as models
    import torch.nn as nn
    
    # Load pre-trained ResNet
    model = models.resnet18(pretrained=True)
    
    # Modify input layer for different channel counts
    if {channels} != 3:
        model.conv1 = nn.Conv2d({channels}, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Replace final layer for your task
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, {num_classes if task_type.lower() == 'classification' else '1'})
    
    # Freeze early layers (optional)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    ```
    
    ### Common Pitfalls to Avoid
    1. **Learning Rate Too Low**: Using 0.001 can lead to poor convergence (~40% accuracy)
    2. **Missing Adaptive Pooling**: Can cause crashes with certain input sizes
    3. **Overly Restrictive Validation**: Overriding user parameters unnecessarily
    4. **Inefficient Tensor Creation**: Using `torch.tensor()` instead of `torch.from_numpy()`
    5. **Poor Budget Handling**: Direct assignment instead of `min(budget, epochs)`

    ### Performance Optimization Tips
    - Start with learning rate 0.01 for better convergence
    - Use Adam optimizer consistently
    - Implement proper error handling for tensor size mismatches
    - Monitor GPU memory usage with large batch sizes
    - Use adaptive pooling for input size flexibility
    - It is very important to use cuda here.
    - device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """

    return description


def describe_categorical_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate description for categorical dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'clustering', etc.).

    Returns:
        String description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset, task_type)

    description = [
        "This is a categorical dataset.",
        f"It has {info.n_samples} samples.",
        "Categorical feature summary:",
    ]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        description.append(f"- {col}: {X[col].nunique()} unique values")

    # Task-specific description of the target variable
    if task_type == "classification":
        description.append(
            f"\nTarget variable has {y.nunique()} unique classes for classification."
        )
        description.append("Label distribution:")
        description.append(str(y.value_counts()))
    elif task_type == "clustering":
        description.append("\nTarget variable is provided for clustering evaluation.")

    # Add handling requirements
    requirements = [
        "\n### Categorical Data Handling Requirements",
        "- **Encoding**: Ensure categorical features are appropriately encoded (e.g., one-hot, target encoding).",
        "- **High Cardinality**: For features with many unique values, consider target encoding or feature hashing.",
        "- **Model Choice**: Tree-based models (like CatBoost, LightGBM) can handle categorical features natively.",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_text_dataset(dataset: Dict[str, Any], task_type: str) -> str:
    """
    Generate description for text dataset, tailored to the ML task.

    Args:
        dataset: Dataset dictionary.
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.
    """
    X, y = dataset["X"], dataset["y"]
    info = get_dataset_info(dataset, task_type)

    # Assuming X is a Series of text for this description
    if not isinstance(X, pd.Series):
        X = pd.Series(X.squeeze(), name="text_feature")

    description = [
        "This is a text dataset.",
        f"Number of text entries: {info.n_samples}",
        f"Average text length: {X.str.len().mean():.2f} characters",
    ]

    # Task-specific description of the target variable
    description.append("\n### Target Variable (y) Analysis")
    if task_type == "classification":
        description.append(f"* **Task**: Text Classification.")
        description.append(f"* **Number of Classes**: {y.nunique()}")
        description.append(f"* **Class Distribution**: {y.value_counts().to_dict()}")
    elif task_type == "regression":
        description.append(
            f"* **Task**: Text Regression (e.g., predicting a sentiment score)."
        )
        description.append(f"* **Target Summary**: \n{y.describe().to_string()}")

    # Add handling requirements
    requirements = [
        "\n### Text Data Handling Requirements",
        "1. **Preprocessing**: Tokenization, stop-word removal, and stemming/lemmatization "
        "are standard.",
        "2. **Feature Extraction**: Convert text to numerical vectors using TF-IDF, word "
        "embeddings (Word2Vec, GloVe), or transformer-based embeddings (BERT).",
        "3. **Model-Specific**: For RNNs/Transformers, ensure inputs are padded/truncated "
        "to a consistent sequence length.",
    ]

    description.extend(requirements)
    return "\n".join(description)


def describe_dataset(
    dataset: Dict[str, Any],
    dataset_type: str = "tabular",
    task_type: str = "classification",
) -> str:
    """
    Generate a detailed, task-aware description of a dataset.

    Args:
        dataset: Dataset dictionary with 'X' and 'y' keys.
        dataset_type: Type of dataset ('tabular', 'time_series', 'image', 'categorical', 'text').
        task_type: The machine learning task ('classification', 'regression', etc.).

    Returns:
        String description of the dataset.

    Raises:
        ValidationError: If dataset type is not supported or processing fails.
    """
    try:
        # First, format the dataset which includes preprocessing
        processed_dataset = format_dataset(dataset)
        validate_dataset_type(dataset_type)

        description_functions = {
            "tabular": describe_tabular_dataset,
            "time_series": describe_time_series_dataset,
            "image": describe_image_dataset,
            "categorical": describe_categorical_dataset,
            "text": describe_text_dataset,
        }

        # Describe the *processed* dataset using the appropriate task-aware function
        description = description_functions[dataset_type](processed_dataset, task_type)
        description = add_general_description(description, processed_dataset)
        return description

    except Exception as e:
        # Re-raise as a validation error for consistent error handling
        raise ValidationError(f"Failed to describe dataset: {e}")


def add_general_description(description: str, dataset: Dict[str, Any]) -> str:
    """
    Add a preview of the processed feature data to the description.
    """
    description += f"\n\nThe dataset features (X) start with the following values:\n{dataset['X'].head().to_string()}"
    return description


def generate_dataset_task_types(dataset_type: str) -> List[str]:
    """
    Generate list of task types for given dataset type.

    Args:
        dataset_type: Type of dataset

    Returns:
        List of supported task types

    Raises:
        ValidationError: If dataset type is not supported
    """
    try:
        validate_dataset_type(dataset_type)
        return DATASET_TASK_MAPPING.get(dataset_type, [])
    except Exception as e:
        raise ValidationError(f"Failed to generate task types: {e}")
