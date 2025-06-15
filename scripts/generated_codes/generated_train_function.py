import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ConfigSpace import Configuration
from typing import Any
import math
import pandas as pd


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """Trains a model based on the given configuration and dataset.

    Args:
        cfg (Configuration): Configuration object containing hyperparameters.
        dataset (Any): Dictionary containing 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: The loss value after training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = dataset["X"], dataset["y"]

    # Determine input shape and number of classes
    n_samples = X.shape[0]

    if isinstance(X, pd.DataFrame):
        X = X.values  # Convert DataFrame to NumPy array

    if len(X.shape) == 2:  # Flattened input
        n_features = X.shape[1]
        height = width = int(math.sqrt(n_features))
        assert height * width == n_features, "Input features are not square."
        X = X.reshape(n_samples, 1, height, width)  # Reshape to (N, 1, H, W)
    elif len(X.shape) == 3:  # 3D input (N, H, W)
        X = X.reshape(n_samples, 1, X.shape[1], X.shape[2])  # Add channel dimension
    elif len(X.shape) == 4:  # 4D input
        pass  # Assume (N, C, H, W) or (N, H, W, C) and handle in model

    num_classes = len(np.unique(y))

    # Normalize data to [0, 1]
    X = X.astype(np.float32) / 255.0

    # Convert data to PyTorch tensors
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=torch.long)  # Ensure labels are long type

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    # Extract hyperparameters from configuration
    model_type = cfg.get("model_type")
    batch_size = cfg.get("batch_size")
    learning_rate = cfg.get("learning_rate")
    optimizer_name = cfg.get("optimizer")
    num_epochs = 5  # Constant epochs for faster execution

    # Model definition
    class CNN(nn.Module):
        def __init__(self, num_classes, cfg):
            super(CNN, self).__init__()
            num_conv_layers = cfg.get("num_conv_layers")
            conv_filters_1 = cfg.get("conv_filters_1")
            conv_filters_2 = cfg.get("conv_filters_2") if num_conv_layers >= 2 else 0
            conv_filters_3 = cfg.get("conv_filters_3") if num_conv_layers == 3 else 0
            kernel_size = cfg.get("kernel_size")
            pooling_type = cfg.get("pooling_type")
            normalization_strategy = cfg.get("normalization_strategy")

            self.conv1 = nn.Conv2d(1, conv_filters_1, kernel_size=kernel_size)
            self.bn1 = nn.BatchNorm2d(conv_filters_1) if normalization_strategy == "batchnorm" else nn.Identity()
            self.ln1 = (
                nn.LayerNorm([conv_filters_1, X.shape[2] - kernel_size + 1, X.shape[3] - kernel_size + 1])
                if normalization_strategy == "layernorm"
                else nn.Identity()
            )

            if pooling_type == "max":
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(conv_filters_1, conv_filters_2, kernel_size=kernel_size) if num_conv_layers >= 2 else None
            self.bn2 = nn.BatchNorm2d(conv_filters_2) if num_conv_layers >= 2 and normalization_strategy == "batchnorm" else nn.Identity()
            self.ln2 = (
                nn.LayerNorm([conv_filters_2, (X.shape[2] - kernel_size + 1) // 2, (X.shape[3] - kernel_size + 1) // 2])
                if num_conv_layers >= 2 and normalization_strategy == "layernorm"
                else nn.Identity()
            )

            if pooling_type == "max":
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) if num_conv_layers >= 2 else None
            else:
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) if num_conv_layers >= 2 else None

            self.conv3 = nn.Conv2d(conv_filters_2, conv_filters_3, kernel_size=kernel_size) if num_conv_layers == 3 else None
            self.bn3 = nn.BatchNorm2d(conv_filters_3) if num_conv_layers == 3 and normalization_strategy == "batchnorm" else nn.Identity()
            self.ln3 = (
                nn.LayerNorm([conv_filters_3, ((X.shape[2] - kernel_size + 1) // 2) // 2, ((X.shape[3] - kernel_size + 1) // 2) // 2])
                if num_conv_layers == 3 and normalization_strategy == "layernorm"
                else nn.Identity()
            )

            if pooling_type == "max":
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) if num_conv_layers == 3 else None
            else:
                self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) if num_conv_layers == 3 else None

            # Calculate the output size of the convolutional layers
            if num_conv_layers == 1:
                conv_out_size = (X.shape[2] - kernel_size + 1) // 2 * (X.shape[3] - kernel_size + 1) // 2 * conv_filters_1
            elif num_conv_layers == 2:
                conv_out_size = ((X.shape[2] - kernel_size + 1) // 2) // 2 * ((X.shape[3] - kernel_size + 1) // 2) // 2 * conv_filters_2
            else:  # num_conv_layers == 3
                conv_out_size = (((X.shape[2] - kernel_size + 1) // 2) // 2) // 2 * (((X.shape[3] - kernel_size + 1) // 2) // 2) // 2 * conv_filters_3

            self.fc1 = nn.Linear(conv_out_size, cfg.get("dense_size_1"))
            self.bn_fc1 = nn.BatchNorm1d(cfg.get("dense_size_1")) if normalization_strategy == "batchnorm" else nn.Identity()
            self.ln_fc1 = nn.LayerNorm(cfg.get("dense_size_1")) if normalization_strategy == "layernorm" else nn.Identity()

            self.fc2 = nn.Linear(cfg.get("dense_size_1"), num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.bn1(x)
            x = self.ln1(x)
            x = self.pool1(x)

            if self.conv2 is not None:
                x = torch.relu(self.conv2(x))
                x = self.bn2(x)
                x = self.ln2(x)
                x = self.pool2(x)

            if self.conv3 is not None:
                x = torch.relu(self.conv3(x))
                x = self.bn3(x)
                x = self.ln3(x)
                x = self.pool3(x)

            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.bn_fc1(x)
            x = self.ln_fc1(x)
            x = self.fc2(x)
            return x

    class MLP(nn.Module):
        def __init__(self, num_classes, cfg):
            super(MLP, self).__init__()
            dense_size_1 = cfg.get("dense_size_1")
            dense_size_2 = cfg.get("dense_size_2")
            dropout_rate = cfg.get("dropout_rate")
            normalization_strategy = cfg.get("normalization_strategy")
            num_dense_layers = cfg.get("num_dense_layers")

            self.fc1 = nn.Linear(X.shape[1] * X.shape[2] * X.shape[3], dense_size_1)
            self.bn1 = nn.BatchNorm1d(dense_size_1) if normalization_strategy == "batchnorm" else nn.Identity()
            self.ln1 = nn.LayerNorm(dense_size_1) if normalization_strategy == "layernorm" else nn.Identity()
            self.dropout1 = nn.Dropout(dropout_rate)

            self.fc2 = nn.Linear(dense_size_1, dense_size_2) if num_dense_layers == 2 else None
            self.bn2 = nn.BatchNorm1d(dense_size_2) if num_dense_layers == 2 and normalization_strategy == "batchnorm" else nn.Identity()
            self.ln2 = nn.LayerNorm(dense_size_2) if num_dense_layers == 2 and normalization_strategy == "layernorm" else nn.Identity()
            self.dropout2 = nn.Dropout(dropout_rate) if num_dense_layers == 2 else None

            self.fc_out = nn.Linear(dense_size_2 if num_dense_layers == 2 else dense_size_1, num_classes)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.bn1(x)
            x = self.ln1(x)
            x = self.dropout1(x)

            if self.fc2 is not None:
                x = torch.relu(self.fc2(x))
                x = self.bn2(x)
                x = self.ln2(x)
                x = self.dropout2(x)

            x = self.fc_out(x)
            return x

    if model_type == "cnn":
        model = CNN(num_classes, cfg).to(device)
    else:  # model_type == "mlp"
        model = MLP(num_classes, cfg).to(device)

    # Optimizer definition
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:  # optimizer_name == "sgd"
        sgd_momentum = cfg.get("sgd_momentum")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for i in range(0, n_samples, batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()
