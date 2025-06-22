import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """Trains a CNN model on the given dataset using the provided configuration.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): A dictionary containing the training data ('X' and 'y').
        seed (int): The random seed for reproducibility.

    Returns:
        float: The negative accuracy on the training data.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract data
    X = dataset["X"]
    y = dataset["y"]

    # Validate data type and shape (critical: NO try/except)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[1] != 784:
        raise ValueError("X must have 784 features")

    # Data preprocessing for CNN
    X = X.astype(np.float32).values / 255.0  # Normalize pixel values
    X = X.reshape(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
    y = y.astype(np.int64).values

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Extract hyperparameters
    learning_rate = cfg.get("learning_rate")
    batch_size = cfg.get("batch_size")
    epochs = cfg.get("epochs")
    optimizer_name = cfg.get("optimizer")
    dropout_rate = cfg.get("dropout_rate")
    num_conv_layers = cfg.get("num_conv_layers")
    num_filters_l1 = cfg.get("num_filters_l1")
    num_filters_l2 = cfg.get("num_filters_l2")
    num_filters_l3 = cfg.get("num_filters_l3")
    kernel_size = cfg.get("kernel_size")

    # Validate hyperparameter values (critical: NO try/except)
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        raise ValueError("Learning rate must be a positive float")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Epochs must be a positive integer")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("Dropout rate must be between 0 and 1")

    # Define the CNN model
    class CNN(nn.Module):
        def __init__(self, num_conv_layers, num_filters_l1, num_filters_l2, num_filters_l3, kernel_size, dropout_rate):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, num_filters_l1, kernel_size=kernel_size)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv_layers = nn.ModuleList()
            num_filters_in = num_filters_l1

            if num_conv_layers > 1:
                self.conv_layers.append(
                    nn.Sequential(nn.Conv2d(num_filters_in, num_filters_l2, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
                )
                num_filters_in = num_filters_l2

            if num_conv_layers > 2:
                self.conv_layers.append(
                    nn.Sequential(nn.Conv2d(num_filters_in, num_filters_l3, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
                )

            self.dropout = nn.Dropout(dropout_rate)

            # Dynamically calculate the input size for the linear layer
            example_input = torch.randn(1, 1, 28, 28)
            x = self.conv1(example_input)
            x = self.relu1(x)
            x = self.pool1(x)
            for layer in self.conv_layers:
                x = layer(x)

            self.flatten = nn.Flatten()
            x = self.flatten(x)

            self.fc = nn.Linear(x.size(1), 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            for layer in self.conv_layers:
                x = layer(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = CNN(num_conv_layers, num_filters_l1, num_filters_l2, num_filters_l3, kernel_size, dropout_rate)

    # Define the optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        momentum = cfg.get("momentum")
        if not isinstance(momentum, float) or momentum < 0 or momentum > 1:
            raise ValueError("Momentum must be between 0 and 1 for SGD")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for i in range(0, len(X_tensor), batch_size):
            # Get batch
            X_batch = X_tensor[i : i + batch_size]
            y_batch = y_tensor[i : i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate accuracy on the training data
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_tensor).sum().item()
        accuracy = correct / len(y_tensor)

    # Return negative accuracy for SMAC optimization
    return -accuracy
