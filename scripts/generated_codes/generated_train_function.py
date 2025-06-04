from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ConfigSpace import Configuration
from math import sqrt


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a model on the given dataset using the provided configuration.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): The dataset dictionary containing 'X' and 'y'.
        seed (int): The random seed for reproducibility.

    Returns:
        float: The average training loss over 10 epochs.
    """
    X = dataset["X"]
    y = dataset["y"]

    # Convert to numpy arrays if they are not already
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Normalize the data
    X = X / 255.0

    # Determine input and output dimensions
    n_samples = X.shape[0]
    n_features = X.shape[1] if len(X.shape) > 1 else 1  # Handle case when input is 1D
    n_classes = len(np.unique(y))

    # Determine image dimensions and reshape
    height = int(sqrt(n_features))
    width = height
    assert height * width == n_features, "Input is not a square image"

    # Reshape to (N, 1, H, W) for CNNs
    X = X.reshape(n_samples, 1, height, width)

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Define the model
    if cfg.get("use_cnn"):

        class CNN(nn.Module):
            def __init__(self, num_filters, kernel_size):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size)
                self.relu1 = nn.ReLU()
                self.maxpool1 = nn.MaxPool2d(kernel_size=2)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear((height // 2) * (width // 2) * num_filters, cfg.get("num_units"))
                self.relu2 = nn.ReLU()
                self.fc2 = nn.Linear(cfg.get("num_units"), n_classes)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.maxpool1(x)
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.relu2(x)
                x = self.fc2(x)
                return x

        model = CNN(num_filters=cfg.get("num_filters"), kernel_size=cfg.get("kernel_size"))
    else:

        class DenseNet(nn.Module):
            def __init__(self):
                super(DenseNet, self).__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(n_features, cfg.get("num_units"))
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(cfg.get("num_units"), n_classes)

            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                return x

        model = DenseNet()

    # Define optimizer
    if cfg.get("optimizer") == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate"))
    elif cfg.get("optimizer") == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.get("learning_rate"))
    elif cfg.get("optimizer") == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.get("learning_rate"))
    else:
        raise ValueError(f"Unknown optimizer: {cfg.get('optimizer')}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    batch_size = cfg.get("batch_size")
    num_epochs = 10

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            # Get batch
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)  # Accumulate loss scaled to batch size

        total_loss += epoch_loss / n_samples  # Scale epoch loss to entire dataset size

    # Calculate average loss
    avg_loss = total_loss / num_epochs
    return float(avg_loss)
