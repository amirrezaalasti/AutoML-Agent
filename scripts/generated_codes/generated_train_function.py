from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ConfigSpace import Configuration
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def train(cfg: Configuration, dataset: Any) -> float:
    """
    Trains a PyTorch model on the provided dataset using the given configuration.

    Args:
        cfg (Configuration): A ConfigurationSpace object containing hyperparameters.
        dataset (Any): A dictionary containing 'X' (features) and 'y' (labels).

    Returns:
        float: The average training loss over 10 epochs.
    """

    X = dataset["X"]
    y = dataset["y"]

    # Infer input and output dimensions dynamically
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    input_size = X.shape[1] if len(X.shape) > 1 else 1
    num_classes = len(np.unique(y))

    # Determine if dataset is image-shaped
    if len(X.shape) == 4:
        # Already image shaped
        pass
    else:
        # Reshape if not image shaped, assuming square images if CNN is used
        side = int(np.sqrt(input_size))
        if side * side != input_size:
            pass
            # raise ValueError(
            #    "Input size is not a perfect square, cannot reshape for CNN. "
            #    "Consider using a different model or dataset."
            # )
        else:
            X = X.reshape(-1, 1, side, side)

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    batch_size = cfg.get("batch_size")
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = nn.Linear(32 * 7 * 7, num_classes)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = torch.flatten(x, 1)  # Flatten the tensor
            x = self.fc(x)
            return x

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, num_classes, num_layers, num_units):
            super(SimpleMLP, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, num_units))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(num_units, num_units))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(num_units, num_classes))
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    # Use CNN if dataset has images
    if len(X.shape) == 4:
        model = SimpleCNN(num_classes=num_classes)
    else:
        num_layers = cfg.get("num_layers", 2)
        num_units = cfg.get("num_units", 64)
        model = SimpleMLP(input_size, num_classes, num_layers, num_units)

    # Define the optimizer
    optimizer_name = cfg.get("optimizer", "adam")
    learning_rate_type = cfg.get("learning_rate", "constant")
    eta0 = cfg.get("eta0", 1e-3)

    if optimizer_name == "sgd":
        if learning_rate_type == "constant":
            optimizer = optim.SGD(model.parameters(), lr=eta0)
        elif learning_rate_type == "adaptive":
            optimizer = optim.Adam(model.parameters(), lr=eta0)
        else:
            optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=eta0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=eta0)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 10
    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss

    avg_loss = total_loss / num_epochs

    return float(avg_loss)
