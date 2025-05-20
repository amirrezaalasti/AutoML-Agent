from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ConfigSpace import Configuration
import math
import pandas as pd


def train(cfg: Configuration, dataset: Any) -> float:
    """
    Trains a neural network model on the given dataset.

    Args:
        cfg (Configuration): The configuration object containing hyperparameters.
        dataset (Any): A dictionary containing the training data ('X' and 'y').

    Returns:
        float: The average training loss over 10 epochs.
    """

    X = dataset["X"]
    y = dataset["y"]

    # Infer input and output dimensions
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    n_samples, n_features = X.shape
    num_classes = len(np.unique(y))

    # Image Data Handling Requirements

    # 1. Input Format Requirements:
    #    - For CNN models: Input must be in (batch, channels, height, width) format
    #    - For dense/linear layers: Input should be flattened

    # 2. Data Processing Steps:
    #    a) For flattened input (2D):
    #       - Calculate dimensions: height = width = int(sqrt(n_features))
    #       - Verify square dimensions: height * height == n_features
    #       - Reshape to (N, 1, H, W) for CNNs
    #    b) For 3D input (N, H, W):
    #       - Add channel dimension: reshape to (N, 1, H, W)
    #    c) For 4D input:
    #       - Verify channel order matches framework requirements

    # Framework-Specific Format:
    #    - PyTorch: (N, C, H, W)
    #    - TensorFlow: (N, H, W, C)
    #    - Convert between formats if necessary

    # Check if the input is flattened
    if len(X.shape) == 2:
        height = width = int(math.sqrt(n_features))
        if height * width != n_features:
            raise ValueError("Input features are not a perfect square.")
        X = X.reshape(n_samples, 1, height, width)
    elif len(X.shape) == 3:
        X = np.expand_dims(X, axis=1)
    elif len(X.shape) == 4:
        pass  # Assume correct format
    else:
        raise ValueError("Unsupported input dimension.")

    # 4. Normalization:
    #    - Scale pixel values to [0, 1] by dividing by 255.0
    #    - Or standardize to mean=0, std=1

    X = X / 255.0  # Scale pixel values to [0, 1]

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = cfg.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = 32  # Use a default valid batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes, num_layers, num_units, dropout, input_height, input_width):
            super(SimpleCNN, self).__init__()
            self.features = nn.ModuleList()
            in_channels = 1
            current_height = input_height
            current_width = input_width
            for i in range(num_layers):
                self.features.append(nn.Conv2d(in_channels, num_units, kernel_size=3, padding=1))
                self.features.append(nn.ReLU())
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = num_units
                num_units = max(num_units // 2, 16)
                current_height = current_height // 2
                current_width = current_width // 2

            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(dropout)
            # Dynamic Calculation
            self.classifier = nn.Linear(in_channels * current_height * current_width, num_classes)

        def forward(self, x):
            for layer in self.features:
                x = layer(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

    # Instantiate the model
    num_layers = cfg.get("num_layers")
    num_units = cfg.get("num_units")
    dropout = cfg.get("dropout")
    input_height = X.shape[2]
    input_width = X.shape[3]
    model = SimpleCNN(num_classes, num_layers, num_units, dropout, input_height, input_width)

    # Define the optimizer
    optimizer_name = cfg.get("optimizer")
    learning_rate = cfg.get("learning_rate")
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 10
    total_loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

    avg_loss = total_loss / num_epochs

    return float(avg_loss)
