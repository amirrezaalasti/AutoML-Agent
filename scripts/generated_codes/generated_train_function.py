from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ConfigSpace import Configuration


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a machine learning model based on the provided configuration and dataset.

    Args:
        cfg (Configuration): Configuration object containing hyperparameters.
        dataset (Any): Dataset dictionary with 'X' (features) and 'y' (labels).
        seed (int): Random seed for reproducibility.

    Returns:
        float: Average training loss over 10 epochs.
    """
    X = dataset["X"]
    y = dataset["y"]

    # Convert data to numpy arrays if it is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    model_type = cfg.get("model_type")
    learning_rate = cfg.get("learning_rate")

    # PyTorch-related setup
    if model_type in ["LSTM", "GRU", "MLP"]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if X_tensor.ndim == 1:
        X_tensor = X_tensor.unsqueeze(1)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)

    if model_type in ["LSTM", "GRU"]:
        # Reshape X_tensor to 3D if it's not already
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0)  # Add a batch dimension
        if X_tensor.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features) for {model_type}, got {X_tensor.shape}")

    if model_type == "LSTM":
        input_size = X_tensor.shape[2] if X_tensor.ndim == 3 else X_tensor.shape[1]
        hidden_size = cfg.get("lstm_units")
        num_layers = cfg.get("lstm_layers")
        dropout = cfg.get("lstm_dropout")
        output_size = 1  # Regression task

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                self.linear = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.linear(out[:, -1, :])  # Use only the last time step's output
                return out

        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)

    elif model_type == "GRU":
        input_size = X_tensor.shape[2] if X_tensor.ndim == 3 else X_tensor.shape[1]
        hidden_size = cfg.get("gru_units")
        num_layers = cfg.get("gru_layers")
        dropout = cfg.get("gru_dropout")
        output_size = 1  # Regression task

        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(GRUModel, self).__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                self.linear = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.gru(x)
                out = self.linear(out[:, -1, :])  # Use only the last time step's output
                return out

        model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout)

    elif model_type == "MLP":
        input_size = X_tensor.shape[1] if X_tensor.ndim == 2 else X_tensor.shape[1] * X_tensor.shape[2]
        hidden_size = cfg.get("mlp_units")
        num_layers = cfg.get("mlp_layers")
        dropout = cfg.get("mlp_dropout")
        output_size = 1  # Regression task

        class MLPModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(MLPModel, self).__init__()
                self.flatten = nn.Flatten()
                layers = []
                for i in range(num_layers - 1):
                    layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(input_size if num_layers == 1 else hidden_size, output_size))
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                x = self.flatten(x)
                return self.layers(x)

        model = MLPModel(input_size, hidden_size, num_layers, output_size, dropout)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    epochs = 10
    total_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / epochs
    return float(avg_loss)
