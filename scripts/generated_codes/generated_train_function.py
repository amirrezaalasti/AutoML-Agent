import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from ConfigSpace import Configuration
from typing import Any


def train(cfg: Configuration, dataset: dict, seed: int, model_dir: str = None) -> float:
    """
    Train a neural network model using the provided configuration and dataset.

    Args:
    - cfg (Configuration): The configuration for the model.
    - dataset (dict): A dictionary containing the training data, with 'X' and 'y' keys.
    - seed (int): The random seed for reproducibility.
    - model_dir (str, optional): The directory to save the trained model. Defaults to None.

    Returns:
    - float: The final loss value of the model.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract hyperparameters efficiently
    learning_rate = cfg.get("learning_rate", 0.001)
    n_layers = cfg.get("n_layers", 1)
    n_units = cfg.get("n_units", 64)
    dropout_rate = cfg.get("dropout_rate", 0.1)
    l1_regularization = cfg.get("l1_regularization", 0.0)
    l2_regularization = cfg.get("l2_regularization", 0.0)
    activation = cfg.get("activation", "relu")
    optimizer_name = cfg.get("optimizer", "adam")
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 100)

    # Prepare data efficiently
    X = torch.tensor(dataset["X"].values, dtype=torch.float32)
    y = torch.tensor(dataset["y"].values, dtype=torch.float32)
    scaler = StandardScaler()
    y_scaled = torch.tensor(scaler.fit_transform(y.reshape(-1, 1)), dtype=torch.float32).squeeze()

    # Initialize model with optimized parameters
    if activation == "relu":
        activation_fn = nn.ReLU()
    elif activation == "tanh":
        activation_fn = nn.Tanh()
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()

    if optimizer_name == "adam":
        optimizer = optim.Adam
    elif optimizer_name == "sgd":
        optimizer = optim.SGD
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc_layers = nn.ModuleList([nn.Linear(X.shape[1], n_units)])
            for i in range(n_layers - 1):
                self.fc_layers.append(nn.Linear(n_units, n_units))
            self.fc_layers.append(nn.Linear(n_units, 1))

        def forward(self, x):
            for i, layer in enumerate(self.fc_layers[:-1]):
                x = activation_fn(layer(x))
                x = nn.Dropout(dropout_rate)(x)
            x = self.fc_layers[-1](x)
            return x

    model = Net()

    # Optimized training loop
    dataset = TensorDataset(X, y_scaled)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)

    for epoch in range(epochs):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save model if directory is provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{seed}.pth")
        torch.save(model.state_dict(), model_path)

    return loss.item()
