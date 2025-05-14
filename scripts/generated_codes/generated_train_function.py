from ConfigSpace import Configuration
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
import pandas as pd

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract dataset
    if isinstance(dataset, pd.DataFrame):
        X = dataset.values
    else:
        X = dataset['X']
    if isinstance(X, pd.DataFrame):
        X = X.values
    y = dataset['y']
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    # Infer input and output dimensions dynamically
    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    # Determine model type
    model_type = cfg.get('model')

    # Define model
    if model_type == 'mlp':
        model = nn.Sequential(
            nn.Linear(input_size, cfg.get('hidden_layer_sizes')),
            nn.ReLU(),
            nn.Linear(cfg.get('hidden_layer_sizes'), num_classes)
        )
    elif model_type == 'cnn':
        # Check if input is already image-shaped
        if len(X.shape) != 4:
            # Reshape input to image format (assuming square images)
            img_size = int(np.sqrt(input_size))
            if img_size * img_size != input_size:
                raise ValueError("Input size is not a perfect square")
            X = X.reshape(-1, 1, img_size, img_size)
        else:
            # Input is already image-shaped
            pass  # No need to reshape

        model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(10 * ((X.shape[2] - 4) // 2) ** 2, cfg.get('hidden_layer_sizes')),
            nn.ReLU(),
            nn.Linear(cfg.get('hidden_layer_sizes'), num_classes)
        )

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define optimizer
    learning_rate_type = cfg.get('learning_rate')
    eta0 = cfg.get('eta0', 0.01)  # Provide a default value for eta0
    if learning_rate_type == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate_type == 'invscaling':
        # PyTorch does not directly support invscaling lr schedule
        # We'll use SGD with momentum as a fallback
        optimizer = optim.SGD(model.parameters(), lr=eta0, momentum=0.9)
    elif learning_rate_type == 'adaptive':
        optimizer = optim.Adam(model.parameters(), lr=eta0)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Train model
    total_loss = 0.0
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Return average training loss over 10 epochs
    return total_loss / 10.0
