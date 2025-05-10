from ConfigSpace import Configuration
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
import pandas as pd

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    X = dataset['X']
    y = dataset['y']

    if isinstance(X, pd.DataFrame):
        X = X.values.reshape(-1, X.shape[1])  # Fix reshape on DataFrame
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    model_type = cfg.get('model')
    if model_type == 'mlp':
        model = nn.Sequential(
            nn.Linear(input_size, cfg.get('hidden_layer_sizes')),
            nn.ReLU(),
            nn.Linear(cfg.get('hidden_layer_sizes'), num_classes)
        )
    elif model_type == 'cnn':
        if len(X.shape) != 4:
            if len(X.shape) == 2:
                side_length = int(np.sqrt(X.shape[1]))
                if side_length ** 2 != X.shape[1]:
                    raise ValueError("Input size is not a perfect square.")
                X = np.reshape(X, (-1, 1, side_length, side_length))
            else:
                raise ValueError("Invalid input shape for CNN.")
        model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * (X.shape[2] - 4) * (X.shape[3] - 4), cfg.get('hidden_layer_sizes')),
            nn.ReLU(),
            nn.Linear(cfg.get('hidden_layer_sizes'), num_classes)
        )
    else:
        raise ValueError("Invalid model type.")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X, y = X.to(device), y.to(device)

    learning_rate_type = cfg.get('learning_rate')
    eta0 = cfg.get('eta0', 0.1)  # Provide default value if eta0 is not present
    if eta0 is None:
        raise ValueError("eta0 is not provided in the configuration.")
    if learning_rate_type == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate_type == 'invscaling':
        optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate_type == 'adaptive':
        optimizer = optim.Adam(model.parameters(), lr=eta0)
    else:
        raise ValueError("Invalid learning rate type.")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / 10.0
    return average_loss
