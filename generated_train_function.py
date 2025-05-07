import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ConfigSpace import Configuration
from typing import Any

class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(cfg: Configuration, seed: int, dataset: dict) -> float:
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Infer input and output dimensions dynamically from the dataset
    X = np.array(dataset['X'])
    if X.ndim == 1 or X.shape[1] == 784:
        X = X.reshape(-1, 784)
    else:
        X = X.reshape(X.shape[0], -1)
    input_size = X.shape[1]
    num_classes = len(np.unique(dataset['y']))

    # Create dataset and data loader
    dataset = ImageDataset(X, dataset['y'])
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    learning_rate = cfg.get('learning_rate')
    eta0 = cfg.get('eta0', 0.01)  # Provide a default value
    if learning_rate == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate == 'invscaling':
        try:
            power_t = cfg.get('power_t', 0.25)  # Provide a default value
            optimizer = optim.SGD(model.parameters(), lr=eta0, power_t=power_t)
        except Exception as e:
            print(f"An error occurred: {e}")
            optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate == 'adaptive':
        optimizer = optim.Adam(model.parameters(), lr=eta0)

    # Train model
    losses = []
    for epoch in range(10):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    # Return average training loss over 10 epochs
    return np.mean(losses)
