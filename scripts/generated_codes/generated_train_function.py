import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ConfigSpace import Configuration
import os


def train(cfg: Configuration, dataset: dict, seed: int, model_dir: str = None) -> float:
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract hyperparameters efficiently
    learning_rate = float(cfg.get("learning_rate", 0.01))
    regularization_strength = float(cfg.get("regularization_strength", 0.0))
    num_hidden_layers = int(cfg.get("num_hidden_layers", 1))
    num_units_in_hidden_layer = int(cfg.get("num_units_in_hidden_layer", 10))
    dropout_rate = float(cfg.get("dropout_rate", 0.0))
    activation_function = str(cfg.get("activation_function", "relu"))
    optimizer_name = str(cfg.get("optimizer", "adam"))

    # Prepare data efficiently
    X = torch.tensor(dataset["X"].values, dtype=torch.float32)
    y = torch.tensor(dataset["y"].values, dtype=torch.long)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Infer input and output dimensions dynamically
    input_dim = X.shape[1]
    output_dim = len(np.unique(y.numpy()))

    # Initialize model with optimized parameters
    class Net(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Net, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, num_units_in_hidden_layer))
            for _ in range(num_hidden_layers - 1):
                self.layers.append(nn.Linear(num_units_in_hidden_layer, num_units_in_hidden_layer))
            self.layers.append(nn.Linear(num_units_in_hidden_layer, output_dim))
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    if activation_function == "relu":
                        x = torch.relu(x)
                    elif activation_function == "tanh":
                        x = torch.tanh(x)
                    elif activation_function == "sigmoid":
                        x = torch.sigmoid(x)
                    x = self.dropout(x)
            return x

    model = Net(input_dim, output_dim)

    # Optimized training loop
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluate the model on the validation set and calculate the accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total

    # Save model if directory is provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{seed}.pth")
        torch.save(model.state_dict(), model_path)

    return -accuracy
