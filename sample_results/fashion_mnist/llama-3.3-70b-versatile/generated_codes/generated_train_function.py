import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ConfigSpace import Configuration


def train(cfg: Configuration, seed: int, dataset: dict) -> float:
    """
    Train a neural network model on the given dataset.

    Args:
    - cfg (Configuration): A Configuration object containing hyperparameters.
    - seed (int): The random seed for reproducibility.
    - dataset (dict): A dictionary containing the feature matrix 'X' and label vector 'y'.

    Returns:
    - loss (float): The average training loss over 10 epochs.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get the input and output dimensions dynamically from the dataset
    input_size = dataset["X"].values.shape[1]
    num_classes = len(np.unique(dataset["y"].values))

    # Check if the input data is already image-shaped
    if len(dataset["X"].values.shape) == 4:
        # If it's already image-shaped, use it as is
        X = dataset["X"].values
    else:
        # If not, reshape it to be image-shaped
        # Assuming the input size is a perfect square
        side_length = int(np.sqrt(input_size))
        if side_length**2 != input_size:
            raise ValueError("Input size is not a perfect square")
        X = dataset["X"].values.reshape(-1, 1, side_length, side_length)

    # Create a PyTorch dataset and data loader
    tensor_X = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(dataset["y"].values).long()
    dataset = TensorDataset(tensor_X, tensor_y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create a simple neural network model
    model = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(10 * (side_length - 4) ** 2, num_classes),
    )

    # Get the learning rate and optimizer from the configuration
    learning_rate = cfg.get("learning_rate")
    eta0 = cfg.get("eta0")

    if learning_rate == "constant":
        optimizer = optim.SGD(model.parameters(), lr=eta0)
    elif learning_rate == "invscaling":
        optimizer = optim.SGD(model.parameters(), lr=eta0, momentum=0.9)
    elif learning_rate == "adaptive":
        optimizer = optim.Adam(model.parameters(), lr=eta0)

    # Train the model for 10 epochs
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for epoch in range(10):
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Return the average training loss
    return total_loss / (10 * len(data_loader.dataset))
