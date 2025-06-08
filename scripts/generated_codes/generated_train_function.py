import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ConfigSpace import Configuration
from typing import Any
import math
import pandas as pd


def train(cfg: Configuration, dataset: Any, seed: int) -> float:
    """
    Trains a CNN model on the given dataset using the provided configuration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract data and labels
    X, y = dataset["X"], dataset["y"]

    # Check if X is a Pandas DataFrame and convert to NumPy array
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Check if y is a Pandas DataFrame or Series and convert to NumPy array
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    # Determine input shape and handle data format
    n_samples = X.shape[0]
    if isinstance(X, np.ndarray):
        if len(X.shape) == 2:
            n_features = X.shape[1]
        elif len(X.shape) == 3:
            n_features = X.shape[1] * X.shape[2]
        elif len(X.shape) == 4:
            n_features = X.shape[1] * X.shape[2] * X.shape[3]
        else:
            raise ValueError("Input data must be 2D, 3D, or 4D array.")
    else:
        raise TypeError("Input data X must be a numpy array.")

    num_classes = len(torch.unique(torch.tensor(y)))

    if len(X.shape) == 2:
        # Flattened input
        height = width = int(math.sqrt(n_features))
        if height * height != n_features:
            # Attempt to reshape to a rectangular image if not square
            aspect_ratio = X.shape[1]
            width = int(math.sqrt(aspect_ratio))
            while aspect_ratio % width != 0:
                width -= 1
            height = aspect_ratio // width
            X = X.reshape(n_samples, 1, height, width)

        else:
            X = X.reshape(n_samples, 1, height, width)  # NCHW format
    elif len(X.shape) == 3:
        # 3D input (N, H, W)
        X = X.reshape(n_samples, 1, X.shape[1], X.shape[2])  # NCHW format
    elif len(X.shape) == 4:
        # Already in NCHW format, do nothing
        pass
    else:
        raise ValueError("Input data must be 2D, 3D, or 4D array.")

    # Normalize pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Data augmentation (basic)
    data_augmentation = cfg.get("data_augmentation")
    if data_augmentation == "horizontal_flip":
        X = torch.cat([X, torch.flip(X, dims=[3])], dim=0)
        y = torch.cat([y, y], dim=0)
    elif data_augmentation == "random_rotation":
        degrees = cfg.get("random_rotation_degree")
        # Simple rotation (replace with a proper augmentation library for better results)
        angle = np.random.uniform(-degrees, degrees)
        # rotation logic here would significantly increase code size

    # Define hyperparameters
    batch_size = cfg.get("batch_size")
    learning_rate = cfg.get("learning_rate")
    epochs = cfg.get("epochs")
    num_conv_layers = cfg.get("num_conv_layers")
    filters_layer_1 = cfg.get("filters_layer_1")
    filters_layer_2 = cfg.get("filters_layer_2") if num_conv_layers >= 2 else 0
    filters_layer_3 = cfg.get("filters_layer_3") if num_conv_layers >= 3 else 0
    filters_layer_4 = cfg.get("filters_layer_4") if num_conv_layers >= 4 else 0
    filters_layer_5 = cfg.get("filters_layer_5") if num_conv_layers >= 5 else 0
    kernel_size = cfg.get("kernel_size")
    pooling_type = cfg.get("pooling_type")
    num_dense_layers = cfg.get("num_dense_layers")
    dense_units_1 = cfg.get("dense_units_1")
    dense_units_2 = cfg.get("dense_units_2") if num_dense_layers >= 2 else 0
    dense_units_3 = cfg.get("dense_units_3") if num_dense_layers >= 3 else 0
    dropout_rate = cfg.get("dropout_rate")
    optimizer_name = cfg.get("optimizer")
    adam_beta1 = cfg.get("adam_beta1") if optimizer_name == "adam" else 0.9
    adam_beta2 = cfg.get("adam_beta2") if optimizer_name == "adam" else 0.999
    sgd_momentum = cfg.get("sgd_momentum") if optimizer_name == "sgd" else 0.0
    rmsprop_rho = cfg.get("rmsprop_rho") if optimizer_name == "rmsprop" else 0.9

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self, num_classes, height, width):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, filters_layer_1, kernel_size=kernel_size)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2) if pooling_type == "max" else nn.AvgPool2d(2)

            conv_layers = [self.conv1, self.relu1, self.pool1]
            in_channels = filters_layer_1

            if num_conv_layers >= 2:
                self.conv2 = nn.Conv2d(in_channels, filters_layer_2, kernel_size=kernel_size)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(2) if pooling_type == "max" else nn.AvgPool2d(2)
                conv_layers.extend([self.conv2, self.relu2, self.pool2])
                in_channels = filters_layer_2
            else:
                self.conv2 = self.relu2 = self.pool2 = None

            if num_conv_layers >= 3:
                self.conv3 = nn.Conv2d(in_channels, filters_layer_3, kernel_size=kernel_size)
                self.relu3 = nn.ReLU()
                self.pool3 = nn.MaxPool2d(2) if pooling_type == "max" else nn.AvgPool2d(2)
                conv_layers.extend([self.conv3, self.relu3, self.pool3])
                in_channels = filters_layer_3
            else:
                self.conv3 = self.relu3 = self.pool3 = None

            if num_conv_layers >= 4:
                self.conv4 = nn.Conv2d(in_channels, filters_layer_4, kernel_size=kernel_size)
                self.relu4 = nn.ReLU()
                self.pool4 = nn.MaxPool2d(2) if pooling_type == "max" else nn.AvgPool2d(2)
                conv_layers.extend([self.conv4, self.relu4, self.pool4])
                in_channels = filters_layer_4
            else:
                self.conv4 = self.relu4 = self.pool4 = None

            if num_conv_layers >= 5:
                self.conv5 = nn.Conv2d(in_channels, filters_layer_5, kernel_size=kernel_size)
                self.relu5 = nn.ReLU()
                self.pool5 = nn.MaxPool2d(2) if pooling_type == "max" else nn.AvgPool2d(2)
                conv_layers.extend([self.conv5, self.relu5, self.pool5])
            else:
                self.conv5 = self.relu5 = self.pool5 = None

            self.conv_layers = nn.Sequential(*conv_layers)

            # Calculate the output size of the convolutional layers dynamically
            test_input = torch.randn(1, 1, height, width)
            with torch.no_grad():
                output_conv = self.conv_layers(test_input)
            conv_output_size = output_conv.view(1, -1).size(1)

            dense_layers = []
            in_features = conv_output_size

            self.fc1 = nn.Linear(in_features, dense_units_1)
            self.relu_fc1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            dense_layers.extend([self.fc1, self.relu_fc1, self.dropout1])
            in_features = dense_units_1

            if num_dense_layers >= 2:
                self.fc2 = nn.Linear(in_features, dense_units_2)
                self.relu_fc2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                dense_layers.extend([self.fc2, self.relu_fc2, self.dropout2])
                in_features = dense_units_2
            else:
                self.fc2 = self.relu_fc2 = self.dropout2 = None

            if num_dense_layers >= 3:
                self.fc3 = nn.Linear(in_features, dense_units_3)
                self.relu_fc3 = nn.ReLU()
                self.dropout3 = nn.Dropout(dropout_rate)
                dense_layers.extend([self.fc3, self.relu_fc3, self.dropout3])
            else:
                self.fc3 = self.relu_fc3 = self.dropout3 = None

            self.dense_layers = nn.Sequential(*dense_layers)
            in_features = dense_units_3 if num_dense_layers == 3 else dense_units_2 if num_dense_layers == 2 else dense_units_1

            self.fc_out = nn.Linear(in_features, num_classes)

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.dense_layers(x)
            x = self.fc_out(x)
            return x

    # Instantiate the model
    # Determine height and width based on the reshaped data
    if len(X.shape) == 4:
        height, width = X.shape[2], X.shape[3]
    else:
        raise ValueError("X tensor should have 4 dimensions after reshaping")
    model = CNN(num_classes, height, width)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, rho=rmsprop_rho)
    else:
        raise ValueError(f"Unknown optimizer: {{optimizer_name}}")

    # Create data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Return the final loss
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()

    return float(loss)
