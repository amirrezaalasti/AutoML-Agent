from ConfigSpace import Configuration
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble, neighbors
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    X = dataset['X']
    y = dataset['y']

    # Check if the data is image-shaped
    is_image_data = len(X.shape) == 4

    # Infer input and output dimensions dynamically
    if is_image_data:
        input_size = X.shape[1] * X.shape[2] * X.shape[3]
    else:
        input_size = X.shape[1]
    num_classes = len(np.unique(y))

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Scale the data if not image data
    if not is_image_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    classifier = cfg.get('classifier')

    if classifier in ['svm', 'rf', 'knn']:
        # Use scikit-learn models
        if classifier == 'svm':
            model = svm.SVC(
                C=cfg.get('C'),
                kernel=cfg.get('kernel'),
                degree=cfg.get('degree') if cfg.get('kernel') == 'poly' else 3,
                gamma=cfg.get('gamma') if cfg.get('kernel') in ['rbf', 'poly', 'sigmoid'] else 'scale',
                coef0=cfg.get('coef0') if cfg.get('kernel') in ['poly', 'sigmoid'] else 0,
                random_state=seed
            )
        elif classifier == 'rf':
            model = ensemble.RandomForestClassifier(
                n_estimators=cfg.get('n_estimators'),
                max_depth=cfg.get('max_depth'),
                random_state=seed
            )
        else:  # classifier == 'knn'
            model = neighbors.KNeighborsClassifier(
                n_neighbors=cfg.get('n_neighbors'),
                weights=cfg.get('weights')
            )

        model.fit(X_train, y_train)
        loss = 1 - model.score(X_val, y_val)
        return loss

    else:
        # Use PyTorch for neural networks
        if is_image_data:
            # Reshape the data
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
            X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], X_val.shape[3])
            input_size = X_train.shape[1]

            # Check if the input size is a perfect square
            if X_train.shape[2] * X_train.shape[3] != X_train.shape[2] ** 2:
                raise ValueError("Input size is not a perfect square.")

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val = torch.tensor(y_val, dtype=torch.long)

            # Define the CNN model
            model = nn.Sequential(
                nn.Conv2d(X_train.shape[1], 10, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Linear(50, num_classes)
            )

        else:
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val = torch.tensor(y_val, dtype=torch.long)

            # Define the MLP model
            model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        # Define the optimizer
        learning_rate = cfg.get('learning_rate', default=0.01)
        optimizer_type = cfg.get('optimizer', default='constant')

        if optimizer_type == 'constant':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'invscaling':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'adaptive':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            loss = loss_fn(outputs, y_val).item()

        return loss
