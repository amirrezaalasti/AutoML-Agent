import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from ConfigSpace import Configuration
from typing import Any

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    """
    Train a neural network model on the given dataset.

    Args:
    - cfg (Configuration): A Configuration object containing hyperparameters.
    - seed (int): The random seed for reproducibility.
    - dataset (dict): A dictionary containing the feature matrix 'X' and label vector 'y'.

    Returns:
    - loss (float): The average training loss over 10 epochs.
    """

    # Get the input and output dimensions dynamically from the dataset
    input_size = dataset['X'].shape[1]
    num_classes = len(np.unique(dataset['y']))

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(dataset['X'], dataset['y'], test_size=0.2, random_state=seed)

    # Get the learning rate and other hyperparameters from the configuration
    learning_rate = cfg.get('learning_rate')
    max_iter = cfg.get('max_iter')
    tol = cfg.get('tol')
    early_stopping = cfg.get('early_stopping') == 'True'
    validation_fraction = cfg.get('validation_fraction')
    warm_start = cfg.get('warm_start') == 'True'
    shuffle = cfg.get('shuffle') == 'True'
    verbose = cfg.get('verbose')

    # Create a neural network model with the given hyperparameters
    model = MLPClassifier(hidden_layer_sizes=(input_size,), max_iter=max_iter, tol=tol, 
                           early_stopping=early_stopping, validation_fraction=validation_fraction, 
                           warm_start=warm_start, shuffle=shuffle, verbose=verbose, random_state=seed)

    # Train the model and get the average training loss over 10 epochs
    losses = []
    for _ in range(10):
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)
        loss = log_loss(y_val, y_pred)
        losses.append(loss)

    # Return the average training loss
    return np.mean(losses)
