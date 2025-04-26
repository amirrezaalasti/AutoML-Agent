from typing import Any
from ConfigSpace import Configuration
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import SGDClassifier
import numpy as np

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    """
    Train a SGDClassifier model on the given dataset with the specified configuration.

    Args:
    - cfg (Configuration): A sampled configuration object.
    - seed (int): The random seed for reproducibility.
    - dataset (Any): A dataset object with 'X' and 'y' attributes.

    Returns:
    - float: The loss value (1.0 - mean cross-validation accuracy).
    """

    # Extract dataset
    X, y = dataset['X'], dataset['y']

    # Extract hyperparameters
    learning_rate: str = cfg.get('learning_rate')
    alpha: float = cfg.get('alpha')
    max_iter: int = cfg.get('max_iter')
    eta0: float = cfg.get('eta0') if learning_rate == 'constant' else 0.1

    # Define model parameters
    params = {
        'loss': 'log_loss',
        'penalty': 'l2',
        'alpha': alpha,
        'max_iter': max_iter,
        'warm_start': True,
        'random_state': seed,
        'learning_rate': learning_rate,
        'eta0': eta0,  # Always provide a valid float for eta0
        'early_stopping': True,
        'n_jobs': -1  # Use all available CPU cores
    }

    # Initialize and fit the model
    model = SGDClassifier(**params)

    # Perform stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Compute and return the loss value
    loss = 1.0 - scores.mean()
    return loss
