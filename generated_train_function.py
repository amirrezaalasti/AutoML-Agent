from typing import Any
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np

def train(cfg: Any, seed: int, dataset: Any) -> float:
    learning_rate: str = cfg.get('learning_rate')
    alpha: float = cfg.get('alpha')
    max_iter: int = cfg.get('max_iter')
    eta0: float = cfg.get('eta0', 1.0)

    X: np.ndarray = dataset['X']
    y: np.ndarray = dataset['y']

    classifier = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=alpha,
        max_iter=max_iter,
        learning_rate=learning_rate,
        eta0=1.0 if learning_rate == 'constant' else 1.0,
        warm_start=True,
        random_state=seed
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        scores = cross_val_score(
            classifier,
            X,
            y,
            cv=5,
            scoring='accuracy',
            error_score='raise'
        )

    loss: float = 1.0 - np.mean(scores)
    return loss
