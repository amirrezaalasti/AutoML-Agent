from typing import Any
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train(cfg: Any, seed: int, dataset: Any) -> float:
    X: np.ndarray = dataset['X']
    y: np.ndarray = dataset['y']

    learning_rate: str = cfg.get('learning_rate')
    alpha: float = cfg.get('alpha')
    max_iter: int = cfg.get('max_iter')
    eta0: float = cfg.get('eta0', 1.0)  # default to 1.0 if not provided

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(zip(np.unique(y), class_weights))

    model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=alpha,
        max_iter=max_iter,
        learning_rate=learning_rate,
        eta0=eta0 if learning_rate == 'constant' else 1.0,
        warm_start=True,
        random_state=seed,
        class_weight=class_weights_dict
    )

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    loss = 1.0 - np.mean(scores)

    return loss
