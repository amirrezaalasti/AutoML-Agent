from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import SGDClassifier
from typing import Any
from ConfigSpace import Configuration

def train(cfg: Configuration, seed: int, dataset: Any) -> float:
    X: Any = dataset['X']
    y: Any = dataset['y']
    learning_rate: str = cfg.get('learning_rate')
    alpha: float = cfg.get('alpha')
    max_iter: int = cfg.get('max_iter')
    eta0: float = cfg.get('eta0') if learning_rate == 'constant' else 1.0

    params: dict = {
        'learning_rate': learning_rate,
        'alpha': alpha,
        'max_iter': max_iter,
        'early_stopping': True,
        'warm_start': True,
        'random_state': seed,
        'eta0': eta0 if learning_rate == 'constant' else 1.0
    }

    model: SGDClassifier = SGDClassifier(**params)
    cv: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: Any = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return 1.0 - scores.mean()
