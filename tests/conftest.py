import numpy as np


class ConstantPredictor:
    """Duck-type predict stub: always returns a fixed constant."""

    def __init__(self, constant: float = 50.0, n_classes: int | None = None):
        self._c = constant
        self._n_classes = n_classes

    def predict(self, X):
        return np.full(len(X), self._c)

    def predict_proba(self, X):
        n = len(X)
        n_classes = self._n_classes or max(int(self._c) + 1, 2)
        proba = np.zeros((n, n_classes))
        proba[:, int(self._c) % n_classes] = 1.0
        return proba
