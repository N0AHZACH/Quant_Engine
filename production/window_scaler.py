import numpy as np
from sklearn.preprocessing import StandardScaler


class WindowScaler:
    """
    Scaler that is fit ONLY on training data.
    Never refit during inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        assert not self.is_fitted, "Scaler already fitted"
        self.scaler.fit(X)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Scaler not fitted"
        return self.scaler.transform(X)
