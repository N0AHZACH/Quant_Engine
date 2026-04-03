import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def train_base_models(X: np.ndarray, y: np.ndarray):
    """
    Trains base models for expectation + uncertainty ensemble.
    """

    # Linear anchor model (stable)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    # Non-linear ensemble model
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    return {
        "ridge": ridge,
        "rf": rf
    }


def predict_base_models(models: dict, X: np.ndarray) -> dict:
    """
    Generates predictions from all base models.
    """

    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X)

    return predictions
