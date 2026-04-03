import numpy as np
import pandas as pd

# ABSOLUTE imports (robust for scripts, tests, -c, CI)
from production.feature_engine import compute_features
from production.target import compute_target
from production.config import PREDICTION_HORIZON


SEQ_LEN = 60


def build_dataset(df: pd.DataFrame):
    """
    Builds model-ready sequences and targets.
    NO leakage. NO scaling before split.
    """

    features = compute_features(df)
    target = compute_target(df)

    X_raw = []
    y = []

    for i in range(SEQ_LEN, len(df) - PREDICTION_HORIZON):
        X_raw.append(features.iloc[i - SEQ_LEN:i].values)
        y.append(target.iloc[i])

    X_raw = np.array(X_raw)
    y = np.array(y)

    assert X_raw.shape[0] == y.shape[0]
    assert X_raw.shape[1] == SEQ_LEN

    return X_raw, y
