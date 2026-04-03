import numpy as np
import pandas as pd
from production.config import PREDICTION_HORIZON


def compute_target(df: pd.DataFrame) -> pd.Series:
    """
    Computes the canonical prediction target.

    Target definition:
        log return over fixed holding period (H = 3 days)

    Formula:
        target[t] = log(Close[t + H] / Close[t])

    Rules:
    - Uses ONLY future prices (no leakage in features)
    - Single, immutable target for the entire system
    """

    assert "Close" in df.columns, "DataFrame must contain 'Close' column"

    target = np.log(
        df["Close"].shift(-PREDICTION_HORIZON) / df["Close"]
    )

    return target
