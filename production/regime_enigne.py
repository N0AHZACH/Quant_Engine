import numpy as np
import pandas as pd

"""
Regime definitions (minimum viable):

R0 = NO TRADE / DISORDER
R1 = LOW VOL TREND
R2 = HIGH VOL TREND
R3 = MEAN REVERSION / CHOP
"""

REGIME_LABELS = {
    0: "NO_TRADE",
    1: "LOW_VOL_TREND",
    2: "HIGH_VOL_TREND",
    3: "MEAN_REVERSION"
}


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """
    Computes daily market regime using ONLY past information.
    """

    close = df["Close"]

    # --- Volatility (rolling) ---
    log_ret = np.log(close / close.shift(1)).fillna(0)
    vol = log_ret.rolling(20).std()

    # --- Trend strength (slope) ---
    slope = close.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0],
        raw=False
    )

    # --- Normalization ---
    vol_norm = vol / vol.rolling(252).mean()
    slope_norm = slope / (close.rolling(20).mean() + 1e-9)

    regime = []

    for v, s in zip(vol_norm, slope_norm):
        if np.isnan(v) or np.isnan(s):
            regime.append(0)
        elif v > 2.0:
            regime.append(0)  # disorder / crisis
        elif abs(s) > 0.002 and v < 1.2:
            regime.append(1)  # clean trend
        elif abs(s) > 0.002 and v >= 1.2:
            regime.append(2)  # volatile trend
        else:
            regime.append(3)  # chop / mean reversion

    return pd.Series(regime, index=df.index)
