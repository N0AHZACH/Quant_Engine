import numpy as np
import pandas as pd

from production.data_contract import validate_ohlcv

# ------------------------------------------------------------------
# Feature contract (DO NOT CHANGE ORDER)
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    "log_ret",
    "volatility",
    "trend_slope",
    "volume_zscore",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes production-grade, leakage-safe features.

    Design guarantees:
    - Explicit causality (all features use t-1 or earlier)
    - No scaling (handled later)
    - Cross-section comparable features
    - Numerically stable
    """

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------
    validate_ohlcv(df, name="FeatureEngine")

    out = pd.DataFrame(index=df.index)

    # --------------------------------------------------------------
    # 1. Log returns (EXPLICITLY causal)
    # --------------------------------------------------------------
    # Raw return uses t / t-1, then shifted to ensure no t usage
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    out["log_ret"] = log_ret.shift(1).fillna(0.0)

    # --------------------------------------------------------------
    # 2. Volatility (rolling std of past returns)
    # --------------------------------------------------------------
    out["volatility"] = (
        out["log_ret"]
        .rolling(window=20, min_periods=20)
        .std()
        .fillna(0.0)
    )

    # --------------------------------------------------------------
    # 3. Trend slope (log-price, scale invariant)
    # --------------------------------------------------------------
    log_price = np.log(df["Close"]).shift(1)

    def _slope(x: np.ndarray) -> float:
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    out["trend_slope"] = (
        log_price
        .rolling(window=20, min_periods=20)
        .apply(_slope, raw=False)
        .fillna(0.0)
    )

    # --------------------------------------------------------------
    # 4. Volume z-score (robust & causal)
    # --------------------------------------------------------------
    vol_log = np.log1p(df["Volume"]).shift(1)

    vol_mean = vol_log.rolling(window=20, min_periods=20).mean()
    vol_std = vol_log.rolling(window=20, min_periods=20).std()

    out["volume_zscore"] = (
        (vol_log - vol_mean) / (vol_std + 1e-9)
    ).clip(-5.0, 5.0).fillna(0.0)

    # --------------------------------------------------------------
    # Final safety guarantees
    # --------------------------------------------------------------
    out = out.replace([np.inf, -np.inf], 0.0)

    assert list(out.columns) == FEATURE_COLUMNS, "Feature column mismatch"
    assert not out.isnull().any().any(), "NaNs detected in features"

    return out
