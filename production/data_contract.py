import pandas as pd


def validate_ohlcv(df: pd.DataFrame, name: str = "DataContract"):
    """
    Validates and conservatively repairs OHLCV data.

    This function is:
    - Leakage-safe
    - Deterministic
    - Suitable for research papers
    - Robust to real-world NSE data issues
    """

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    # --------------------------------------------------
    # Column existence check
    # --------------------------------------------------
    for col in required_cols:
        assert col in df.columns, f"{name}: Missing required column '{col}'"

    # --------------------------------------------------
    # Drop rows with missing critical data
    # --------------------------------------------------
    df.dropna(subset=required_cols, inplace=True)

    # --------------------------------------------------
    # Detect invalid OHLC rows
    # --------------------------------------------------
    bad_low = df["Low"] > df[["Open", "Close", "High"]].min(axis=1)
    bad_high = df["High"] < df[["Open", "Close", "Low"]].max(axis=1)

    n_bad_low = int(bad_low.sum())
    n_bad_high = int(bad_high.sum())
    n_bad_total = n_bad_low + n_bad_high

    # --------------------------------------------------
    # Conservative repair (NO smoothing, NO leakage)
    # --------------------------------------------------
    if n_bad_total > 0:
        # Fix Low violations
        df.loc[bad_low, "Low"] = df.loc[
            bad_low, ["Open", "Close", "High"]
        ].min(axis=1)

        # Fix High violations
        df.loc[bad_high, "High"] = df.loc[
            bad_high, ["Open", "Close", "Low"]
        ].max(axis=1)

        print(
            f"[{name}] Repaired {n_bad_total} invalid OHLC rows "
            f"(Low fixes: {n_bad_low}, High fixes: {n_bad_high})"
        )

    # --------------------------------------------------
    # Final hard assertions (must never fail)
    # --------------------------------------------------
    assert (df["Low"] <= df["High"]).all(), f"{name}: Low > High after repair"
    assert (df["Volume"] >= 0).all(), f"{name}: Negative volume detected"
