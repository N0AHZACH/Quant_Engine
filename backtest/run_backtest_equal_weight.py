import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from production.dataset_builder import build_dataset
from production.base_models import train_base_models, predict_base_models
from production.ensemble_predictor import compute_mu_sigma, compute_edge


# ======================================================
# RESEARCH CONFIG (PAPER MODE)
# ======================================================
RETRAIN_EVERY = 50
MIN_TRAIN_SIZE = 300


def run_single_symbol_returns(df: pd.DataFrame) -> pd.Series:
    """
    Walk-forward backtest for ONE symbol.
    Returns daily returns (research mode).
    """

    X_seq, y = build_dataset(df)
    prices = df["Close"].values

    models = None
    daily_returns = []

    for t in range(len(X_seq) - 1):

        # ---------------------------------------------
        # Walk-forward retraining
        # ---------------------------------------------
        if models is None or (t % RETRAIN_EVERY == 0 and t >= MIN_TRAIN_SIZE):
            if t < MIN_TRAIN_SIZE:
                daily_returns.append(0.0)
                continue

            models = train_base_models(
                X_seq[:t, -1, :],
                y[:t]
            )

        # ---------------------------------------------
        # Predict ensemble edge
        # ---------------------------------------------
        preds = predict_base_models(
            models,
            X_seq[t:t + 1, -1, :]
        )

        mu, sigma = compute_mu_sigma(preds)
        edge = compute_edge(mu, sigma)[0]

        # ---------------------------------------------
        # Research return (SIGN ONLY)
        # ---------------------------------------------
        direction = np.sign(edge)
        ret = (prices[t + 1] / prices[t]) - 1.0

        daily_returns.append(direction * ret)

    return pd.Series(daily_returns)


def run_equal_weight_portfolio(gold_dir: str):
    """
    Equal-weight research portfolio across all gold symbols.

    Returns:
        portfolio_returns : pd.Series
        equity_curve      : pd.Series
    """

    files = glob.glob(os.path.join(gold_dir, "*.csv"))
    assert len(files) > 0, f"No gold CSV files found in {gold_dir}"

    symbol_returns = []

    print(f"\nRunning equal-weight research portfolio on {len(files)} symbols\n")

    for fp in tqdm(files, desc="Symbols", ncols=100):
        df = pd.read_csv(fp, index_col=0, parse_dates=True)

        # Safety guard
        if len(df) < MIN_TRAIN_SIZE + 10:
            continue

        r = run_single_symbol_returns(df)
        symbol_returns.append(r)

    assert len(symbol_returns) > 0, "No valid symbols processed"

    # --------------------------------------------------
    # Align lengths (truncate to shortest)
    # --------------------------------------------------
    min_len = min(len(r) for r in symbol_returns)
    aligned = [
        r.iloc[-min_len:].reset_index(drop=True)
        for r in symbol_returns
    ]

    # --------------------------------------------------
    # Equal-weight portfolio return
    # --------------------------------------------------
    portfolio_returns = pd.concat(aligned, axis=1).mean(axis=1)

    # Normalize equity to 1.0 (paper standard)
    equity_curve = (1.0 + portfolio_returns).cumprod()

    return portfolio_returns, equity_curve
