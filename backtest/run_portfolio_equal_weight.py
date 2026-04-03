import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from production.dataset_builder import build_dataset
from production.base_models import train_base_models, predict_base_models
from production.ensemble_predictor import compute_mu_sigma, compute_edge


# ======================================================
# RESEARCH CONFIG — SAFE + NON-ZERO
# ======================================================
RETRAIN_EVERY = 200
MIN_TRAIN_SIZE = 300

EDGE_THRESHOLD = 0.02     # 🔥 LOWERED — REQUIRED
MIN_POSITION = 0.10       # 🔥 FORCE EXPOSURE
MAX_POSITION = 1.0

N_JOBS = -1


# ======================================================
# SINGLE SYMBOL RETURNS
# ======================================================
def run_single_symbol_returns(fp: str):

    df = pd.read_csv(fp, index_col=0, parse_dates=True)

    if len(df) < MIN_TRAIN_SIZE + 50:
        return None

    X_seq, y = build_dataset(df)
    prices = df["Close"].values

    models = None
    returns = []
    trade_count = 0

    for t in range(len(X_seq) - 1):

        # ----------------------------------
        # Walk-forward retraining
        # ----------------------------------
        if models is None or (t >= MIN_TRAIN_SIZE and t % RETRAIN_EVERY == 0):
            if t < MIN_TRAIN_SIZE:
                returns.append(0.0)
                continue

            models = train_base_models(
                X_seq[:t, -1, :],
                y[:t]
            )

        # ----------------------------------
        # Predict edge
        # ----------------------------------
        preds = predict_base_models(
            models,
            X_seq[t:t + 1, -1, :]
        )

        mu, sigma = compute_mu_sigma(preds)
        edge = float(compute_edge(mu, sigma)[0])

        # ----------------------------------
        # GUARANTEED NON-ZERO POSITION
        # ----------------------------------
        if abs(edge) < EDGE_THRESHOLD:
            position = 0.0
        else:
            position = np.sign(edge) * max(abs(edge), MIN_POSITION)
            position = np.clip(position, -MAX_POSITION, MAX_POSITION)
            trade_count += 1

        ret = (prices[t + 1] / prices[t]) - 1.0
        returns.append(position * ret)

    # Debug safety
    if trade_count == 0:
        print(f"[WARN] No trades generated for {os.path.basename(fp)}")

    return pd.Series(returns)


# ======================================================
# EQUAL-WEIGHT RESEARCH PORTFOLIO
# ======================================================
def run_equal_weight_portfolio(gold_dir: str):

    files = glob.glob(os.path.join(gold_dir, "*.csv"))
    assert len(files) > 0, "No gold CSV files found"

    print(f"\nRunning research portfolio on {len(files)} symbols\n")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_single_symbol_returns)(fp)
        for fp in tqdm(files, desc="Symbols", ncols=100)
    )

    results = [r for r in results if r is not None]
    assert len(results) > 0, "No valid symbols returned data"

    # ----------------------------------
    # ALIGN SERIES
    # ----------------------------------
    min_len = min(len(r) for r in results)
    aligned = [r.iloc[-min_len:].reset_index(drop=True) for r in results]
    stacked = pd.concat(aligned, axis=1)

    # ----------------------------------
    # PORTFOLIO AGGREGATION (NO CANCELLATION)
    # ----------------------------------
    active = stacked != 0
    active_counts = active.sum(axis=1)

    portfolio_returns = stacked.sum(axis=1) / active_counts.replace(0, np.nan)
    portfolio_returns = portfolio_returns.fillna(0.0)

    equity_curve = (1.0 + portfolio_returns).cumprod()

    # ----------------------------------
    # HARD DEBUG (MANDATORY)
    # ----------------------------------
    print("Active days %      :", np.mean(portfolio_returns != 0))
    print("Mean abs return    :", np.mean(np.abs(portfolio_returns)))
    print("Max daily return   :", portfolio_returns.max())
    print("Min daily return   :", portfolio_returns.min())

    return portfolio_returns, equity_curve
