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
# RESEARCH CONFIG (PAPER MODE)
# ======================================================
RETRAIN_EVERY = 200
MIN_TRAIN_SIZE = 300

LONG_Q = 0.9     # top decile
SHORT_Q = 0.1    # bottom decile

N_JOBS = -1


# ======================================================
# SINGLE SYMBOL EDGE + RETURN SERIES
# ======================================================
def run_single_symbol_edge_series(fp: str):

    df = pd.read_csv(fp, index_col=0, parse_dates=True)

    if len(df) < MIN_TRAIN_SIZE + 50:
        return None

    X_seq, y = build_dataset(df)
    prices = df["Close"].values

    models = None
    edges = []
    rets = []

    for t in range(len(X_seq) - 1):

        # Walk-forward retraining
        if models is None or (t >= MIN_TRAIN_SIZE and t % RETRAIN_EVERY == 0):
            if t < MIN_TRAIN_SIZE:
                edges.append(0.0)
                rets.append(0.0)
                continue

            models = train_base_models(
                X_seq[:t, -1, :],
                y[:t]
            )

        preds = predict_base_models(
            models,
            X_seq[t:t + 1, -1, :]
        )

        mu, sigma = compute_mu_sigma(preds)
        edge = float(compute_edge(mu, sigma)[0])

        ret = (prices[t + 1] / prices[t]) - 1.0

        edges.append(edge)
        rets.append(ret)

    return pd.DataFrame({
        "edge": edges,
        "ret": rets
    })


# ======================================================
# CROSS-SECTIONAL PORTFOLIO
# ======================================================
def run_cross_sectional_portfolio(gold_dir: str):

    files = glob.glob(os.path.join(gold_dir, "*.csv"))
    assert len(files) > 0, "No gold CSV files found"

    print(f"\nRunning CROSS-SECTIONAL research portfolio on {len(files)} symbols\n")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_single_symbol_edge_series)(fp)
        for fp in tqdm(files, desc="Symbols", ncols=100)
    )

    results = [r for r in results if r is not None]
    assert len(results) > 0, "No valid symbols processed"

    # --------------------------------------------------
    # ALIGN TIME AXIS
    # --------------------------------------------------
    min_len = min(len(r) for r in results)
    aligned = [r.iloc[-min_len:].reset_index(drop=True) for r in results]

    stacked = pd.concat(aligned, axis=1, keys=range(len(aligned)))

    edge_df = stacked.xs("edge", level=1, axis=1)
    ret_df  = stacked.xs("ret",  level=1, axis=1)

    # --------------------------------------------------
    # CROSS-SECTIONAL RANKING
    # --------------------------------------------------
    rank = edge_df.rank(axis=1, pct=True)

    positions = np.where(
        rank > LONG_Q,  1,
        np.where(rank < SHORT_Q, -1, 0)
    )

    # --------------------------------------------------
    # PORTFOLIO RETURNS (MARKET-NEUTRAL)
    # --------------------------------------------------
    portfolio_returns = (positions * ret_df.values).mean(axis=1)
    equity_curve = (1.0 + portfolio_returns).cumprod()

    # --------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------
    print("Avg positions/day :", np.mean(np.abs(positions).sum(axis=1)))
    print("Active days %     :", np.mean(portfolio_returns != 0))
    print("Mean abs return   :", np.mean(np.abs(portfolio_returns)))

    return portfolio_returns, equity_curve
