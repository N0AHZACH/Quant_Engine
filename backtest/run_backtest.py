import numpy as np
import pandas as pd
from tqdm import tqdm

from production.dataset_builder import build_dataset
from backtest.run_portfolio_equal_weight import run_equal_weight_portfolio
from production.performance import compute_metrics
import matplotlib.pyplot as plt



FIXED_NOTIONAL = 200_000


def run_backtest(df: pd.DataFrame, initial_capital: float = 1_000_000):
    capital = initial_capital
    equity_curve = []

    prices = df["Close"].values

    for t in tqdm(range(len(prices) - 1), desc="Backtest", ncols=100):

        # --------------------------------------------------
        # FORCE LONG POSITION (NO MODEL)
        # --------------------------------------------------
        notional = FIXED_NOTIONAL

        # --------------------------------------------------
        # RETURN-BASED PnL
        # --------------------------------------------------
        ret = (prices[t + 1] / prices[t]) - 1.0
        pnl = notional * ret

        capital += pnl
        equity_curve.append(capital)

    return pd.Series(equity_curve)
