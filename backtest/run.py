import pandas as pd
import matplotlib.pyplot as plt

from backtest.run_backtest import run_backtest
from production.performance import compute_metrics


if __name__ == "__main__":
    # PICK ONE SYMBOL (IMPORTANT)
    df = pd.read_csv(
        "lake/gold/AARTIIND_processed.csv",  # <- ONE FILE ONLY
        index_col=0,
        parse_dates=True,
    )

    equity = run_backtest(df, initial_capital=1_000_000)
    metrics = compute_metrics(equity)

    print("\n===== BACKTEST RESULTS (SINGLE SYMBOL) =====")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")

    equity.plot(title="Equity Curve — Single Symbol")
    plt.show()
