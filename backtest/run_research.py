import numpy as np
import matplotlib.pyplot as plt

from backtest.run_portfolio_cross_sectional import (
    run_cross_sectional_portfolio
)

def main():
    gold_dir = "lake/gold"

    returns, equity = run_cross_sectional_portfolio(gold_dir)

    print("\n===== CROSS-SECTIONAL RESEARCH RESULTS =====")
    print("Total Return :", equity.iloc[-1] - 1.0)
    print("Sharpe       :", returns.mean() / (returns.std() + 1e-9))
    print("Max Drawdown :", (equity / equity.cummax() - 1).min())
    print("Win Rate     :", (returns > 0).mean())

    plt.figure(figsize=(10, 5))
    plt.plot(equity.values)
    plt.title("Cross-Sectional Equity Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
