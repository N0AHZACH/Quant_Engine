import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series):
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-12)

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    print("\nCapital start:", equity.iloc[0])
    print("Capital end  :", equity.iloc[-1])
    print("Capital diff :", equity.iloc[-1] - equity.iloc[0])

    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
    }
