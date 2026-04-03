import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "..", "lake", "gold")
INDEX_FILE = os.path.join(BASE_DIR, "..", "lake", "NSE_Complete_Dataset_2019_2025.csv")

# ======================================================
# CONFIG (SEMI-AGGRESSIVE, CONTROLLED)
# ======================================================
INITIAL_CAPITAL = 100_000.0

RISK_PER_TRADE = 0.006        # 🔼 0.6% per trade
MAX_TOTAL_RISK = 0.03         # 🔼 3% portfolio risk
MAX_POSITIONS = 5             # 🔼 more capital deployed

STOP_ATR_MULT = 1.4
TARGET_R_MULT = 2.2
MAX_HOLD_DAYS = 6
BROKERAGE = 20.0

MIN_DOLLAR_VOL = 5e7          # liquidity filter
MIN_MOMENTUM = 0.04           # slightly looser momentum

# ======================================================
# INDICATORS
# ======================================================
def compute_atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def prepare_stock(df):
    df['ATR'] = compute_atr(df)
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['Momentum'] = df['Close'].pct_change(5)
    df['DollarVolume'] = df['Close'] * df['Volume']

    df['Signal'] = (
        (df['Close'].shift(1) > df['SMA50'].shift(1)) &
        (df['Momentum'].shift(1) > MIN_MOMENTUM)
    )
    return df.dropna()

# ======================================================
# TRADE OBJECT
# ======================================================
class Trade:
    def __init__(self, ticker, entry, stop, qty):
        self.ticker = ticker
        self.entry = entry
        self.stop = stop
        self.qty = qty
        self.days = 0

# ======================================================
# BACKTEST
# ======================================================
def run_backtest():

    # -----------------------------
    # LOAD INDEX (MASTER CALENDAR)
    # -----------------------------
    index_df = pd.read_csv(INDEX_FILE, parse_dates=['Date'])
    index_df.set_index('Date', inplace=True)
    index_df.sort_index(inplace=True)

    # Ensure one row per day
    index_df = index_df.groupby(index_df.index).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    index_df['SMA200'] = index_df['Close'].rolling(200).mean()
    index_df['Vol20'] = index_df['Close'].pct_change().rolling(20).std()
    index_df['Vol100'] = index_df['Close'].pct_change().rolling(100).std()
    index_df.dropna(inplace=True)

    dates = index_df.index

    # -----------------------------
    # LOAD STOCKS (LIQUID ONLY)
    # -----------------------------
    stock_data = {}
    for f in glob.glob(os.path.join(DATA_FOLDER, "*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df = prepare_stock(df)

            if (
                len(df) >= 200 and
                df['DollarVolume'].rolling(20).mean().iloc[-1] > MIN_DOLLAR_VOL
            ):
                ticker = os.path.basename(f).replace(".csv", "")
                stock_data[ticker] = df
        except:
            continue

    print(f"Loaded {len(stock_data)} liquid stocks")

    cash = INITIAL_CAPITAL
    trades = []
    equity_curve = []

    # ==================================================
    # MAIN LOOP
    # ==================================================
    for d in tqdm(dates, desc="Simulating"):

        # -------- EXITS --------
        remaining = []
        for t in trades:
            df = stock_data.get(t.ticker)
            if df is None or d not in df.index:
                remaining.append(t)
                continue

            row = df.loc[d]
            exit_price = None

            if row['Open'] <= t.stop:
                exit_price = row['Open']
            elif row['Low'] <= t.stop:
                exit_price = t.stop

            risk = t.entry - t.stop
            target = t.entry + TARGET_R_MULT * risk
            if exit_price is None and row['High'] >= target:
                exit_price = target

            t.days += 1
            if exit_price is None and t.days >= MAX_HOLD_DAYS:
                exit_price = row['Close']

            if exit_price:
                cash += exit_price * t.qty - BROKERAGE * 2
            else:
                remaining.append(t)

        trades = remaining

        # -------- NAV --------
        nav = cash
        for t in trades:
            df = stock_data[t.ticker]
            if d in df.index:
                nav += df.loc[d, 'Close'] * t.qty

        equity_curve.append({'Date': d, 'Equity': nav})

        # -------- MARKET REGIME --------
        if index_df.loc[d, 'Close'] < index_df.loc[d, 'SMA200']:
            continue

        # 🔓 looser volatility filter
        if index_df.loc[d, 'Vol20'] > index_df.loc[d, 'Vol100'] * 1.6:
            continue

        # -------- PORTFOLIO RISK --------
        current_risk = sum((t.entry - t.stop) * t.qty for t in trades)
        if current_risk >= nav * MAX_TOTAL_RISK:
            continue

        # -------- SCAN & RANK --------
        candidates = []
        for ticker, df in stock_data.items():
            if d not in df.index:
                continue
            row = df.loc[d]
            if row['Signal'] and row['ATR'] > 0:
                score = row['Momentum'] / row['ATR']
                candidates.append((score, ticker, row))

        candidates.sort(reverse=True)

        # -------- ENTRIES --------
        for _, ticker, row in candidates:
            if len(trades) >= MAX_POSITIONS:
                break

            risk_per_share = STOP_ATR_MULT * row['ATR']
            qty = int((nav * RISK_PER_TRADE) / risk_per_share)
            if qty <= 0:
                continue

            cost = row['Open'] * qty
            if cost > cash:
                continue

            stop = row['Open'] - STOP_ATR_MULT * row['ATR']
            cash -= cost
            trades.append(Trade(ticker, row['Open'], stop, qty))

    # ==================================================
    # RESULTS
    # ==================================================
    eq = pd.DataFrame(equity_curve).set_index('Date')
    dd = eq['Equity'] / eq['Equity'].cummax() - 1

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq['Equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    print("\n==============================")
    print("LIVE SWING SYSTEM (TUNED)")
    print("==============================")
    print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:    ₹{eq['Equity'].iloc[-1]:,.2f}")
    print(f"CAGR:            {cagr:.2%}")
    print(f"Max Drawdown:    {dd.min():.2%}")
    print("==============================")

# ======================================================
if __name__ == "__main__":
    run_backtest()
