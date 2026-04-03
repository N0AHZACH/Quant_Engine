# ======================================================
# PATH FIX (CRITICAL)
# ======================================================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# IMPORTS
# ======================================================
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.model import AttentionModel, CNNModel
from src.features import add_institutional_features

# ======================================================
# PATHS
# ======================================================
DATA_FOLDER = os.path.join(PROJECT_ROOT, "lake", "gold")
INDEX_FILE = os.path.join(PROJECT_ROOT, "lake", "NSE_Complete_Dataset_2019_2025.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ======================================================
# CONFIG
# ======================================================
INITIAL_CAPITAL = 100_000.0

RISK_PER_TRADE = 0.006
MAX_TOTAL_RISK = 0.03
MAX_POSITIONS = 5

STOP_ATR_MULT = 1.4
TARGET_R_MULT = 2.2
MAX_HOLD_DAYS = 6
BROKERAGE = 20.0

SEQ_LEN = 60
MIN_JUDGE_SCORE = 0.002   # 🔒 real filter

# ======================================================
# JUDGE NETWORK
# ======================================================
class JudgeNetwork(nn.Module):
    def __init__(self):
        super(JudgeNetwork, self).__init__()
        self.layer_1 = nn.Linear(5, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.output_layer(x)

# ======================================================
# LOAD MODELS
# ======================================================
def load_models(device):
    lstm = AttentionModel("lstm").to(device)
    gru  = AttentionModel("gru").to(device)
    cnn  = CNNModel().to(device)

    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Universal_LSTM.pth"), weights_only=True))
    gru.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Universal_GRU.pth"), weights_only=True))
    cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Universal_CNN.pth"), weights_only=True))

    lstm.eval(); gru.eval(); cnn.eval()

    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(os.path.join(MODEL_DIR, "Base_XGB.json"))
    rf_m = joblib.load(os.path.join(MODEL_DIR, "Base_RF.pkl"))

    judge = JudgeNetwork().to(device)
    judge.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Judge_Network.pth"), weights_only=True))
    judge.eval()

    return lstm, gru, cnn, xgb_m, rf_m, judge

# ======================================================
# META FEATURES
# ======================================================
@torch.no_grad()
def generate_meta(X, models, device):
    lstm, gru, cnn, xgb_m, rf_m, _ = models

    X_t = torch.tensor(X).float().to(device)

    p_lstm = lstm(X_t).cpu().numpy().flatten()
    p_gru  = gru(X_t).cpu().numpy().flatten()
    p_cnn  = cnn(X_t).cpu().numpy().flatten()

    X_flat = X.reshape(len(X), -1)
    p_xgb = xgb_m.predict(X_flat)
    p_rf  = rf_m.predict(X_flat)

    return np.column_stack([p_lstm, p_gru, p_cnn, p_xgb, p_rf])

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
# ATR
# ======================================================
def compute_atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ======================================================
# MAIN
# ======================================================
def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    judge = models[-1]

    # ---------- INDEX ----------
    index_df = pd.read_csv(INDEX_FILE, parse_dates=['Date'])
    index_df.set_index('Date', inplace=True)
    index_df = index_df.groupby(index_df.index).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    index_df['SMA200'] = index_df['Close'].rolling(200).mean()
    index_df.dropna(inplace=True)
    dates = index_df.index

    # ---------- STOCK DATA ----------
    stock_data = {}
    for f in glob.glob(os.path.join(DATA_FOLDER, "*.csv")):
        try:
            df = pd.read_csv(f, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            df['ATR'] = compute_atr(df)
            feats = add_institutional_features(df)
            df = df.join(feats, rsuffix="_f").dropna()

            if len(df) > SEQ_LEN + 200:
                stock_data[os.path.basename(f).replace(".csv", "")] = df
        except:
            continue

    print(f"Loaded {len(stock_data)} stocks")

    # FIX: Explicitly define the 20 training features
    MODEL_FEATURES = [
        'Log_Ret', 'YZ_Vol', 'Z_Score', 'Vol_Imbalance', 'Vol_x_Ret', 'RSI', 'Vol_ZScore', 
        'Log_Ret_L1', 'YZ_Vol_L1', 'Z_Score_L1', 'Vol_Imbalance_L1', 'Vol_x_Ret_L1', 
        'Log_Ret_L2', 'YZ_Vol_L2', 'Z_Score_L2', 'Vol_Imbalance_L2', 'Vol_x_Ret_L2',
        'RSI_L1', 'Vol_ZScore_L1', 'Z_Score_50'
    ]

    # ---------- FIT SCALER ONCE ----------
    scaler = MinMaxScaler()
    base = np.concatenate(
        [df.iloc[:SEQ_LEN][MODEL_FEATURES].values for df in stock_data.values()],
        axis=0
    )
    scaler.fit(base)

    cash = INITIAL_CAPITAL
    trades = []
    equity = []

    # ==================================================
    for d in tqdm(dates, desc="Judge Simulation"):

        # ----- EXITS -----
        remaining = []
        for t in trades:
            df = stock_data[t.ticker]
            if d not in df.index:
                remaining.append(t)
                continue

            row = df.loc[d]
            exit_price = None

            if row['Low'] <= t.stop:
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

        # ----- NAV -----
        nav = cash + sum(
            stock_data[t.ticker].loc[d, 'Close'] * t.qty
            for t in trades if d in stock_data[t.ticker].index
        )
        equity.append(nav)

        # ----- REGIME -----
        if index_df.loc[d, 'Close'] < index_df.loc[d, 'SMA200']:
            continue

        # ----- BUILD JUDGE INPUT -----
        X_seq, rows = [], []
        for ticker, df in stock_data.items():
            if d not in df.index:
                continue
            idx = df.index.get_loc(d)
            if idx < SEQ_LEN:
                continue
            X_seq.append(df.iloc[idx-SEQ_LEN:idx][MODEL_FEATURES].values)
            rows.append((ticker, df.loc[d]))

        if not X_seq:
            continue

        X_seq = scaler.transform(
            np.array(X_seq).reshape(-1, len(MODEL_FEATURES))
        ).reshape(len(X_seq), SEQ_LEN, len(MODEL_FEATURES))

        meta = generate_meta(X_seq, models, device)
        scores = judge(torch.tensor(meta).float().to(device)).cpu().numpy().flatten()

        ranked = sorted(
            [(s, r) for s, r in zip(scores, rows) if s > MIN_JUDGE_SCORE],
            reverse=True
        )

        # ----- ENTRIES -----
        for score, (ticker, row) in ranked:
            if len(trades) >= MAX_POSITIONS:
                break

            risk_ps = STOP_ATR_MULT * row['ATR']
            qty = int((nav * RISK_PER_TRADE) / risk_ps)
            if qty <= 0:
                continue

            cost = row['Open'] * qty
            if cost > cash:
                continue

            stop = row['Open'] - STOP_ATR_MULT * row['ATR']
            cash -= cost
            trades.append(Trade(ticker, row['Open'], stop, qty))

    # ---------- RESULTS ----------
    equity = pd.Series(equity)
    dd = equity / equity.cummax() - 1
    years = len(equity) / 252
    cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    print("\n==============================")
    print("JUDGE SWING SYSTEM (FIXED)")
    print("==============================")
    print(f"Final Equity: ₹{equity.iloc[-1]:,.2f}")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Max DD:       {dd.min():.2%}")
    print("==============================")

# ======================================================
if __name__ == "__main__":
    run()
