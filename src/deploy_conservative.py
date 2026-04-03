import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import joblib
import xgboost as xgb
import sys
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tabulate import tabulate 
from model import AttentionModel, CNNModel 
from features import add_institutional_features 

# --- CONFIGURATION ---
GOLD_FOLDER = "lake/gold"
LSTM_PATH = "models/Universal_LSTM.pth"
GRU_PATH = "models/Universal_GRU.pth"
CNN_PATH = "models/Universal_CNN.pth"
XGB_PATH = "models/Base_XGB.json"
RF_PATH = "models/Base_RF.pkl"

# --- 💰 MONEY SETTINGS 💰 ---
TOTAL_CAPITAL = 10000.0  # <--- CHANGE THIS TO YOUR ACTUAL CAPITAL (e.g., 500000.0)

# --- CONSERVATIVE SETTINGS ---
TARGET_ANNUAL_VOL = 0.30      
MAX_POSITIONS = 5             
MAX_ALLOCATION = 0.20         
CONFIDENCE_THRESHOLD = 0.003  
SMA_FILTER = 200              

# Exits
STOP_ATR_MULT = 2.5    
PROFIT_ATR_MULT = 5.0  

# Training
SEQ_LEN = 60
TRAIN_WINDOW = 756
BATCH_SIZE = 1024
JUDGE_EPOCHS = 15

# --- MODELS ---
class JudgeNetwork(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def load_universal_models(device):
    if not os.path.exists(LSTM_PATH):
        print(f"❌ ERROR: Models missing.")
        sys.exit()
    try:
        m_lstm = AttentionModel("lstm").to(device); m_lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True)); m_lstm.eval()
        m_gru = AttentionModel("gru").to(device); m_gru.load_state_dict(torch.load(GRU_PATH, weights_only=True)); m_gru.eval()
        m_cnn = CNNModel().to(device); m_cnn.load_state_dict(torch.load(CNN_PATH, weights_only=True)); m_cnn.eval()
        xgb_m = xgb.XGBRegressor(); xgb_m.load_model(XGB_PATH)
        rf_m = joblib.load(RF_PATH)
        return m_lstm, m_gru, m_cnn, xgb_m, rf_m
    except Exception: sys.exit()

def generate_meta_features(X_arr, models, device):
    m_lstm, m_gru, m_cnn, xgb_m, rf_m = models
    meta = []
    with torch.no_grad():
        for i in range(0, len(X_arr), 1024):
            batch = torch.from_numpy(X_arr[i:i+1024]).float().to(device)
            p1 = m_lstm(batch).cpu().numpy().flatten()
            p2 = m_gru(batch).cpu().numpy().flatten()
            p3 = m_cnn(batch).cpu().numpy().flatten()
            meta.append(np.stack([p1, p2, p3], axis=1))
    if not meta: return np.array([])
    meta = np.concatenate(meta, axis=0)
    X_flat = X_arr.reshape(len(X_arr), -1)
    p4 = xgb_m.predict(X_flat)
    p5 = rf_m.predict(X_flat)
    return np.column_stack([meta, p4, p5])

def train_judge(X_meta, y_true, device):
    model = JudgeNetwork(X_meta.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X_meta).float().to(device)
    y_t = torch.from_numpy(y_true).float().to(device)
    model.train()
    for _ in range(JUDGE_EPOCHS):
        opt.zero_grad()
        loss_fn(model(X_t).squeeze(), y_t).backward()
        opt.step()
    model.eval()
    return model

def run_live(device):
    print("\n--- 🛡️ CONSERVATIVE SNIPER DEPLOYMENT 🛡️ ---")
    
    models = load_universal_models(device)
    files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    if not files: print("❌ No Data."); return
    
    stock_cache = []
    print(f"1. Scanning {len(files)} stocks (Filter: >{SMA_FILTER} SMA)...")
    
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            if len(df) < 500: continue
            
            df['target'] = df['Close'].pct_change(3).shift(-3).fillna(0)
            df_feats = add_institutional_features(df.copy())
            
            atr = (df['High'] - df['Low']).rolling(20).mean()
            sma = df['Close'].rolling(SMA_FILTER).mean()
            
            stock_cache.append({
                'ticker': os.path.basename(f).replace(".csv",""),
                'raw_feats': df_feats.values.astype(np.float32),
                'target': df['target'].values,
                'close': df['Close'].values,
                'atr_val': atr.values,
                'sma_filter': sma.values,
                'dates': df.index
            })
        except: continue

    if not stock_cache: print("❌ No valid data."); return

    print("2. Calibrating AI Risk Model...", end=" ")
    X_train_raw = []
    train_subset = random.sample(stock_cache, min(len(stock_cache), 250))
    for s in train_subset:
        if len(s['raw_feats']) > SEQ_LEN + TRAIN_WINDOW:
            X_train_raw.append(s['raw_feats'][-TRAIN_WINDOW:])
            
    if not X_train_raw: print("❌ Train Error."); return

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(X_train_raw, axis=0))
    
    X_seq_train, y_seq_train = [], []
    for s in train_subset:
        if len(s['raw_feats']) > SEQ_LEN + TRAIN_WINDOW:
            chunk = scaler.transform(s['raw_feats'][-TRAIN_WINDOW:])
            indices = np.linspace(SEQ_LEN, len(chunk)-1, 10, dtype=int)
            for i in indices:
                X_seq_train.append(chunk[i-SEQ_LEN:i])
                y_seq_train.append(s['target'][-TRAIN_WINDOW + i])

    X_meta = generate_meta_features(np.array(X_seq_train), models, device)
    judge = train_judge(X_meta, np.array(y_seq_train), device)
    print("✅ Done")
    
    candidates = []
    
    for s in stock_cache:
        latest_raw = s['raw_feats'][-SEQ_LEN:]
        latest_atr = s['atr_val'][-1]
        latest_close = s['close'][-1]
        latest_sma = s['sma_filter'][-1]
        
        if latest_close < latest_sma: continue 

        latest_scaled = scaler.transform(latest_raw.reshape(1, -1) if latest_raw.ndim==1 else latest_raw)
        seq_input = np.array([latest_scaled])
        meta_input = generate_meta_features(seq_input, models, device)
        
        with torch.no_grad():
            pred = judge(torch.from_numpy(meta_input).float().to(device)).item()
            
        if pred < CONFIDENCE_THRESHOLD: continue
            
        score = pred / ((latest_atr/latest_close) + 1e-9)
        
        candidates.append({
            'Ticker': s['ticker'], 'Close': latest_close, 'ATR': latest_atr,
            'Pred': pred, 'Score': score
        })
        
    df_picks = pd.DataFrame(candidates).sort_values('Score', ascending=False).head(MAX_POSITIONS)
    
    if df_picks.empty:
        print("\n⚠️ NO CONSERVATIVE TRADES FOUND.")
    else:
        results = []
        daily_vol_target = TARGET_ANNUAL_VOL / 16.0
        
        for _, row in df_picks.iterrows():
            vol_pct = row['ATR'] / row['Close']
            alloc = min(daily_vol_target / max(vol_pct, 0.01), MAX_ALLOCATION)
            
            # --- CALCULATE INR AMOUNT ---
            inr_amount = alloc * TOTAL_CAPITAL
            shares = int(inr_amount / row['Close'])
            
            stop_price = row['Close'] - (row['ATR'] * STOP_ATR_MULT)
            target_price = row['Close'] + (row['ATR'] * PROFIT_ATR_MULT)
            
            results.append({
                'Ticker': row['Ticker'],
                'Action': 'BUY',
                'Alloc (INR)': f"₹{inr_amount:,.0f}",
                'Qty': shares,
                'Entry': round(row['Close'], 2),
                'SL': round(stop_price, 2),
                'TP': round(target_price, 2)
            })

        print("\n" + "="*80)
        print(f"📢 CONSERVATIVE SIGNALS (Capital: ₹{TOTAL_CAPITAL:,.0f})")
        print("="*80)
        print(tabulate(results, headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_live(device)