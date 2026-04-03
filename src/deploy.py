import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import joblib
import xgboost as xgb
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from model import AttentionModel, CNNModel 
from features import add_institutional_features 

# ======================================================
# 🌙 NIGHT BEFORE (AMO) CONFIGURATION
# ======================================================

GOLD_FOLDER = "lake/gold"
LSTM_PATH = "models/Universal_LSTM.pth"
GRU_PATH = "models/Universal_GRU.pth"
CNN_PATH = "models/Universal_CNN.pth"
XGB_PATH = "models/Base_XGB.json"
RF_PATH = "models/Base_RF.pkl"

# --- ⚡ QUALITY FILTERS (No Penny Stocks) ---
MIN_PRICE = 50.0                # Exclude stocks below ₹50 (Penny Stocks)
MIN_AVG_VOLUME = 50000          # Exclude illiquid stocks (<50k avg daily vol)

# --- ⚡ STRATEGY FILTERS ---
PREDICTION_HORIZON = 10 
CONFIDENCE_THRESHOLD = 0.0010 
SMA_FILTER = 20                 # Trend must be active
RSI_MIN = 45                   
RSI_MAX = 80                   

# --- 🎯 EXIT SETTINGS ---
STOP_ATR_MULT = 3.0
PROFIT_ATR_MULT = 8.0

# --- 💰 POSITION SIZING ---
TOTAL_CAPITAL = 10000.0   
TARGET_ANNUAL_VOL = 0.50 
MAX_SINGLE_ALLOCATION = 0.25 
MAX_LEVERAGE = 1.2 

# ======================================================
# MODELS & LOGIC
# ======================================================

class JudgeNetwork(nn.Module):
    def __init__(self, input_dim=5):
        super(JudgeNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x) 
        x = self.relu(x)
        return self.output_layer(x)

def load_universal_models(device):
    if not os.path.exists(LSTM_PATH):
        print("❌ CRITICAL: Universal models not found.")
        sys.exit()
    
    m_lstm = AttentionModel("lstm").to(device); m_lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True)); m_lstm.eval()
    m_gru = AttentionModel("gru").to(device); m_gru.load_state_dict(torch.load(GRU_PATH, weights_only=True)); m_gru.eval()
    m_cnn = CNNModel().to(device); m_cnn.load_state_dict(torch.load(CNN_PATH, weights_only=True)); m_cnn.eval()
    xgb_m = xgb.XGBRegressor(); xgb_m.load_model(XGB_PATH)
    rf_m = joblib.load(RF_PATH)
    return m_lstm, m_gru, m_cnn, xgb_m, rf_m

def generate_meta_features(X_sequence, universal_models, device):
    m_lstm, m_gru, m_cnn, xgb_m, rf_m = universal_models
    if isinstance(X_sequence, torch.Tensor): X_sequence = X_sequence.cpu().numpy()
    meta_dl_list = []
    
    BATCH_SIZE = 1024
    with torch.no_grad():
        for i in range(0, len(X_sequence), BATCH_SIZE):
            batch = torch.from_numpy(X_sequence[i:i+BATCH_SIZE]).float().to(device)
            p1 = m_lstm(batch).cpu().numpy().flatten()
            p2 = m_gru(batch).cpu().numpy().flatten()
            p3 = m_cnn(batch).cpu().numpy().flatten()
            meta_dl_list.append(np.stack([p1, p2, p3], axis=1))
            
    if not meta_dl_list: return np.array([])
    meta_dl = np.concatenate(meta_dl_list, axis=0)
    
    X_tree = X_sequence.reshape(X_sequence.shape[0], -1)
    p_xgb = xgb_m.predict(X_tree)
    p_rf = rf_m.predict(X_tree)
    
    return np.column_stack([meta_dl, p_xgb.reshape(-1, 1), p_rf.reshape(-1, 1)])

def generate_live_signals(device):
    print("\n--- 🌙 GENERATING AMO SIGNALS (NO PENNY STOCKS) ---")
    
    models = load_universal_models(device)
    files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    
    candidates = []
    print(f"Scanning {len(files)} stocks from Today's Close...")
    
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            if len(df) < 200: continue
            
            last_idx = -1 
            
            # --- QUALITY FILTERS (Penny Stock Check) ---
            current_close = df['Close'].iloc[last_idx]
            avg_volume = df['Volume'].rolling(20).mean().iloc[last_idx]
            
            if current_close < MIN_PRICE: continue          # Reject < ₹50
            if avg_volume < MIN_AVG_VOLUME: continue        # Reject Illiquid
            
            # --- STRATEGY INDICATORS ---
            df['SMA'] = df['Close'].rolling(SMA_FILTER).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['Vol_Spike'] = (df['Volume'] > df['Vol_SMA']).astype(int)
            df['TR'] = (df['High'] - df['Low']) 
            df['ATR_Pct'] = (df['TR'].rolling(14).mean() / df['Close'])
            
            # Strategy Check
            current_sma = df['SMA'].iloc[last_idx]
            current_rsi = df['RSI'].iloc[last_idx]
            current_vol_spike = df['Vol_Spike'].iloc[last_idx]
            
            if current_close < current_sma: continue          
            if current_rsi < RSI_MIN or current_rsi > RSI_MAX: continue 
            if current_vol_spike == 0: continue               
            
            # Features
            df_feats = add_institutional_features(df.copy())
            SEQ_LEN = 60
            if len(df_feats) < SEQ_LEN: continue
            
            raw_seq = df_feats.values[-SEQ_LEN:].astype(np.float32)
            scaler = MinMaxScaler()
            scaled_seq = scaler.fit_transform(raw_seq)
            
            clean_ticker = os.path.basename(f).replace('.csv','').replace('_processed','')

            candidates.append({
                'ticker': clean_ticker,
                'close': current_close,
                'atr_pct': df['ATR_Pct'].iloc[last_idx],
                'X': scaled_seq[np.newaxis, :, :] 
            })
            
        except Exception: continue
            
    if not candidates:
        print("❌ No stocks passed the filters.")
        return

    print(f"✅ {len(candidates)} quality stocks found. Calculating Allocation...")
    
    X_batch = np.concatenate([c['X'] for c in candidates], axis=0)
    meta_feats = generate_meta_features(X_batch, models, device)
    consensus_scores = np.mean(meta_feats, axis=1)
    ranked_indices = np.argsort(consensus_scores)[::-1]
    
    exposure = 0.0
    
    print("\n" + "="*95)
    print(f"🌙 AMO ORDERS (For Tomorrow Morning) | CAPITAL: ₹{TOTAL_CAPITAL:,.0f}")
    print("="*95)
    print(f"{'TICKER':<15} {'BUY LIMIT':<10} {'STOP LOSS':<10} {'TARGET':<10} {'ATR%':<7} {'SCORE':<6} {'ALLOCATION'}")
    print("-" * 95)
    
    for i in ranked_indices:
        c = candidates[i]
        score = consensus_scores[i]
        
        if score < CONFIDENCE_THRESHOLD: continue
        
        stock_atr_pct = max(c['atr_pct'], 0.01)
        daily_target_vol = TARGET_ANNUAL_VOL / np.sqrt(252)
        allocation_pct = min(daily_target_vol / stock_atr_pct, MAX_SINGLE_ALLOCATION)
        
        if exposure + allocation_pct > MAX_LEVERAGE:
            allocation_pct = MAX_LEVERAGE - exposure
        if allocation_pct <= 0: break
        
        rupees = TOTAL_CAPITAL * allocation_pct
        exposure += allocation_pct
        
        entry_price = c['close']
        buy_limit = entry_price * 1.005 # 0.5% buffer for AMO
        
        sl_price = entry_price * (1 - (stock_atr_pct * STOP_ATR_MULT))
        tp_price = entry_price * (1 + (stock_atr_pct * PROFIT_ATR_MULT))
        pred_return_pct = score * 100 
        
        print(f"{c['ticker']:<15} {buy_limit:<10.2f} {sl_price:<10.2f} {tp_price:<10.2f} {stock_atr_pct*100:<6.1f}% {pred_return_pct:<6.2f} ₹{rupees:,.0f}")
        
    print("="*95)
    print(f"Total Allocation: {exposure*100:.1f}%")
    print("NOTE: Place 'AMO LIMIT' orders tonight using the 'BUY LIMIT' price.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    generate_live_signals(DEVICE)