import torch
import torch.nn as nn
import torch.optim as optim
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
from model import AttentionModel, CNNModel 
from features import add_institutional_features 

# ======================================================
# 🇮🇳 NSE "RATCHET" CONFIGURATION (Target: Consistent 40%)
# ======================================================

GOLD_FOLDER = "lake/gold"
LSTM_PATH = "models/Universal_LSTM.pth"
GRU_PATH = "models/Universal_GRU.pth"
CNN_PATH = "models/Universal_CNN.pth"
XGB_PATH = "models/Base_XGB.json"
RF_PATH = "models/Base_RF.pkl"

# --- 💰 CAPITAL ---
TOTAL_CAPITAL = 10000.0   
PRINCIPAL = TOTAL_CAPITAL

# --- 🔧 THE RATCHET STRATEGY ---
# 1. TIMEFRAME
PREDICTION_HORIZON = 10 

# 2. SIZING (Aggressive but Controlled)
TARGET_ANNUAL_VOL = 0.50      

# 3. FILTERS (Momentum Lock)
CONFIDENCE_THRESHOLD = 0.0010  
SMA_FILTER = 20                # Faster entry
RSI_MIN = 45                   
RSI_MAX = 80                   

# 4. RATCHET EXITS (The Secret Sauce)
#    We don't just wait. We trail our stop.
INITIAL_STOP_ATR = 2.0         # Start with 2x risk
RATCHET_TRIGGER = 1.0          # If price moves +1 ATR...
RATCHET_LOCK = 0.0             # ...Move Stop to Breakeven (0.0)
PROFIT_ATR_MULT = 6.0          # Ultimate Target

# 5. LEVERAGE
MAX_POSITIONS_PER_DAY = 6     
MAX_LEVERAGE = 1.2             # Mild Leverage (20% Margin)
MAX_SINGLE_ALLOCATION = 0.25   

# --- 🛑 FRICTION (0.2%) ---
TOTAL_FRICTION = 0.0020  

# --- TRAINING ---
SEQ_LEN = 60
TRAIN_WINDOW = 756  
TEST_WINDOW = 63    
JUDGE_EPOCHS = 20   
JUDGE_LR = 0.001
BATCH_SIZE = 1024

# ======================================================
# MODELS
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
    try:
        m_lstm = AttentionModel("lstm").to(device); m_lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True)); m_lstm.eval()
        m_gru = AttentionModel("gru").to(device); m_gru.load_state_dict(torch.load(GRU_PATH, weights_only=True)); m_gru.eval()
        m_cnn = CNNModel().to(device); m_cnn.load_state_dict(torch.load(CNN_PATH, weights_only=True)); m_cnn.eval()
        xgb_m = xgb.XGBRegressor(); xgb_m.load_model(XGB_PATH)
        rf_m = joblib.load(RF_PATH)
        return m_lstm, m_gru, m_cnn, xgb_m, rf_m
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit()

def generate_meta_features(X_sequence, universal_models, device):
    m_lstm, m_gru, m_cnn, xgb_m, rf_m = universal_models
    if isinstance(X_sequence, torch.Tensor): X_sequence = X_sequence.cpu().numpy()
    meta_dl_list = []
    
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

def train_judge_on_data(X_meta, y_true, device):
    torch.manual_seed(42)
    judge = JudgeNetwork(X_meta.shape[1]).to(device)
    opt = optim.AdamW(judge.parameters(), lr=JUDGE_LR)
    loss_fn = nn.MSELoss()
    
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_meta).float(), torch.from_numpy(y_true).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    
    judge.train()
    for _ in range(JUDGE_EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(judge(xb.to(device)).squeeze(), yb.to(device)).backward()
            opt.step()
    judge.eval()
    return judge

def print_stats(rets):
    eq = [PRINCIPAL]
    for r in rets: eq.append(eq[-1] * (1 + r))
    eq = pd.Series(eq)
    
    total_years = len(rets) / 252.0
    if total_years < 0.1: total_years = 0.1
    
    cagr = (eq.iloc[-1]/PRINCIPAL)**(1/total_years) - 1 if len(rets)>0 else 0
    dd = (eq/eq.cummax() - 1).min()
    
    print(f"\n📊 PHASE 7: RATCHET STRATEGY RESULTS")
    print(f"CAGR:        {cagr:.2%}")
    print(f"Max DD:      {dd:.2%}")
    print(f"Final Equity: ₹{eq.iloc[-1]:,.2f}")

def run_walk_forward(device):
    print("\n--- PHASE 7: RATCHET STRATEGY (Trailing Stops) ---")
    random.seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    models = load_universal_models(device)
    files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    files.sort(); random.shuffle(files)
    
    stock_cache = []
    print(f"Loading {len(files)} stocks...")
    
    for f in tqdm(files):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            if len(df) < 500: continue
            
            # 10 Day Hold
            df['Target'] = df['Close'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON).fillna(0)
            
            # Indicators
            df['SMA'] = df['Close'].rolling(SMA_FILTER).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['Vol_Spike'] = (df['Volume'] > df['Vol_SMA']).astype(int)
            
            df['Next_Open'] = df['Open'].shift(-1)
            
            df['TR'] = (df['High'] - df['Low']) 
            df['ATR'] = df['TR'].rolling(14).mean() / df['Close']
            
            # --- RATCHET SIMULATION DATA ---
            # We need to know the 'High' over the holding period to check if we triggered the ratchet
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PREDICTION_HORIZON)
            df['Max_Gain_In_Hold'] = (df['High'].rolling(window=indexer).max() - df['Next_Open']) / df['Next_Open']
            
            df.dropna(inplace=True)
            df_feats = add_institutional_features(df.copy())
            
            stock_cache.append({
                'ticker': os.path.basename(f),
                'dates': df.index,
                'X': df_feats.values.astype(np.float32),
                'y': df['Target'].values,
                'atr': df['ATR'].values,
                'close': df['Close'].values,
                'next_open': df['Next_Open'].values,
                'max_gain': df['Max_Gain_In_Hold'].values, # For Ratchet
                'rsi': df['RSI'].values,
                'sma': df['SMA'].values,
                'vol_spike': df['Vol_Spike'].values
            })
        except: continue
        
    all_dates = sorted(list(set([d for s in stock_cache for d in s['dates']])))
    start = SEQ_LEN + TRAIN_WINDOW
    rets = []
    
    for t in tqdm(range(start, len(all_dates)-TEST_WINDOW, TEST_WINDOW), desc="Backtesting"):
        
        # Train
        Xtr, ytr = [], []
        subset = random.sample(stock_cache, min(len(stock_cache), 200))
        for s in subset:
            d_idx = s['dates'].get_loc(all_dates[t]) if all_dates[t] in s['dates'] else -1
            if d_idx > SEQ_LEN + TRAIN_WINDOW:
                chunk = s['X'][d_idx-TRAIN_WINDOW:d_idx]
                sc = MinMaxScaler()
                chunk = sc.fit_transform(chunk)
                for i in range(SEQ_LEN, len(chunk), 10):
                    Xtr.append(chunk[i-SEQ_LEN:i])
                    ytr.append(s['y'][d_idx-TRAIN_WINDOW+i])
        if not Xtr: continue
        judge = train_judge_on_data(generate_meta_features(np.array(Xtr), models, device), np.array(ytr), device)
        
        # Test
        for d in all_dates[t:t+TEST_WINDOW]:
            candidates = []
            for s in stock_cache:
                if d not in s['dates']: continue
                idx = s['dates'].get_loc(d)
                if idx < SEQ_LEN: continue
                
                # Filters
                if s['close'][idx] < s['sma'][idx]: continue       
                if s['rsi'][idx] < RSI_MIN or s['rsi'][idx] > RSI_MAX: continue 
                if s['vol_spike'][idx] == 0: continue              
                
                raw = s['X'][idx-SEQ_LEN:idx]
                sc = MinMaxScaler().fit(raw)
                candidates.append({'s': s, 'idx': idx, 'X': sc.transform(raw)})
                
            if not candidates: rets.append(0.0); continue
            
            X_test = np.array([c['X'] for c in candidates])
            meta = generate_meta_features(X_test, models, device)
            with torch.no_grad():
                preds = judge(torch.from_numpy(meta).float().to(device)).cpu().numpy().flatten()
            
            day_pnl = 0.0
            exposure = 0.0
            ranked = sorted(zip(candidates, preds), key=lambda x: x[1], reverse=True)[:MAX_POSITIONS_PER_DAY]
            
            for c, score in ranked:
                if score < CONFIDENCE_THRESHOLD: continue
                
                atr = c['s']['atr'][c['idx']]
                alloc = min(TARGET_ANNUAL_VOL/np.sqrt(252)/max(atr, 0.01), MAX_SINGLE_ALLOCATION)
                if exposure + alloc > MAX_LEVERAGE: alloc = MAX_LEVERAGE - exposure
                if alloc <= 0: break

                # --- RATCHET SIMULATION LOGIC ---
                raw_return = c['s']['y'][c['idx']]
                max_gain = c['s']['max_gain'][c['idx']]
                
                # 1. Did we gap down below Initial Stop?
                stop_dist = atr * INITIAL_STOP_ATR
                if c['s']['next_open'][c['idx']] < c['s']['close'][c['idx']] * (1-stop_dist):
                     final_ret = -stop_dist * 1.5 # Gap penalty
                
                # 2. Did we hit the Ratchet Trigger?
                elif max_gain > (atr * RATCHET_TRIGGER):
                    # Triggered! Stop moves to Breakeven.
                    # If the final return is negative, we assume we got stopped at 0.0 (Breakeven)
                    if raw_return < 0:
                        final_ret = 0.0 # Saved by Ratchet!
                    else:
                        final_ret = raw_return # Kept the profit
                
                # 3. Normal Stop Loss
                elif raw_return < -stop_dist:
                    final_ret = -stop_dist
                
                # 4. Profit Target
                elif raw_return > (atr * PROFIT_ATR_MULT):
                    final_ret = atr * PROFIT_ATR_MULT
                
                else:
                    final_ret = raw_return
                
                # Amortize over hold period
                daily_ret = final_ret / PREDICTION_HORIZON
                day_pnl += (alloc * (daily_ret - (TOTAL_FRICTION/PREDICTION_HORIZON)))
                exposure += alloc
                
            rets.append(day_pnl)
            
    print_stats(rets)

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_walk_forward(dev)