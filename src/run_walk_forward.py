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
import gc
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from model import AttentionModel, CNNModel 
from features import add_institutional_features 

# --- GLOBAL CONFIGURATION ---
GOLD_FOLDER = "lake/gold"
META_MODEL_PATH = "models/Judge_Network.pth"

# Universal Model Paths
LSTM_PATH = "models/Universal_LSTM.pth"
GRU_PATH = "models/Universal_GRU.pth"
CNN_PATH = "models/Universal_CNN.pth"
XGB_PATH = "models/Base_XGB.json"
RF_PATH = "models/Base_RF.pkl"

# General Parameters
SEQ_LEN = 60
PRINCIPAL = 10000.0

# --- 🎯 30/30 TARGET PARAMETERS 🎯 ---

# 1. RISK TARGET:
#    We target exactly 30% Annual Volatility to align with your 30% Max DD goal.
TARGET_ANNUAL_VOL = 0.30      

# 2. CONVICTION & FILTERS:
#    Higher confidence needed to pull the trigger.
CONFIDENCE_THRESHOLD = 0.003  # Only trades with strong AI signal
USE_SMA_200_FILTER = True     # CRITICAL: Never buy below 200 SMA (Avoids crashes)
MAX_POSITIONS_PER_DAY = 5     # Top 5 Ideas only

# 3. SWING BRACKETS (Tighter for Safety):
#    Stop Loss = 2.5x ATR (Tighter than before to cap losses)
#    Target    = 5.0x ATR (2:1 Reward/Risk ratio)
STOP_ATR_MULT = 2.5    
PROFIT_ATR_MULT = 5.0  

# 4. Sizing Caps
MAX_SINGLE_ALLOCATION = 0.20  # Max 20% in any one stock
MAX_LEVERAGE = 1.2            # Cap total exposure at 120%

# Costs
RISK_FREE_RATE = 0.045          
TOTAL_COST_PER_TRADE = 0.001

# Walk-Forward Settings
TRAIN_WINDOW = 756  
TEST_WINDOW = 63    
JUDGE_EPOCHS = 20     
JUDGE_LR = 0.001
BATCH_SIZE = 1024

# --- MODEL CLASSES ---
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
    if not os.path.exists(LSTM_PATH) or not os.path.exists(XGB_PATH):
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
    
    if isinstance(X_sequence, torch.Tensor):
        X_sequence = X_sequence.cpu().numpy()
        
    meta_dl_list = []
    total_samples = len(X_sequence)
    
    with torch.no_grad():
        for i in range(0, total_samples, BATCH_SIZE):
            batch_slice = X_sequence[i : i + BATCH_SIZE]
            X_batch_tensor = torch.from_numpy(batch_slice).float().to(device)
            
            p_lstm = m_lstm(X_batch_tensor).cpu().numpy().flatten()
            p_gru = m_gru(X_batch_tensor).cpu().numpy().flatten()
            p_cnn = m_cnn(X_batch_tensor).cpu().numpy().flatten()
            
            batch_meta = np.stack([p_lstm, p_gru, p_cnn], axis=1)
            meta_dl_list.append(batch_meta)
            del X_batch_tensor
            
    if len(meta_dl_list) > 0:
        meta_dl = np.concatenate(meta_dl_list, axis=0)
    else:
        return np.array([]) 

    X_tree = X_sequence.reshape(X_sequence.shape[0], -1)
    p_xgb = xgb_m.predict(X_tree)
    p_rf = rf_m.predict(X_tree)

    X_meta = np.concatenate([meta_dl, p_xgb.reshape(-1, 1), p_rf.reshape(-1, 1)], axis=1)
    return X_meta

def train_judge_on_data(X_meta, y_true, device):
    X_tensor = torch.from_numpy(X_meta).float().to(device)
    y_tensor = torch.from_numpy(y_true).float().to(device)

    judge = JudgeNetwork(input_dim=X_meta.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(judge.parameters(), lr=JUDGE_LR)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True) 

    judge.train()
    for epoch in range(JUDGE_EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = judge(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    judge.eval()
    return judge

def print_comprehensive_stats(daily_returns_list):
    if not daily_returns_list:
        print("No returns data to analyze.")
        return

    equity_curve = [PRINCIPAL]
    for r in daily_returns_list:
        equity_curve.append(equity_curve[-1] * (1 + r))
    
    equity_series = pd.Series(equity_curve)
    returns_series = pd.Series(daily_returns_list)
    
    total_return = (equity_series.iloc[-1] / PRINCIPAL) - 1
    days = len(returns_series)
    years = days / 252.0
    if years < 0.1: years = 0.1 
    
    cagr = (1 + total_return)**(1/years) - 1
    ann_volatility = returns_series.std() * np.sqrt(252)
    
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    
    sharpe = (cagr - RISK_FREE_RATE) / (ann_volatility + 1e-9)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    win_rate = (returns_series > 0).sum() / len(returns_series) if len(returns_series) > 0 else 0

    print("\n=======================================================")
    print("📊 30/30 TARGET STRATEGY PERFORMANCE")
    print("=======================================================")
    print(f"1. Annualized Return (CAGR):  {cagr:.2%}")
    print(f"2. Max Drawdown:              {max_dd:.2%}")
    print(f"3. Sharpe Ratio:              {sharpe:.2f}")
    print(f"4. Calmar Ratio:              {calmar:.2f}")
    print(f"5. Win Rate (Days):           {win_rate:.2%}")
    print(f"6. Final Equity:              {equity_series.iloc[-1]:,.2f} INR")
    print("=======================================================")

def run_walk_forward(device):
    print("\n--- PHASE 4: 30/30 OPTIMIZED WALK-FORWARD ---")
    
    universal_models = load_universal_models(device)
    all_files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    
    print(f"Loading {len(all_files)} stocks...")
    stock_cache = []
    
    # 1. CACHE RAW DATA (No Leakage)
    for f in tqdm(all_files, desc="Caching Data"):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            
            # Target: 3-Day Return
            df['Realized_Ret_3D'] = df['Close'].pct_change(3).shift(-3).fillna(0)
            
            # Indicators
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
            df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR_20'] = df['TR'].rolling(20).mean()
            df['ATR_Pct'] = (df['ATR_20'] / df['Close']).fillna(0.02)
            df['ATR_Pct_Lagged'] = df['ATR_Pct'].shift(1).fillna(0.02) 
            
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Trend_Up'] = (df['Close'] > df['SMA_50']).astype(int)
            df['Bull_Market'] = (df['Close'] > df['SMA_200']).astype(int)
            
            df_feats = add_institutional_features(df.copy())
            
            stock_cache.append({
                'ticker': os.path.basename(f),
                'dates': df.index,
                'data_raw': df_feats.values.astype(np.float32), 
                'target': df['Realized_Ret_3D'].values,
                'atr': df['ATR_Pct_Lagged'].values, 
                'bull_check': df['Bull_Market'].values,
                'trend': df['Trend_Up'].values,
                'full_len': len(df)
            })
        except Exception:
            continue

    if not stock_cache: return
    all_dates = sorted(list(set([d for s in stock_cache for d in s['dates']])))
    
    if len(all_dates) < (TRAIN_WINDOW + TEST_WINDOW):
        print("❌ Not enough data.")
        return

    start_idx = SEQ_LEN + TRAIN_WINDOW
    end_idx = len(all_dates) - TEST_WINDOW
    full_history_daily_returns = []
    
    loop_pbar = tqdm(range(start_idx, end_idx, TEST_WINDOW), desc="Walk-Forward Loop")
    
    for current_idx in loop_pbar:
        
        # --- A. STRICT SCALING ---
        X_train_raw_list = []
        training_subset = random.sample(stock_cache, min(len(stock_cache), 150))
        
        for stock in training_subset:
             current_date = all_dates[current_idx]
             if current_date not in stock['dates']: continue
             curr_stock_idx = stock['dates'].get_loc(current_date)
             
             if curr_stock_idx > SEQ_LEN + TRAIN_WINDOW:
                 slice_start = curr_stock_idx - TRAIN_WINDOW
                 slice_end = curr_stock_idx
                 X_chunk = stock['data_raw'][slice_start:slice_end]
                 X_train_raw_list.append(X_chunk)
        
        if not X_train_raw_list: continue
        
        X_train_concat = np.concatenate(X_train_raw_list, axis=0)
        scaler = MinMaxScaler()
        scaler.fit(X_train_concat) 
        
        # --- B. TRAIN JUDGE ---
        X_judge_train = []
        y_judge_train = []
        
        for stock in training_subset:
             current_date = all_dates[current_idx]
             if current_date not in stock['dates']: continue
             curr_stock_idx = stock['dates'].get_loc(current_date)
             
             if curr_stock_idx > SEQ_LEN + TRAIN_WINDOW:
                 slice_start = curr_stock_idx - TRAIN_WINDOW
                 slice_end = curr_stock_idx
                 X_raw_chunk = stock['data_raw'][slice_start:slice_end]
                 X_scaled_chunk = scaler.transform(X_raw_chunk) 
                 
                 sample_indices = np.linspace(SEQ_LEN, len(X_scaled_chunk)-1, 10, dtype=int)
                 for i in sample_indices:
                     X_judge_train.append(X_scaled_chunk[i-SEQ_LEN:i])
                     y_judge_train.append(stock['target'][slice_start + i])

        if not X_judge_train: continue
        X_train_agg = np.array(X_judge_train)
        y_train_agg = np.array(y_judge_train)
        
        X_train_meta = generate_meta_features(X_train_agg, universal_models, device)
        judge = train_judge_on_data(X_train_meta, y_train_agg, device)
        
        # --- C. TEST DAY SIMULATION ---
        test_start_date = all_dates[current_idx]
        test_end_date = all_dates[min(current_idx + TEST_WINDOW, len(all_dates)-1)]
        test_dates = [d for d in all_dates if test_start_date <= d < test_end_date]
        
        for d in test_dates:
            day_X_raw = []
            day_meta = [] 
            
            for stock in stock_cache:
                if d in stock['dates']:
                    idx = stock['dates'].get_loc(d)
                    if idx >= SEQ_LEN:
                        raw_seq = stock['data_raw'][idx-SEQ_LEN:idx]
                        day_X_raw.append(raw_seq)
                        
                        daily_proxy_ret = stock['target'][idx] / 3.0 
                        
                        day_meta.append({
                            'ret': daily_proxy_ret, 
                            'atr': stock['atr'][idx],
                            'bull': stock['bull_check'][idx],
                            'trend': stock['trend'][idx]
                        })
            
            if not day_X_raw:
                full_history_daily_returns.append(0.0)
                continue
            
            X_batch_raw = np.array(day_X_raw) 
            N, L, F = X_batch_raw.shape
            X_batch_flat = X_batch_raw.reshape(-1, F)
            X_batch_scaled_flat = scaler.transform(X_batch_flat)
            X_batch_scaled = X_batch_scaled_flat.reshape(N, L, F)
            
            X_meta = generate_meta_features(X_batch_scaled, universal_models, device)
            X_meta_tensor = torch.from_numpy(X_meta).float().to(device)
            
            with torch.no_grad():
                preds = judge(X_meta_tensor).cpu().numpy().flatten()
            
            df_day = pd.DataFrame(day_meta)
            df_day['pred'] = preds
            df_day['Abs_Pred'] = df_day['pred'].abs()
            
            # --- FILTERS & SELECTION ---
            # Using .copy() to fix SettingWithCopyWarning
            candidates = df_day[
                (df_day['Abs_Pred'] > CONFIDENCE_THRESHOLD) &      
                (df_day['bull'] == 1) &             
                (df_day['trend'] == 1)              
            ].copy()
            
            if len(candidates) > 0:
                candidates['Score'] = candidates['Abs_Pred'] / (candidates['atr'] + 1e-9)
                candidates = candidates.sort_values('Score', ascending=False).head(MAX_POSITIONS_PER_DAY)
                
                day_pnl = 0.0
                total_allocation = 0.0
                
                for _, row in candidates.iterrows():
                    if row['atr'] > 0.08: continue # Volatility sanity check
                    
                    # 30/30 Vol Sizing
                    # Target 30% Annual Vol -> ~1.9% Daily Vol
                    daily_target_vol = TARGET_ANNUAL_VOL / np.sqrt(252) 
                    stock_atr = max(row['atr'], 0.01) 
                    allocation = min(daily_target_vol / stock_atr, MAX_SINGLE_ALLOCATION)
                    
                    # Check Leverage Cap
                    if total_allocation + allocation > MAX_LEVERAGE:
                        allocation = MAX_LEVERAGE - total_allocation
                    
                    if allocation <= 0: break
                    
                    total_allocation += allocation
                    
                    if row['pred'] > 0:
                        actual_ret = row['ret']
                        # Stop Loss Simulation (2.5x ATR)
                        stop_dist = row['atr'] * STOP_ATR_MULT
                        if actual_ret < -stop_dist: 
                            actual_ret = -stop_dist
                            
                        trade_pnl = (allocation * actual_ret) - (TOTAL_COST_PER_TRADE / 3.0) 
                        day_pnl += trade_pnl
                
                # Portfolio Stop (Hard cap at -3% per day)
                if day_pnl < -0.03: day_pnl = -0.03
                full_history_daily_returns.append(day_pnl)
            else:
                full_history_daily_returns.append(0.0)

    print_comprehensive_stats(full_history_daily_returns)

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
    run_walk_forward(DEVICE)