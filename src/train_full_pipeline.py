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

# --- 🛡️ LOW DRAWDOWN "SAFE GROWTH" PARAMETERS 🎯 ---

# 1. Diversification: Hold 12 stocks to dilute risk of any single crash
MAX_POSITIONS_PER_DAY = 12      

# 2. Safety Cap: Never put more than 10% in one stock
BASE_ALLOCATION = 0.10          

# 3. Balanced Risk: 0.8% Risk per trade (Safe but still grows)
TARGET_RISK_PER_TRADE = 0.008   

# 4. Strict Safety: Ignore stocks moving >3% a day (Avoids "Falling Knives")
MAX_ALLOWED_VOLATILITY = 0.03   

# 5. Quality Filter
CONFIDENCE_THRESHOLD = 0.0025   

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
    sortino = (cagr - RISK_FREE_RATE) / (returns_series[returns_series<0].std() * np.sqrt(252) + 1e-9)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    positive_days = (returns_series > 0).sum()
    win_rate_daily = positive_days / len(returns_series) if len(returns_series) > 0 else 0

    print("\n=======================================================")
    print("📊 SAFE GROWTH PERFORMANCE METRICS (Low Drawdown)")
    print("=======================================================")
    print(f"1. Annualized Return (CAGR):  {cagr:.2%}")
    print(f"2. Max Drawdown:              {max_dd:.2%}")
    print(f"3. Sharpe Ratio:              {sharpe:.2f}")
    print(f"4. Calmar Ratio:              {calmar:.2f}")
    print(f"5. Daily Win Rate:            {win_rate_daily:.2%}")
    print(f"6. Final Equity:              {equity_series.iloc[-1]:,.2f} INR")
    print("=======================================================")
    
    if max_dd > -0.20:
        print("✅ SUCCESS: Drawdown is Safe (<20%).")
    else:
        print("⚠️ VERDICT: Optimization still required.")

def run_walk_forward(device):
    print("\n--- PHASE 4: SAFE GROWTH WALK-FORWARD VALIDATION ---")
    
    universal_models = load_universal_models(device)
    all_files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    
    print(f"Loading {len(all_files)} stocks for analysis...")
    stock_cache = []
    
    for f in tqdm(all_files, desc="Caching Stocks"):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            df['Realized_Ret'] = df['Close'].pct_change().shift(-1).fillna(0)
            df['Daily_Vol'] = df['Close'].pct_change().rolling(20).std().fillna(0.02)
            
            # --- TREND FILTER: 20-Day SMA ---
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['Trend_Up'] = (df['Close'] > df['SMA_20']).astype(int)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df_feats = add_institutional_features(df.copy())
            data_values = df_feats.values.astype(np.float32)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data_values)
            
            stock_cache.append({
                'ticker': os.path.basename(f),
                'dates': df.index,
                'data_scaled': data_scaled,
                'realized_ret': df['Realized_Ret'].values,
                'vol': df['Daily_Vol'].values,
                'rsi': df['RSI'].values,
                'trend': df['Trend_Up'].values, 
                'full_len': len(df)
            })
        except Exception:
            continue

    if not stock_cache:
        print("❌ CRITICAL: No valid data found.")
        return

    all_dates = sorted(list(set([d for s in stock_cache for d in s['dates']])))
    
    if len(all_dates) < (TRAIN_WINDOW + TEST_WINDOW):
        print("❌ Not enough historical data.")
        return

    start_idx = SEQ_LEN + TRAIN_WINDOW
    end_idx = len(all_dates) - TEST_WINDOW
    full_history_daily_returns = []
    
    loop_pbar = tqdm(range(start_idx, end_idx, TEST_WINDOW), desc="Walk-Forward Windows")
    
    for current_idx in loop_pbar:
        
        # --- A. TRAIN JUDGE ---
        X_train_list = []
        y_train_list = []
        training_subset = random.sample(stock_cache, min(len(stock_cache), 150))

        for stock in training_subset:
            if stock['full_len'] > (SEQ_LEN + TRAIN_WINDOW):
                 start = np.random.randint(SEQ_LEN, stock['full_len'] - TRAIN_WINDOW)
                 slice_end = start + TRAIN_WINDOW
                 X_raw = stock['data_scaled'][start:slice_end]
                 indices = range(SEQ_LEN, len(X_raw), 5)
                 X_seq_temp = []
                 y_seq_temp = []
                 for i in indices:
                     X_seq_temp.append(X_raw[i-SEQ_LEN:i])
                     y_seq_temp.append(X_raw[i, 0])
                 if X_seq_temp:
                    X_train_list.append(np.array(X_seq_temp))
                    y_train_list.append(np.array(y_seq_temp))
        
        if not X_train_list: continue
        X_train_agg = np.concatenate(X_train_list, axis=0)
        y_train_agg = np.concatenate(y_train_list, axis=0)
        
        del X_train_list, y_train_list
        gc.collect()
        
        X_train_meta = generate_meta_features(X_train_agg, universal_models, device)
        judge = train_judge_on_data(X_train_meta, y_train_agg, device)
        
        del X_train_agg, y_train_agg
        gc.collect()
        
        # --- B. TEST SIMULATION ---
        test_start_date = all_dates[current_idx]
        test_end_date = all_dates[min(current_idx + TEST_WINDOW, len(all_dates)-1)]
        test_dates = [d for d in all_dates if test_start_date <= d < test_end_date]
        
        for d in test_dates:
            day_X = []
            day_meta = [] 
            
            for stock in stock_cache:
                if d in stock['dates']:
                    idx = stock['dates'].get_loc(d)
                    if idx >= SEQ_LEN:
                        day_X.append(stock['data_scaled'][idx-SEQ_LEN:idx])
                        day_meta.append({
                            'ret': stock['realized_ret'][idx],
                            'vol': stock['vol'][idx],
                            'rsi': stock['rsi'][idx],
                            'trend': stock['trend'][idx] # Filter Variable
                        })
            
            if not day_X:
                full_history_daily_returns.append(0.0)
                continue
            
            X_batch_arr = np.array(day_X)
            X_meta = generate_meta_features(X_batch_arr, universal_models, device)
            X_meta_tensor = torch.from_numpy(X_meta).float().to(device)
            
            with torch.no_grad():
                preds = judge(X_meta_tensor).cpu().numpy().flatten()
            
            df_day = pd.DataFrame(day_meta)
            df_day['pred'] = preds
            df_day['Abs_Pred'] = df_day['pred'].abs()
            
            # --- TREND SNIPER FILTER ---
            longs = df_day[
                (df_day['pred'] > CONFIDENCE_THRESHOLD) & 
                (df_day['rsi'] > 45) &
                (df_day['trend'] == 1) # Uptrend
            ]
            
            shorts = df_day[
                (df_day['pred'] < -CONFIDENCE_THRESHOLD) & 
                (df_day['rsi'] < 55) &
                (df_day['trend'] == 0) # Downtrend
            ]
            
            candidates = pd.concat([longs, shorts])
            candidates = candidates.sort_values('Abs_Pred', ascending=False).head(MAX_POSITIONS_PER_DAY)
            
            day_pnl = 0.0
            
            for _, row in candidates.iterrows():
                if row['vol'] > MAX_ALLOWED_VOLATILITY: continue
                
                vol = max(row['vol'], 0.01)
                calculated_allocation = TARGET_RISK_PER_TRADE / vol
                allocation = min(BASE_ALLOCATION, calculated_allocation)
                
                trade_ret = (np.sign(row['pred']) * row['ret']) - TOTAL_COST_PER_TRADE
                day_pnl += allocation * trade_ret
            
            full_history_daily_returns.append(day_pnl)

    print_comprehensive_stats(full_history_daily_returns)

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        DEVICE = torch.device("cpu")
    run_walk_forward(DEVICE)