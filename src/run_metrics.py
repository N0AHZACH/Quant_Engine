import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import joblib
import xgboost as xgb
import sys
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
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

# --- 🎯 AGGRESSIVE GROWTH SETTINGS (Your Selection) 🎯 ---
MAX_POSITIONS_PER_DAY = 6       # Balanced: Allows diversification without over-trading
BASE_ALLOCATION = 1.0 / MAX_POSITIONS_PER_DAY 
TARGET_RISK_PER_TRADE = 0.015   # 1.5% Risk (High Growth)
MAX_ALLOWED_VOLATILITY = 0.06   # 6% Volatility Cap (Allows mid-cap movers)
CONFIDENCE_THRESHOLD = 0.004    # 0.4% Predicted Return (Captures more setups)
RISK_FREE_RATE = 0.045          
TOTAL_COST_PER_TRADE = 0.0015 

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

def load_models(device):
    print("⏳ Loading Models...")
    required = [LSTM_PATH, GRU_PATH, CNN_PATH, XGB_PATH, RF_PATH]
    for f in required:
        if not os.path.exists(f):
            print(f"❌ CRITICAL: Missing model file: {f}")
            sys.exit()

    m_lstm = AttentionModel("lstm").to(device); m_lstm.load_state_dict(torch.load(LSTM_PATH, weights_only=True)); m_lstm.eval()
    m_gru = AttentionModel("gru").to(device); m_gru.load_state_dict(torch.load(GRU_PATH, weights_only=True)); m_gru.eval()
    m_cnn = CNNModel().to(device); m_cnn.load_state_dict(torch.load(CNN_PATH, weights_only=True)); m_cnn.eval()
    
    xgb_m = xgb.XGBRegressor(); xgb_m.load_model(XGB_PATH)
    rf_m = joblib.load(RF_PATH)
    
    judge = None
    if os.path.exists(META_MODEL_PATH):
        print("✅ Judge Network Found. Loading...")
        judge = JudgeNetwork(input_dim=5).to(device)
        judge.load_state_dict(torch.load(META_MODEL_PATH, weights_only=True))
        judge.eval()
    else:
        print("⚠️ Judge Network NOT found. Using 'Voting Ensemble' (Average).")
    
    return (m_lstm, m_gru, m_cnn, xgb_m, rf_m), judge

def get_predictions_for_stock(df, models, judge, device):
    df_features = add_institutional_features(df.copy())
    data_values = df_features.values.astype(np.float32)
    
    # Honest Out-of-Sample Scaling
    split_idx = int(len(data_values) * 0.5)
    if split_idx < SEQ_LEN: return None
    
    scaler = MinMaxScaler()
    scaler.fit(data_values[:split_idx]) 
    data_scaled = scaler.transform(data_values)
    
    X_raw = np.array([data_scaled[i-SEQ_LEN:i] for i in range(split_idx, len(data_scaled))])
    if len(X_raw) == 0: return None
    
    m_lstm, m_gru, m_cnn, xgb_m, rf_m = models
    X_tensor = torch.from_numpy(X_raw).to(device)
    X_tree = X_raw.reshape(X_raw.shape[0], -1)

    with torch.no_grad():
        p_lstm = m_lstm(X_tensor).cpu().numpy().flatten()
        p_gru = m_gru(X_tensor).cpu().numpy().flatten()
        p_cnn = m_cnn(X_tensor).cpu().numpy().flatten()
    
    p_xgb = xgb_m.predict(X_tree)
    p_rf = rf_m.predict(X_tree)

    if judge:
        X_meta = np.stack([p_lstm, p_gru, p_cnn, p_xgb, p_rf], axis=1)
        X_meta_tensor = torch.from_numpy(X_meta).float().to(device)
        with torch.no_grad():
            final_preds = judge(X_meta_tensor).cpu().numpy().flatten()
    else:
        final_preds = np.mean([p_lstm, p_gru, p_cnn, p_xgb, p_rf], axis=0)
        
    full_preds = np.full(len(df), np.nan)
    full_preds[split_idx:] = final_preds
    return full_preds

def calculate_advanced_metrics(equity_curve, trade_log_df):
    daily_rets = equity_curve.pct_change().fillna(0)
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    years = max(len(equity_curve) / 252.0, 0.1)
    cagr = (1 + total_ret)**(1/years) - 1
    
    vol = daily_rets.std() * np.sqrt(252)
    downside_rets = daily_rets[daily_rets < 0]
    downside_dev = downside_rets.std() * np.sqrt(252)
    
    sharpe = (cagr - RISK_FREE_RATE) / (vol + 1e-9)
    sortino = (cagr - RISK_FREE_RATE) / (downside_dev + 1e-9)
    
    cumulative = (1 + daily_rets).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    mdd = drawdown.min()
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    if trade_log_df.empty:
        return {"Error": "No Trades Executed", "TotalTrades": 0}

    wins = trade_log_df[trade_log_df['PnL'] > 0]
    losses = trade_log_df[trade_log_df['PnL'] <= 0]
    
    win_rate = len(wins) / len(trade_log_df)
    avg_win = wins['PnL'].mean() if not wins.empty else 0
    avg_loss = losses['PnL'].mean() if not losses.empty else 0
    
    gross_profit = wins['PnL'].sum()
    gross_loss = abs(losses['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    trade_std = trade_log_df['PnL'].std()
    sqn = (expectancy / trade_std) * np.sqrt(len(trade_log_df)) if trade_std != 0 else 0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDD": mdd,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "Expectancy": expectancy,
        "SQN": sqn,
        "TotalTrades": len(trade_log_df)
    }

def run_metrics_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Processing on {device}...")
    
    models, judge = load_models(device)
    all_files = glob.glob(f"{GOLD_FOLDER}/*.csv")
    
    print("Scanning market for Safe High-Growth opportunities...")
    all_opportunities = []
    
    for f in tqdm(all_files, desc="Scanning Stocks"):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            # Targets
            df['Realized_Ret'] = df['Close'].pct_change().shift(-1).fillna(0)
            df['Daily_Vol'] = df['Close'].pct_change().rolling(20).std().fillna(0.02)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            if len(df) < SEQ_LEN + 50: continue
            
            preds = get_predictions_for_stock(df, models, judge, device)
            if preds is None: continue
            
            df['Pred'] = preds
            valid_df = df.dropna(subset=['Pred'])
            
            strong = valid_df[valid_df['Pred'].abs() > CONFIDENCE_THRESHOLD].copy()
            strong['Ticker'] = os.path.basename(f).replace('.csv', '')
            
            if not strong.empty:
                keep = ['Ticker', 'Pred', 'Realized_Ret', 'RSI', 'Daily_Vol']
                all_opportunities.append(strong[keep].reset_index().rename(columns={'index': 'Date'}))
                
        except Exception: continue

    if not all_opportunities:
        print("❌ No valid signals found.")
        return

    master_df = pd.concat(all_opportunities, ignore_index=True)
    master_df['Abs_Pred'] = master_df['Pred'].abs()
    master_df = master_df.sort_values(by=['Date', 'Abs_Pred'], ascending=[True, False])
    
    print(f"Simulating Strategy (Strict 100% Exposure Cap)...")
    
    dates = np.sort(master_df['Date'].unique())
    portfolio_equity = [PRINCIPAL]
    trade_log = [] 
    
    for d in tqdm(dates, desc="Simulating Days"):
        daily_opportunities = master_df[master_df['Date'] == d]
        
        longs = daily_opportunities[(daily_opportunities['Pred'] > 0) & (daily_opportunities['RSI'] > 45)]
        shorts = daily_opportunities[(daily_opportunities['Pred'] < 0) & (daily_opportunities['RSI'] < 55)]
        
        candidates = pd.concat([longs, shorts])
        candidates = candidates.sort_values('Abs_Pred', ascending=False).head(MAX_POSITIONS_PER_DAY)
        
        if candidates.empty:
            portfolio_equity.append(portfolio_equity[-1])
            continue
            
        day_pnl = 0.0
        current_exposure = 0.0
        
        for _, row in candidates.iterrows():
            if row['Daily_Vol'] > MAX_ALLOWED_VOLATILITY: continue 

            vol = max(row['Daily_Vol'], 0.01)
            calculated_allocation = TARGET_RISK_PER_TRADE / vol
            allocation = min(BASE_ALLOCATION, calculated_allocation)
            
            if current_exposure + allocation > 1.0:
                allocation = 1.0 - current_exposure
                if allocation <= 0: break
            
            current_exposure += allocation
            
            direction = np.sign(row['Pred'])
            raw_ret = (direction * row['Realized_Ret'])
            net_ret = raw_ret - TOTAL_COST_PER_TRADE
            
            trade_pnl_value = (portfolio_equity[-1] * allocation) * net_ret
            day_pnl += allocation * net_ret
            
            trade_log.append({
                'Date': d,
                'Ticker': row['Ticker'],
                'Direction': 'LONG' if direction > 0 else 'SHORT',
                'Alloc%': allocation,
                'Return%': net_ret,
                'PnL': trade_pnl_value
            })
            
        new_equity = portfolio_equity[-1] * (1 + day_pnl)
        portfolio_equity.append(new_equity)

    # --- REPORTING ---
    equity_curve = pd.Series(portfolio_equity)
    trade_df = pd.DataFrame(trade_log)
    
    stats = calculate_advanced_metrics(equity_curve, trade_df)
    
    print("\n" + "="*60)
    print("🏆 PROFESSIONAL PERFORMANCE REPORT (OOS Test)")
    print("="*60)
    
    if "Error" in stats:
        print(f"❌ Metrics Error: {stats['Error']}")
        return

    print(f"\n--- 📈 RETURN METRICS ---")
    print(f"Final Equity:      ₹{portfolio_equity[-1]:,.2f}")
    print(f"Total Return:      {(portfolio_equity[-1]/PRINCIPAL - 1):.2%}")
    print(f"CAGR (Annual):     {stats['CAGR']:.2%}")
    
    print(f"\n--- 🛡️ RISK METRICS ---")
    print(f"Volatility (Ann):  {stats['Vol']:.2%}")
    print(f"Max Drawdown:      {stats['MaxDD']:.2%}")
    print(f"Sharpe Ratio:      {stats['Sharpe']:.2f}  (>1.0 is Good, >2.0 is Great)")
    print(f"Sortino Ratio:     {stats['Sortino']:.2f}  (Penalizes only downside volatility)")
    print(f"Calmar Ratio:      {stats['Calmar']:.2f}  (Return / Max Drawdown)")
    
    print(f"\n--- ⚡ TRADE QUALITY ---")
    print(f"Total Trades:      {stats['TotalTrades']}")
    print(f"Win Rate:          {stats['WinRate']:.2%}")
    print(f"Profit Factor:     {stats['ProfitFactor']:.2f}  (>1.5 is ideal)")
    print(f"Expectancy:        ₹{stats['Expectancy']:.2f} per trade")
    print(f"SQN Score:         {stats['SQN']:.2f}  (System Quality Number)")
    print(f"Avg Win / Loss:    ₹{stats['AvgWin']:.2f} / ₹{stats['AvgLoss']:.2f}")

    print("="*60)
    
    if not trade_df.empty:
        trade_df.to_csv("backtest_trade_log.csv", index=False)
        print("\n💾 Saved detailed trade log to 'backtest_trade_log.csv'")

if __name__ == "__main__":
    run_metrics_pipeline()