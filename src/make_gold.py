import pandas as pd
import pandas_ta as ta
import numpy as np
import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil

# --- CONFIGURATION ---
INPUT_DIR = "lake/silver"
OUTPUT_DIR = "lake/gold"
MIN_MEDIAN_VOLUME = 50000 

# --- AUTO-CLEANER ---
# If the folder exists, delete it and recreate it empty.
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)


# --- CONFIGURATION ---
INPUT_DIR = "lake/silver"
OUTPUT_DIR = "lake/gold"
MIN_MEDIAN_VOLUME = 50000 
# Create/Recreate the Gold directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- THE 18-FEATURE FORMULA ---
def process_stock(file_path):
    try:
        # 1. Read Raw Data from Silver
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Filter 1: History Check (Need 1 year of data)
        if len(df) < 300: return None
        
        # Deduplicate index just in case
        df = df[~df.index.duplicated(keep='first')]
        
        # Keep only raw columns to start fresh
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not set(required).issubset(df.columns): return None
        
        df = df[required].copy()

        # --- 🛡️ FILTER 2: LIQUIDITY CHECK (The "Circuit Breaker" Defense) ---
        # Calculate median volume over the last year
        # If a stock trades less than 50k shares a day, it is dangerous. SKIP IT.
        median_vol = df['Volume'].rolling(200).median().iloc[-1]
        if median_vol < MIN_MEDIAN_VOLUME:
            return None

        # 2. Calculate Hybrid Features (Same as before)
        
        # --- A. Momentum ---
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RSI'] = ta.rsi(df['Close'], length=14)
        mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['MFI'] = mfi if mfi is not None else 0
        df['WillR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)

        # --- B. Trend ---
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd[macd.columns[0]] 
            df['MACD_Hist'] = macd[macd.columns[1]]
        
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None: df['ADX'] = adx[adx.columns[0]]
            
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

        # --- C. Volatility & Volume ---
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_Slope'] = df['OBV'].pct_change()
        
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None: 
            df['BB_Width'] = bbands[bbands.columns[2]] - bbands[bbands.columns[0]]

        # --- D. Lags (Velocity) ---
        df['Ret_Lag1'] = df['Log_Ret'].shift(1)
        df['Ret_Lag3'] = df['Log_Ret'].shift(3)
        df['Ret_Lag5'] = df['Log_Ret'].shift(5)

        # --- E. Time Embeddings ---
        df['Date'] = df.index
        day = df['Date'].dt.day
        month = df['Date'].dt.month
        df['Day_Sin'] = np.sin(2 * np.pi * day / 31)
        df['Day_Cos'] = np.cos(2 * np.pi * day / 31)
        df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * month / 12)
        
        # 3. Final Polish
        feature_cols = [
            'Log_Ret', 'Ret_Lag1', 'Ret_Lag3', 'Ret_Lag5',
            'RSI', 'MFI', 'WillR', 'MACD', 'MACD_Hist',
            'ADX', 'Dist_EMA50', 'ATR_Pct', 'OBV_Slope', 'BB_Width',
            'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos'
        ]
        
        df.dropna(subset=feature_cols, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Save to Gold
        final_cols = required + feature_cols
        file_name = os.path.basename(file_path)
        save_path = os.path.join(OUTPUT_DIR, file_name)
        
        df[final_cols].to_csv(save_path)
        return file_name
        
    except Exception as e:
        return None

def main():
    files = glob.glob(f"{INPUT_DIR}/*.csv") + glob.glob(f"{INPUT_DIR}/*.NS")
    files = [f for f in files if os.path.isfile(f)]
    
    print(f"🏭 Manufacturing Gold Lake from {len(files)} Silver stocks...")
    print(f"   (Filtering: Removing stocks with Median Volume < {MIN_MEDIAN_VOLUME})")
    
    results = Parallel(n_jobs=-1)(delayed(process_stock)(f) for f in tqdm(files))
    
    success = [r for r in results if r is not None]
    print(f"✅ Gold Lake Created! {len(success)} high-quality files saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()