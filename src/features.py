import numpy as np
import pandas as pd

# --- NEW FUNCTION FOR MAGNITUDE FEATURE ---
# CHANGE 1A: Default window changed from 14 to 21 for smoother RSI
def get_rsi(df, window=21): 
    """ Standard Relative Strength Index (RSI), normalized 0-1. """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # EWM calculation is stable for large series
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    # Prevent division by zero
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 100).fillna(0) # Normalize to 0-1 range

# CHANGE 1B: Default window changed from 30 to 50 for smoother volatility estimate
def get_yang_zhang_volatility(df, window=50): 
    """
    Elite Volatility Estimator (Yang-Zhang).
    Accounts for overnight gaps (Open vs previous Close).
    """
    # Prevent divide by zero errors
    open_clean = df['Open'].replace(0, 1e-9)
    close_shift_clean = df['Close'].shift(1).replace(0, 1e-9)
    
    log_ho = (df['High'] / open_clean).apply(np.log)
    log_lo = (df['Low'] / open_clean).apply(np.log)
    log_co = (df['Close'] / open_clean).apply(np.log)
    
    log_oc = (df['Open'] / close_shift_clean).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (df['Close'] / close_shift_clean).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    # Use the adjusted window value (50) for rolling calculations
    close_vol = log_cc_sq.rolling(window=window).sum() / (window - 1.0)
    open_vol = log_oc_sq.rolling(window=window).sum() / (window - 1.0)
    window_rs = rs.rolling(window=window).sum() / (window - 1.0)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(252)
    return result.fillna(0)

def add_institutional_features(df):
    """
    Transforms raw OHLCV data into 20 'Elite' statistical features.
    
    The output is a DataFrame with exactly 20 columns.
    """
    df = df.copy()
    
    # 1. Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # 2. Yang-Zhang Volatility
    # Now uses the window=50 from the updated function
    df['YZ_Vol'] = get_yang_zhang_volatility(df) 
    
    # 3. Z-Score (20-day Mean Reversion Signal)
    roll_mean_20 = df['Close'].rolling(window=20).mean()
    roll_std_20 = df['Close'].rolling(window=20).std()
    df['Z_Score'] = ((df['Close'] - roll_mean_20) / (roll_std_20 + 1e-9)).fillna(0)
    
    # 4. Volume Imbalance (Order Flow Proxy)
    bar_range = (df['High'] - df['Low']).replace(0, 1e-9)
    df['Vol_Imbalance'] = ((df['Close'] - df['Open']) / bar_range) * df['Volume']
    df['Vol_Imbalance'] = (df['Vol_Imbalance'] - df['Vol_Imbalance'].rolling(20).mean()) / (df['Vol_Imbalance'].rolling(20).std() + 1e-9)

    # 5. Interaction Features (Volatility x Return)
    df['Vol_x_Ret'] = df['YZ_Vol'] * df['Log_Ret']
    
    # --- NEW MAGNITUDE FEATURES ---
    # 6. RSI (Momentum/Conviction)
    # Now uses the window=21 from the updated function
    df['RSI'] = get_rsi(df) 
    
    # 7. Volume Z-Score (Conviction/Magnitude)
    log_vol = np.log1p(df['Volume']) 
    df['Vol_ZScore'] = (log_vol - log_vol.rolling(window=30).mean()) / (log_vol.rolling(window=30).std() + 1e-9)
    
    # CHANGE 2: Add Stronger, Long-Term Z-Score Feature (50-day)
    roll_mean_50 = df['Close'].rolling(window=50).mean()
    roll_std_50 = df['Close'].rolling(window=50).std()
    df['Z_Score_50'] = ((df['Close'] - roll_mean_50) / (roll_std_50 + 1e-9)).fillna(0) # <-- NEW FEATURE

    # --- GENERATE LAGS ---
    feature_set = [
        'Log_Ret', 'YZ_Vol', 'Z_Score', 'Vol_Imbalance', 'Vol_x_Ret', 
        'RSI', 'Vol_ZScore' # 7 base features
    ]
    
    # Generate 3 lags for the first 5 base features
    for lag in [1, 2, 3]:
        for feat in feature_set[0:5]: 
            df[f'{feat}_L{lag}'] = df[feat].shift(lag)
            
    # Generate 1 lag for the 2 magnitude features
    for lag in [1]: 
        for feat in feature_set[5:]: 
            df[f'{feat}_L{lag}'] = df[feat].shift(lag)
            
    # --- FINAL 20-FEATURE LIST ---
    # We replaced 'Z_Score_L3' with 'Z_Score_50'
    final_cols = [
        # 7 Base Features
        'Log_Ret', 'YZ_Vol', 'Z_Score', 'Vol_Imbalance', 'Vol_x_Ret', 'RSI', 'Vol_ZScore', 
        
        # 10 Lag-1 and Lag-2 Features
        'Log_Ret_L1', 'YZ_Vol_L1', 'Z_Score_L1', 'Vol_Imbalance_L1', 'Vol_x_Ret_L1', 
        'Log_Ret_L2', 'YZ_Vol_L2', 'Z_Score_L2', 'Vol_Imbalance_L2', 'Vol_x_Ret_L2',
        
        # 3 Final Features (RSI_L1, Vol_ZScore_L1, Z_Score_50)
        'RSI_L1', 'Vol_ZScore_L1',
        'Z_Score_50' # <-- NEW STRONGER FEATURE
    ] 
    
    return df[final_cols].fillna(0)