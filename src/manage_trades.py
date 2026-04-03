import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import os

# --- CONFIGURATION ---
LOG_FILE = "paper_trading_log.csv"
INITIAL_CAPITAL = 10000.0       # 💰 STARTING BALANCE
ATR_MULTIPLIER_SCALE_OUT = 3.0  # Sell 50% target
TIME_STOP_DAYS = 5              # Sell ALL if stagnant > 5 days
MIN_PROFIT_TO_SURVIVE = 1.0     # Must be > 1x ATR to survive Time Stop

# 🙈 IGNORE LIST: Tickers to hide from view (Visual only)
IGNORE_TICKERS = [] 

def manage_portfolio_nightly():
    print("🌙 NIGHTLY PORTFOLIO MANAGER (Auto-Execution Mode)...")
    
    if not os.path.exists(LOG_FILE):
        print("❌ No trade log found. Run 'deploy_active.py' first.")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Identify OPEN trades
    if 'Status' not in df.columns: df['Status'] = 'OPEN'
    open_indices = df[df['Status'] == 'OPEN'].index
    
    if len(open_indices) == 0:
        print("✅ No open trades to manage.")
        print_account_summary(df)
        return

    # 3. Download Live Data (FIXED: Threads=True prevents hanging)
    tickers = [f"{t}.NS" if not t.endswith('.NS') else t for t in df.loc[open_indices, 'Ticker'].unique()]
    print(f"🌍 Fetching live data for {len(tickers)} positions...")
    
    try:
        # ✅ FIX: threads=True (Parallel downloads) + progress=True (Visual feedback)
        data_download = yf.download(tickers, period="5d", progress=True, auto_adjust=True, threads=True)
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return

    print("\n" + "="*135)
    print(f"{'TICKER':<12} {'ENTRY':<10} {'CURRENT':<10} {'QTY':<5} {'INVESTED':<12} {'PnL (INR)':<12} {'DAYS':<5} {'ACTION / STATUS'}")
    print("="*135)

    changes_made = False

    # 4. Process Each Trade
    for idx in open_indices:
        row = df.loc[idx]
        ticker = row['Ticker']
        
        # Skip Ignored Tickers
        if ticker in IGNORE_TICKERS: continue

        yf_ticker = f"{ticker}.NS" if not ticker.endswith('.NS') else ticker
        
        # --- Get Current Price ---
        current_price = None
        try:
            if not data_download.empty:
                # Handle Multi-Index (Multiple Tickers) vs Single Index (One Ticker)
                if isinstance(data_download.columns, pd.MultiIndex):
                    if yf_ticker in data_download['Close'].columns:
                        current_price = data_download['Close'][yf_ticker].iloc[-1]
                    elif ticker in data_download['Close'].columns:
                        current_price = data_download['Close'][ticker].iloc[-1]
                else:
                    # Fallback for single ticker dataframe
                    current_price = data_download['Close'].iloc[-1]
        except:
            pass
        
        # Safety Fallback
        if current_price is None or pd.isna(current_price):
            current_price = float(row['Entry_Price'])
            print(f"{ticker:<12} ⚠️ No Data - Using Entry Price")
            continue

        # --- Metrics ---
        entry = float(row['Entry_Price'])
        
        # Recover Quantity if missing
        if 'Quantity' in row and row['Quantity'] > 0:
            qty = float(row['Quantity'])
        else:
            qty = int(float(row.get('Allocation_INR', 10000)) // entry)
            
        invested_amt = entry * qty
        current_val = current_price * qty
        pnl_inr = current_val - invested_amt
        
        # --- Date Logic ---
        try:
            trade_date = pd.to_datetime(row['Date']).tz_localize(None)
        except:
            trade_date = pd.to_datetime(row['Date'])
        days_held = (datetime.now() - trade_date).days

        # --- Strategy Logic ---
        atr_val = row['ATR_Pct'] * entry if 'ATR_Pct' in row and not pd.isna(row['ATR_Pct']) else entry * 0.02 
        
        action = "HOLD"
        
        # 🛑 EXECUTION LOGIC 🛑
        
        # Rule 1: TIME STOP
        if days_held >= TIME_STOP_DAYS:
            if (current_price - entry) < (MIN_PROFIT_TO_SURVIVE * atr_val):
                action = "🔴 SOLD (Time Stop)"
                df.at[idx, 'Status'] = 'CLOSED'
                df.at[idx, 'Exit_Price'] = current_price
                df.at[idx, 'Exit_Date'] = datetime.now().strftime("%Y-%m-%d")
                df.at[idx, 'PnL'] = pnl_inr
                changes_made = True

        # Rule 2: HARD STOP LOSS
        if 'Stop_Loss' in row and row['Stop_Loss'] > 0:
            if current_price < row['Stop_Loss']:
                action = "🛑 SOLD (Stop Loss Hit)"
                df.at[idx, 'Status'] = 'CLOSED'
                df.at[idx, 'Exit_Price'] = current_price
                df.at[idx, 'Exit_Date'] = datetime.now().strftime("%Y-%m-%d")
                df.at[idx, 'PnL'] = pnl_inr
                changes_made = True

        # Rule 3: SCALE OUT (Visual)
        if (current_price - entry) >= (ATR_MULTIPLIER_SCALE_OUT * atr_val):
             action = "🟢 PROFIT TARGET (Sell 50%)"

        # Formatting
        invested_str = f"₹{invested_amt:,.0f}"
        pnl_str = f"₹{pnl_inr:,.0f}"
        
        print(f"{ticker:<12} {entry:<10.2f} {current_price:<10.2f} {int(qty):<5} {invested_str:<12} {pnl_str:<12} {days_held:<5} {action}")

    print("="*135)

    # 5. Save Updates
    if changes_made:
        df.to_csv(LOG_FILE, index=False)
        print("💾 UPDATED CSV: Closed positions marked as 'CLOSED'.")
    else:
        print("✅ No Sell Actions triggered tonight.")

    print_account_summary(df)

def print_account_summary(df):
    # Calculate Realized PnL (Closed Trades)
    if 'Status' in df.columns and 'PnL' in df.columns:
        closed_trades = df[df['Status'] == 'CLOSED']
        realized_pnl = closed_trades['PnL'].sum()
    else:
        realized_pnl = 0.0
    
    current_balance = INITIAL_CAPITAL + realized_pnl
    
    # Calculate Buying Power
    open_trades = df[df['Status'] == 'OPEN']
    if not open_trades.empty:
        invested_capital = (open_trades['Entry_Price'] * open_trades['Quantity']).sum()
    else:
        invested_capital = 0.0

    buying_power = current_balance - invested_capital

    print(f"\n🏦 ACCOUNT DASHBOARD (Starting Cap: ₹{INITIAL_CAPITAL:,.0f})")
    print(f"   💸 Realized PnL:       ₹{realized_pnl:,.2f}")
    print(f"   💳 Account Balance:    ₹{current_balance:,.2f}")
    print(f"   📉 Locked Capital:     ₹{invested_capital:,.2f}")
    print(f"   🔓 BUYING POWER:       ₹{buying_power:,.2f}")
    print("="*135)

if __name__ == "__main__":
    manage_portfolio_nightly()