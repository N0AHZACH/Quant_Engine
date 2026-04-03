import pandas as pd
import glob
import os
import pandas_ta as ta  # pip install pandas_ta
from datetime import datetime

# --- CONFIGURATION ---
INITIAL_CAPITAL = 10000.0       # 💰 Total Account Size
MAX_TRADES = 5                  # Max open positions
ALLOCATION_PER_TRADE = INITIAL_CAPITAL / MAX_TRADES 
LOG_FILE = "paper_trading_log.csv"
DATA_LAKE = "lake/gold/*.csv"   # Where your stock data lives

# --- STRATEGY SETTINGS ---
MIN_RSI = 40            # Buy if RSI < 40 (Oversold Dip)
TREND_SMA = 200         # Buy if Price > 200 SMA (Uptrend)

# --- JUNK FILTERS ---
MIN_PRICE = 20.0        # Skip stocks cheaper than ₹20 (Penny stocks)
MIN_TURNOVER = 5000000  # Skip if Daily Turnover < ₹50 Lakhs (Illiquid)

def get_wallet_status():
    if not os.path.exists(LOG_FILE):
        return [], INITIAL_CAPITAL, []

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    except:
        return [], INITIAL_CAPITAL, []
    
    # 1. Realized PnL
    realized_pnl = df[df['Status'] == 'CLOSED']['PnL'].sum() if 'PnL' in df.columns else 0.0

    # 2. Open Trades
    if 'Status' not in df.columns: df['Status'] = 'OPEN'
    open_trades = df[df['Status'] == 'OPEN']
    open_tickers = open_trades['Ticker'].unique().tolist()
    
    # 3. Invested Capital
    used_capital = 0.0
    if not open_trades.empty:
        if 'Entry_Price' in df.columns and 'Quantity' in df.columns:
            used_capital = (open_trades['Entry_Price'] * open_trades['Quantity']).sum()
    
    buying_power = (INITIAL_CAPITAL + realized_pnl) - used_capital
    closed_tickers = df[df['Status'] == 'CLOSED']['Ticker'].tail(5).tolist()
    
    return open_tickers, buying_power, closed_tickers

def analyze_stock(filepath):
    """
    Returns (Is_Buy, Price, RSI, Trend, Reason, Is_Junk)
    """
    try:
        df = pd.read_csv(filepath)
        # Check Data Quality
        if len(df) < 205: 
            return False, 0.0, 0.0, "N/A", "Not Enough Data", True

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        current_price = df['Close'].iloc[-1]
        
        # --- 1. JUNK FILTER (Garbage Collection) ---
        # Calculate Turnover (Price * Avg Volume)
        avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
        turnover = current_price * avg_vol
        
        if current_price < MIN_PRICE:
            return False, current_price, 0.0, "DOWN", "Penny Stock", True
            
        if turnover < MIN_TURNOVER:
            return False, current_price, 0.0, "DOWN", "Illiquid", True

        # --- 2. STRATEGY INDICATORS ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        rsi = df['RSI'].iloc[-1]
        sma = df['SMA_200'].iloc[-1]
        
        is_uptrend = current_price > sma
        is_dip = rsi < MIN_RSI
        trend_str = "UP" if is_uptrend else "DOWN"

        if is_uptrend and is_dip:
            return True, current_price, rsi, trend_str, "Signal Confirmed", False
        
        return False, current_price, rsi, trend_str, "No Signal", False

    except:
        return False, 0.0, 0.0, "Error", "Corrupt File", True

def log_trade(ticker, entry_price, quantity):
    stop_loss = entry_price * 0.95 
    
    trade_data = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Action": "BUY",
        "Entry_Price": round(entry_price, 2),
        "Quantity": quantity,
        "Allocation_INR": round(entry_price * quantity, 2),
        "Stop_Loss": round(stop_loss, 2),
        "ATR_Pct": 0.02, 
        "Status": "OPEN",
        "Exit_Price": 0.0,
        "Exit_Date": None,
        "PnL": 0.0,
        "Return_Pct": 0.0
    }
    
    df = pd.DataFrame([trade_data])
    header = not os.path.isfile(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=header, index=False)

def deploy():
    print("🚀 STARTING MARKET SCANNER...")
    
    open_tickers, buying_power, closed_history = get_wallet_status()
    print(f"💰 BUYING POWER:  ₹{buying_power:,.2f}")
    
    files = glob.glob(DATA_LAKE)
    if not files:
        print(f"❌ NO DATA in {DATA_LAKE}")
        return
    
    print(f"🔍 Scanning {len(files)} stocks...")
    
    opportunities = []
    junk_count = 0
    
    # --- 1. SCANNING PHASE ---
    for filepath in files:
        raw_name = os.path.basename(filepath)
        ticker = raw_name.replace('.csv', '').replace('.NS', '').replace('_processed', '')
        
        is_buy, price, rsi, trend, reason, is_junk = analyze_stock(filepath)
        
        if is_junk:
            junk_count += 1
            continue  # 🗑️ SKIP IMMEDIATELY
        
        if is_buy:
            opportunities.append({
                'Ticker': ticker,
                'Price': price,
                'RSI': rsi,
                'Trend': trend,
                'Reason': reason
            })
            
    # --- 2. REPORTING PHASE ---
    print("\n" + "="*125)
    print(f"{'TICKER':<15} {'PRICE':<10} {'RSI':<6} {'TREND':<6} {'QTY':<8} {'COST (Est)':<12} {'ACTION':<18} {'REASON'}")
    print("="*125)

    if not opportunities:
        print("😴 No valid stocks met the criteria today.")
    else:
        for opp in opportunities:
            ticker = opp['Ticker']
            price = opp['Price']
            rsi = opp['RSI']
            trend = opp['Trend']
            
            # Theoretical Calculation
            target_spend = ALLOCATION_PER_TRADE
            theoretical_qty = int(target_spend // price)
            theoretical_cost = theoretical_qty * price
            
            qty = 0
            cost = 0.0
            action = "WATCH"
            reason = opp['Reason']
            
            # CHECK RULES
            if ticker in open_tickers:
                action = "SKIP"
                reason = "🔒 Already Own"
                qty = 0
                cost = 0
            elif ticker in closed_history:
                action = "SKIP"
                reason = "⏳ Recently Sold"
                qty = theoretical_qty
                cost = theoretical_cost
            elif len(open_tickers) >= MAX_TRADES:
                action = "SKIP (Full)"
                reason = "🚫 Portfolio Limit"
                qty = theoretical_qty
                cost = theoretical_cost
            elif buying_power < 1000:
                action = "SKIP (No Funds)"
                reason = "💸 Low Cash"
                qty = theoretical_qty
                cost = theoretical_cost
            else:
                # EXECUTE
                real_spend = min(ALLOCATION_PER_TRADE, buying_power)
                real_qty = int(real_spend // price)
                
                if real_qty > 0:
                    qty = real_qty
                    cost = qty * price
                    action = "✅ BUY"
                    reason = "Signal Confirmed"
                    
                    log_trade(ticker, price, qty)
                    buying_power -= cost
                    open_tickers.append(ticker)
                else:
                    action = "SKIP"
                    reason = "⚠️ Too Expensive"
                    qty = theoretical_qty
                    cost = theoretical_cost

            cost_str = f"₹{cost:,.0f}" if cost > 0 else "-"
            qty_str = str(qty) if qty > 0 else "-"
            
            print(f"{ticker:<15} {price:<10.2f} {rsi:<6.1f} {trend:<6} {qty_str:<8} {cost_str:<12} {action:<18} {reason}")

    print("="*125)
    print(f"💰 FINAL CASH:    ₹{buying_power:,.2f}")
    print(f"🗑️  JUNK SKIPPED:  {junk_count} stocks (Penny/Illiquid/Bad Data)")
    print("="*125)

if __name__ == "__main__":
    deploy()