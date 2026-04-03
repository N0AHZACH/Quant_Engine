import pandas as pd
import yfinance as yf
import glob
import os
import subprocess
import sys
from datetime import datetime, timedelta
from tqdm import tqdm

# --- CONFIGURATION ---
GOLD_SCRIPT_NAME = "make_gold.py"

def get_project_root():
    """
    Smartly finds the root folder (Quant_Engine).
    If this script is in 'src', it goes up one level.
    """
    script_location = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we are inside a 'src' folder
    if os.path.basename(script_location) == "src":
        return os.path.dirname(script_location) # Go up to parent
    return script_location

def get_clean_ticker(filename):
    base = os.path.basename(filename).replace(".csv", "")
    if base.lower().endswith("_processed"):
        base = base[:-10]
    if not base.endswith(".NS"):
        base = f"{base}.NS"
    return base

def update_silver_lake():
    print(f"\n--- 1. 🔄 UPDATING SILVER LAKE (RAW OHLCV) ---")
    
    # 1. Locate the Silver folder relative to Project Root
    root_dir = get_project_root()
    silver_path = os.path.join(root_dir, "lake", "silver")

    if not os.path.exists(silver_path):
        print(f"❌ Error: Silver folder not found at: {silver_path}")
        print(f"   Please check your folder structure.")
        return False

    files = glob.glob(f"{silver_path}/*.csv")
    if not files:
        print(f"❌ No files found in {silver_path}.")
        return False

    print(f"Found {len(files)} stocks in Silver. Checking for updates...")
    
    success_count = 0
    
    for file_path in tqdm(files, desc="Fetching Prices"):
        try:
            # Read existing data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df.empty: continue
                
            ticker_symbol = get_clean_ticker(file_path)
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            
            if last_date >= today:
                continue 
            
            # Download missing days
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = yf.download(ticker_symbol, start=start_date, progress=False, auto_adjust=True, multi_level_index=False)
            
            if not new_data.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if set(required_cols).issubset(new_data.columns):
                    new_data = new_data[required_cols]
                    updated_df = pd.concat([df, new_data])
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    updated_df.to_csv(file_path)
                    success_count += 1
                
        except Exception:
            continue

    print(f"✅ Silver Update Complete. {success_count} files updated.")
    return True

def trigger_gold_production():
    print(f"\n--- 2. 🏭 STARTING GOLD PRODUCTION (FEATURE ENGINEERING) ---")
    
    # Locate make_gold.py in the SAME folder as this script (src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gold_script_path = os.path.join(script_dir, GOLD_SCRIPT_NAME)
    
    if not os.path.exists(gold_script_path):
        print(f"❌ Error: Could not find '{GOLD_SCRIPT_NAME}' at: {gold_script_path}")
        return

    print(f"🚀 Launching {GOLD_SCRIPT_NAME}...")
    
    try:
        # Run make_gold.py
        # We assume make_gold.py relies on the current working directory (Quant_Engine)
        # to find 'lake/silver', which is correct based on your terminal prompt.
        subprocess.run([sys.executable, gold_script_path], check=True)
        print(f"\n✅ PIPELINE COMPLETE. Gold Lake is fully engineered.")
        
    except subprocess.CalledProcessError:
        print(f"❌ Error: {GOLD_SCRIPT_NAME} failed to run.")

if __name__ == "__main__":
    if update_silver_lake():
        trigger_gold_production()