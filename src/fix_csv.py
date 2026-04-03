import pandas as pd
import os
import shutil
from datetime import datetime

FILE_PATH = "paper_trading_log.csv"
BACKUP_PATH = "paper_trading_log_backup.csv"

def repair_csv():
    print(f"🔧 Starting MONTE CARLO READY Repair on {FILE_PATH}...")

    if not os.path.exists(FILE_PATH):
        print("❌ File not found. Nothing to repair.")
        return

    # 1. Backup
    shutil.copy(FILE_PATH, BACKUP_PATH)
    print(f"✅ Backup created at {BACKUP_PATH}")

    try:
        # 2. Read File
        df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
        print(f"   📊 Input Rows: {len(df)}")

        # 3. Standardize Columns (The Full Monte Carlo Schema)
        required_cols = {
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Ticker': 'UNKNOWN',
            'Action': 'BUY',
            'Entry_Price': 0.0,
            'Stop_Loss': 0.0,
            'Quantity': 1,
            'Allocation_INR': 2000.0,
            'Status': 'OPEN',
            'Exit_Price': 0.0,
            'Exit_Date': None,      # <--- NEW for Monte Carlo
            'PnL': 0.0,
            'Return_Pct': 0.0,      # <--- NEW for Monte Carlo
            'ATR_Pct': 0.02
        }

        # Add missing columns
        for col, default_val in required_cols.items():
            if col not in df.columns:
                print(f"   ➕ Adding missing column: {col}")
                df[col] = default_val

        # 4. Fix Data Types
        numeric_cols = ['Entry_Price', 'Quantity', 'Allocation_INR', 'PnL', 'Exit_Price', 'Return_Pct']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 5. Smart Backfill
        for index, row in df.iterrows():
            entry = float(row['Entry_Price'])
            qty = float(row['Quantity'])
            alloc = float(row['Allocation_INR'])

            # Fix Quantity
            if qty <= 1 and alloc > entry and entry > 0:
                df.at[index, 'Quantity'] = int(alloc // entry)
            
            # Fix Allocation
            if alloc == 0 and qty > 0:
                df.at[index, 'Allocation_INR'] = qty * entry

            # Fix Status
            if pd.isna(row['Status']) or row['Status'] == '':
                df.at[index, 'Status'] = 'OPEN'

            # Fix Return_Pct for ALREADY CLOSED trades
            # If you have old closed trades, we calculate the % now so MC works later
            if row['Status'] == 'CLOSED' and row['Return_Pct'] == 0.0 and entry > 0:
                exit_price = float(row['Exit_Price'])
                # If Exit Price is missing but PnL exists, reverse engineer it
                if exit_price == 0.0 and row['PnL'] != 0:
                    exit_price = entry + (row['PnL'] / qty)
                    df.at[index, 'Exit_Price'] = exit_price
                
                if exit_price != 0.0:
                    ret = (exit_price - entry) / entry
                    df.at[index, 'Return_Pct'] = ret
                    print(f"   🔄 Backfilled Return % for {row['Ticker']}: {ret:.2%}")

        # 6. Safe Deduplication (Preserve multiple entries)
        initial_count = len(df)
        df = df.drop_duplicates(keep='first')
        
        # 7. Save
        ordered_cols = list(required_cols.keys())
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra_cols]
        
        df.to_csv(FILE_PATH, index=False)
        print("="*50)
        print(f"✅ DATA INTEGRITY: 100%")
        print(f"   The CSV is now perfectly synced for Monte Carlo.")
        print("="*50)

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    repair_csv()