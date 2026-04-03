import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# We check MCX because you mentioned it specifically. 
# You can change this to any other ticker if needed.
TARGET_FILE = "lake/gold/MCX_processed.csv"

def check_stock_health():
    # 1. Resolve Path (Handles running from 'src' or root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Smart check: If we are in 'src', go up one level to find 'lake' if needed
    # (Adjust this logic based on where your 'lake' folder actually sits)
    if os.path.basename(base_dir) == "src":
        # Hypothesis 1: lake is inside src (src/lake/gold)
        path_variant_1 = os.path.join(base_dir, TARGET_FILE)
        # Hypothesis 2: lake is sibling to src (../lake/gold)
        path_variant_2 = os.path.join(os.path.dirname(base_dir), TARGET_FILE)
    else:
        path_variant_1 = os.path.join(base_dir, TARGET_FILE)
        path_variant_2 = path_variant_1

    if os.path.exists(path_variant_1):
        file_path = path_variant_1
    elif os.path.exists(path_variant_2):
        file_path = path_variant_2
    else:
        print(f"❌ Error: Could not find file. Checked:\n  - {path_variant_1}\n  - {path_variant_2}")
        return

    print(f"🔍 Inspecting: {file_path}")
    
    # 2. Load Data
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if df.empty:
        print("❌ File is empty.")
        return

    # 3. The Comparison Test
    last_row_file = df.iloc[-1]
    last_date_file = last_row_file.name
    
    # Simulate the model's cleaning process (dropping NaNs)
    df_clean = df.dropna()
    
    if df_clean.empty:
        print("❌ CRITICAL: The entire file is effectively empty after dropping NaNs!")
        print("   This means EVERY row has at least one missing value.")
        return

    last_row_model = df_clean.iloc[-1]
    last_date_model = last_row_model.name

    print(f"\n📅 LATEST DATES DETECTED:")
    print(f"   📂 File actually has data until:  {last_date_file}")
    print(f"   🤖 Model is choosing to use:      {last_date_model}")

    # 4. The Verdict
    if last_date_file != last_date_model:
        print("\n🚨 DIAGNOSIS: SILENT FAILURE DETECTED")
        print(f"   The model is IGNORING the new data from {last_date_file}.")
        print("   Reason: The last row contains missing values (NaNs).")
        
        # Identify the culprits
        print("\n   ⚠️  The following columns are empty in the last row:")
        missing_cols = last_row_file[last_row_file.isna()].index.tolist()
        for col in missing_cols:
            print(f"      - {col}")
            
        print("\n   💡 FIX: These features (likely lags) need more history or failed to calculate.")
    else:
        print("\n✅ DIAGNOSIS: HEALTHY")
        print("   The model is successfully using the very latest data row.")
        print("   If the score didn't change, the market conditions are genuinely similar to the previous prediction.")

if __name__ == "__main__":
    check_stock_health()