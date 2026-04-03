import numpy as np
import pandas as pd
import glob
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import multiprocessing

# --- CONFIGURATION ---
INPUT_FOLDER = "lake/gold"
OUTPUT_FILE = "lake/processed_market.npz"
SEQ_LEN = 60
FEATURES = 18

# Try to import your Elite Features engine
try:
    from features import add_institutional_features 
except ImportError:
    print("❌ Critical Error: 'features.py' not found in src folder.")
    sys.exit()

def preprocess_all():
    # 1. Gather files
    files = glob.glob(f"{INPUT_FOLDER}/*.csv")
    if not files:
        print(f"❌ No CSV files found in {INPUT_FOLDER}")
        return

    print(f"🌍 Found {len(files)} stocks. Starting 'Alpha Factory' processing...")
    
    # 2. Master lists for compiled data
    master_data = []
    valid_start_indices = []
    current_global_idx = 0
    
    # 3. Process each file
    pbar = tqdm(files, desc="Compiling Market Data")
    
    for f in pbar:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(df) < 300: continue
            
            # 1. GENERATE ELITE FEATURES 
            df = add_institutional_features(df)
            
            # 2. NORMALIZE
            data = df.values.astype(np.float32)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            # 3. STORE DATA (No windowing yet!)
            rows = len(data_scaled)
            master_data.append(data_scaled)
            
            # 4. CALCULATE VALID WINDOWS
            # We must only train on windows that DO NOT cross stock boundaries
            if rows > SEQ_LEN:
                # Valid sequences start from current_global_idx up to (rows - SEQ_LEN) rows before the end
                start = current_global_idx
                end = current_global_idx + rows - SEQ_LEN
                
                # Append range of valid start points
                valid_start_indices.extend(range(start, end))
            
            current_global_idx += rows
            
        except Exception as e: 
            # Catch errors like missing columns or bad data in a specific file
            continue

    if not master_data:
        print("❌ Processing failed. No valid data compiled.")
        return

    # 4. FINAL COMPILATION AND SAVE
    print("\n📦 Concatenating massive array...")
    full_array = np.concatenate(master_data, axis=0)
    indices_array = np.array(valid_start_indices, dtype=np.int32)
    
    # 5. DEFENSIVE CHECK (Checking for NaNs/Infs before saving)
    print("🛡️ Running Defensive Data Check...")
    if np.isnan(full_array).any() or np.isinf(full_array).any():
        print("❌ CRITICAL FAILURE: Found NaN or Inf values in processed data. Stopping.")
        sys.exit()
    print("✅ Data passed defensive checks.")
    
    # 6. STATS
    size_mb = full_array.nbytes / (1024 * 1024)
    print(f"\n✅ COMPLETED!")
    print(f"   Total Rows:      {len(full_array):,}")
    print(f"   Training Samples:{len(indices_array):,}")
    print(f"   Dataset Size:    {size_mb:.2f} MB")
    
    # Save Compressed
    np.savez_compressed(OUTPUT_FILE, data=full_array, indices=indices_array)
    print(f"💾 Saved to:        {OUTPUT_FILE}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    preprocess_all()