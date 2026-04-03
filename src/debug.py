import pandas as pd
import glob
import os

# Find one file
files = glob.glob("lake/silver/*.csv")
if files:
    test_file = files[0]
    print(f"--- 🕵️ INSPECTING: {os.path.basename(test_file)} ---")
    
    # Read raw
    df = pd.read_csv(test_file)
    print("\nCOLUMNS FOUND:")
    print(df.columns.tolist())
    
    print("\nFIRST 3 ROWS:")
    print(df.head(3))
else:
    print("No CSV files found!")