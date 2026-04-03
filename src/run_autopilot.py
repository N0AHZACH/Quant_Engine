import subprocess
import sys
import time
from datetime import datetime
import os

def run_step(script_name, description):
    print(f"\n🚀 STARTING STEP: {description}...")
    start = time.time()
    
    script_path = os.path.join("src", script_name)
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Could not find {script_path}")
        sys.exit(1)

    try:
        # Runs the python script and waits for it to finish
        result = subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start
        print(f"✅ COMPLETED in {elapsed:.1f}s")
    except subprocess.CalledProcessError:
        print(f"❌ CRITICAL ERROR in {script_name}. Autopilot Aborted.")
        sys.exit(1)

def main():
    print("="*60)
    print(f"🤖 TRADING BOT AUTOPILOT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # 1. DOWNLOAD RAW DATA
    run_step("fetch_all.py", "Fetching Today's Close Prices")
    
    # 2. FILTER & REFINE (Quality Control)
    run_step("make_gold_filtered.py", "Removing Junk Stocks & Calculating AI Features")
    
    # 3. PREDICT (The 5-Head Brain)
    run_step("scan_for_tomorrow.py", "Scanning for High-Confidence Trades")
    
    print("\n" + "="*60)
    print("🏁 MISSION COMPLETE. CHECK RESULTS ABOVE.")
    print("="*60)
    
    # Keep window open so you can read the buy signals
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()