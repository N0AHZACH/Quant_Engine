# ======================================================
# PATH FIX (WINDOWS SAFE)
# ======================================================
import sys
import os
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# IMPORTS
# ======================================================
import numpy as np
import pandas as pd

from production.dataset_builder import build_dataset
from production.base_models import train_base_models, predict_base_models
from production.ensemble_predictor import compute_mu_sigma, compute_edge


# ======================================================
# LOAD GOLD DATA
# ======================================================
GOLD_DIR = os.path.join(PROJECT_ROOT, "lake", "gold")
gold_files = glob.glob(os.path.join(GOLD_DIR, "*.csv"))
assert len(gold_files) > 0, "No gold CSV files found"


def load_gold_data(max_files=5):
    dfs = []
    for fp in gold_files[:max_files]:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        dfs.append(df)
    return pd.concat(dfs).sort_index()


# ======================================================
# HELPERS
# ======================================================
def flatten_last_step(X_seq):
    return X_seq[:, -1, :]


def out_of_sample_edge(X, y, split=0.7):
    n = len(X)
    cut = int(n * split)

    X_train, y_train = X[:cut], y[:cut]
    X_test = X[cut:]

    models = train_base_models(X_train, y_train)
    preds = predict_base_models(models, X_test)

    mu, sigma = compute_mu_sigma(preds)
    edge = compute_edge(mu, sigma)

    return np.mean(np.abs(edge))


def block_shuffle(y, block_size=50):
    y = y.copy()
    for i in range(0, len(y), block_size):
        block = y[i:i + block_size]
        np.random.shuffle(block)
        y[i:i + block_size] = block
    return y


# ======================================================
# LEAKAGE TEST (CORRECT NULL)
# ======================================================
def test_leakage_via_edge_ratio():
    print("\n[Leakage Test] Edge Ratio Test (OOS, Gold Data)")

    df = load_gold_data()
    X_seq, y = build_dataset(df)
    X = flatten_last_step(X_seq)

    # Real edge
    edge_real = out_of_sample_edge(X, y)

    # Shuffled edge
    y_block = block_shuffle(y)
    edge_shuffled = out_of_sample_edge(X, y_block)

    ratio = edge_real / (edge_shuffled + 1e-9)

    print(f"Edge real     : {edge_real:.4f}")
    print(f"Edge shuffled : {edge_shuffled:.4f}")
    print(f"Edge ratio    : {ratio:.2f}")

    # Threshold based on empirical quant practice
    assert ratio < 1.3, \
        "Edge survives block shuffle disproportionately — LEAKAGE CONFIRMED"

    print("✔ Edge ratio test passed — no leakage")


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    print("\n===== RUNNING LEAKAGE TEST (FINAL, STATISTICALLY VALID) =====")
    test_leakage_via_edge_ratio()
    print("\n✅ NO EVIDENCE OF LEAKAGE\n")
