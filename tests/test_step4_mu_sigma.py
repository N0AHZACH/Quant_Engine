# ======================================================
# PATH FIX (CRITICAL FOR WINDOWS)
# ======================================================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

"""
STEP 4 VERIFICATION SCRIPT
Tests expectation (mu) + uncertainty (sigma) pipeline.

PASS = Step 4 is DONE
FAIL = Stop immediately, do NOT proceed
"""

import numpy as np

from production.base_models import train_base_models, predict_base_models
from production.ensemble_predictor import compute_mu_sigma, compute_edge


# ============================================================
# TEST 1 — SHAPE & SANITY TEST
# ============================================================

def test_shapes_and_sanity():
    np.random.seed(42)

    X_train = np.random.randn(500, 10)
    y_train = np.random.randn(500)
    X_test = np.random.randn(100, 10)

    models = train_base_models(X_train, y_train)
    preds = predict_base_models(models, X_test)

    mu, sigma = compute_mu_sigma(preds)
    edge = compute_edge(mu, sigma)

    assert mu.shape == (100,), "mu shape incorrect"
    assert sigma.shape == (100,), "sigma shape incorrect"
    assert edge.shape == (100,), "edge shape incorrect"

    assert np.all(np.isfinite(mu)), "mu contains NaN or Inf"
    assert np.all(np.isfinite(sigma)), "sigma contains NaN or Inf"
    assert np.all(sigma > 0), "sigma must be strictly positive"

    print("✔ TEST 1 PASSED — Shapes & sanity OK")


# ============================================================
# TEST 2 — MODEL DISAGREEMENT TEST
# ============================================================

def test_model_disagreement():
    preds = {
        "model_a": np.ones(100),
        "model_b": -np.ones(100)
    }

    mu, sigma = compute_mu_sigma(preds)

    assert abs(mu.mean()) < 1e-6, "mu should collapse to ~0"
    assert sigma.mean() > 0.8, "sigma should be large for disagreement"

    print("✔ TEST 2 PASSED — Disagreement inflates uncertainty")


# ============================================================
# TEST 3 — REGIME SENSITIVITY TEST
# ============================================================

def test_regime_sensitivity():
    np.random.seed(0)

    # Low volatility environment
    X_train_lv = np.random.randn(500, 10) * 0.5
    y_train_lv = np.random.randn(500) * 0.5
    X_test_lv = np.random.randn(200, 10) * 0.5

    models_lv = train_base_models(X_train_lv, y_train_lv)
    preds_lv = predict_base_models(models_lv, X_test_lv)
    _, sigma_lv = compute_mu_sigma(preds_lv)

    # High volatility environment
    X_train_hv = np.random.randn(500, 10) * 3.0
    y_train_hv = np.random.randn(500) * 3.0
    X_test_hv = np.random.randn(200, 10) * 3.0

    models_hv = train_base_models(X_train_hv, y_train_hv)
    preds_hv = predict_base_models(models_hv, X_test_hv)
    _, sigma_hv = compute_mu_sigma(preds_hv)

    assert sigma_hv.mean() > sigma_lv.mean(), \
        "sigma must be higher in volatile regimes"

    print("✔ TEST 3 PASSED — Sigma reacts to regime volatility")


# ============================================================
# TEST 4 — EDGE COLLAPSE TEST (STEP-4 CORRECT VERSION)
# ============================================================

def test_edge_collapse():
    np.random.seed(123)

    X_train = np.random.randn(500, 10)
    y_train = np.random.randn(500)
    X_test = np.random.randn(300, 10)

    models = train_base_models(X_train, y_train)
    preds = predict_base_models(models, X_test)

    mu, sigma = compute_mu_sigma(preds)
    edge = compute_edge(mu, sigma)

    median_edge = np.median(edge)
    edge_std = np.std(edge)

    # Step-4 guarantees: bounded, not zero-mean yet
    assert np.abs(median_edge) < 0.5, \
        "Edge should be modest before cross-sectional ranking"

    assert edge_std < 2.5, \
        "Edge variance too high — potential leakage"

    print("✔ TEST 4 PASSED — Edge magnitude and variance acceptable")


# ============================================================
# TEST 5 — NO-TRADE DAY TEST
# ============================================================

def test_no_trade_day():
    np.random.seed(99)

    X_train = np.random.randn(500, 10)
    y_train = np.random.randn(500)
    X_test = np.random.randn(100, 10)

    models = train_base_models(X_train, y_train)
    preds = predict_base_models(models, X_test)

    mu, sigma = compute_mu_sigma(preds)

    # Artificially inflate uncertainty
    sigma *= 10
    edge = compute_edge(mu, sigma)

    trade_threshold = 0.5
    trades = edge[np.abs(edge) > trade_threshold]

    assert len(trades) == 0, "No-trade day violated"

    print("✔ TEST 5 PASSED — System allows zero-trade days")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("\n===== RUNNING STEP 4 VERIFICATION =====\n")

    test_shapes_and_sanity()
    test_model_disagreement()
    test_regime_sensitivity()
    test_edge_collapse()
    test_no_trade_day()

    print("\n✅ ALL STEP 4 TESTS PASSED — STEP 4 IS COMPLETE\n")
