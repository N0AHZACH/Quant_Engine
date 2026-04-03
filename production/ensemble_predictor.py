import numpy as np

"""
Ensemble prediction utilities.

This module produces:
- Expected return (mu)
- Uncertainty (sigma)
- Robust, normalized edge score

This version is FAT-TAIL SAFE and RF/XGB SAFE.
"""


def compute_mu_sigma(predictions: dict) -> tuple:
    """
    Computes expected return (mu) and uncertainty (sigma)
    from ensemble predictions.

    Parameters
    ----------
    predictions : dict
        {model_name: np.ndarray of predictions}

    Returns
    -------
    mu : np.ndarray
        Mean prediction across models
    sigma : np.ndarray
        Standard deviation across models (uncertainty proxy)
    """

    assert isinstance(predictions, dict), "Predictions must be a dict"
    assert len(predictions) >= 2, "At least 2 models required for uncertainty"

    preds = np.vstack(list(predictions.values()))

    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0) + 1e-6  # prevent divide-by-zero

    return mu, sigma


def compute_edge(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Computes a ROBUST, NORMALIZED edge score.

    Raw edge = mu / sigma
    Final edge = robust z-score using Median Absolute Deviation (MAD)

    This prevents fat tails from tree-based models and
    makes edge variance statistically meaningful.
    """

    assert mu.shape == sigma.shape, "mu and sigma shape mismatch"

    # Add floor to sigma to prevent division by tiny values
    # When models agree closely, sigma → 0, causing raw_edge to explode
    sigma_floor = 0.6 * sigma.mean()
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Raw risk-adjusted signal
    raw_edge = mu / sigma_safe
    
    # Store absolute scale before normalization
    # Cap at 1.0: when raw signals are strong (scale > 1), use normalized edge
    # When uncertainty is high (scale < 1), suppress edge proportionally
    raw_edge_scale = min(np.mean(np.abs(raw_edge)), 1.0)

    # Robust normalization (Median Absolute Deviation)
    median = np.median(raw_edge)
    mad = np.median(np.abs(raw_edge - median)) + 1e-6

    # Scale MAD by 1.4826 to match std normalization for normal distributions
    # This maintains robustness while ensuring std ≈ 1.0
    edge_normalized = (raw_edge - median) / (mad * 1.4826)
    
    # Winsorize extreme values to prevent outliers from dominating
    # Clip at ±1.5 to ensure no-trade days are possible
    edge_normalized = np.clip(edge_normalized, -1.5, 1.5)
    
    # Rescale to preserve suppression when uncertainty is high
    edge = edge_normalized * raw_edge_scale

    return edge
