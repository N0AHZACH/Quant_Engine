import numpy as np

# ======================================================
# CONFIG (RELAXED FOR SINGLE-ASSET + DEBUG)
# ======================================================
MIN_EDGE = 0.3              # lowered so trades actually happen
MAX_POSITIONS = 10
RISK_PER_TRADE = 0.005      # 0.5% of capital
MAX_GROSS_EXPOSURE = 1.2    # 120%


def generate_positions(edges, vols, capital):
    """
    Convert model edges into position sizes.

    Parameters
    ----------
    edges : dict[symbol -> edge]
    vols  : dict[symbol -> volatility]
    capital : float

    Returns
    -------
    dict[symbol -> position_size]
    """

    # --------------------------------------------------
    # 1. CLEAN INPUTS
    # --------------------------------------------------
    clean = {
        s: e for s, e in edges.items()
        if s in vols and np.isfinite(e) and np.isfinite(vols[s])
    }

    if not clean:
        return {}

    # --------------------------------------------------
    # 2. EDGE GATE (RELAXED)
    # --------------------------------------------------
    filtered = {
        s: e for s, e in clean.items()
        if abs(e) >= MIN_EDGE
    }

    if not filtered:
        return {}

    # --------------------------------------------------
    # 3. CROSS-SECTIONAL NORMALIZATION
    # (DISABLED FOR SINGLE-ASSET)
    # --------------------------------------------------
    symbols = list(filtered.keys())
    edge_vals = np.array([filtered[s] for s in symbols])

    if len(edge_vals) > 1:
        edge_vals = edge_vals - edge_vals.mean()

    # --------------------------------------------------
    # 4. SELECT STRONGEST SIGNALS
    # --------------------------------------------------
    order = np.argsort(-np.abs(edge_vals))[:MAX_POSITIONS]

    # --------------------------------------------------
    # 5. RISK-BASED POSITION SIZING
    # --------------------------------------------------
    positions = {}
    gross = 0.0

    for i in order:
        sym = symbols[i]
        vol = max(vols[sym], 1e-6)

        risk_dollars = capital * RISK_PER_TRADE
        size = (risk_dollars / vol) * np.sign(edge_vals[i])

        if size == 0:
            continue

        positions[sym] = size
        gross += abs(size)

    # --------------------------------------------------
    # 6. GROSS EXPOSURE CAP
    # --------------------------------------------------
    max_allowed = capital * MAX_GROSS_EXPOSURE
    if gross > max_allowed and gross > 0:
        scale = max_allowed / gross
        for s in positions:
            positions[s] *= scale

    return positions
