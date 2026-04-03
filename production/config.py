"""
Global system configuration.
This file MUST NOT be modified without restarting the entire research cycle.
"""

# ===============================
# SYSTEM IDENTITY
# ===============================

SYSTEM_NAME = "NSE Cross-Sectional Swing Alpha"

ASSET_UNIVERSE = "NSE_EQUITIES_ONLY"

# ===============================
# PREDICTION SETTINGS
# ===============================

# FIXED holding period (DO NOT CHANGE)
PREDICTION_HORIZON = 3  # days

# ===============================
# RISK CONSTRAINTS
# ===============================

MAX_DRAWDOWN_TARGET = 0.20  # 20%
ALLOW_ZERO_TRADE_DAYS = True

# ===============================
# EXECUTION PLACEHOLDER
# (execution modeling comes later)
# ===============================

ASSUMED_ENTRY = "t"
ASSUMED_EXIT = "t + 3"
