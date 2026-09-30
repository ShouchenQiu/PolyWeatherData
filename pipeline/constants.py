"""Shared constants for the ICDM Polymarket weather study."""
from __future__ import annotations

# Integrated canonical window: every node's artifact folder is named after it.
ARTIFACT = "20251205_20260810"
START_DATE = "2025-12-05"
CUTOFF = "2026-08-10"

# Chronological split boundaries (see paper Section III-B).
TRAIN_END = "2026-05-31"
VALID_END = "2026-06-30"
TEST_END = "2026-07-31"

# Horizon grid: hours before the daily temperature peak (the argmax time).
HORIZONS = [48, 42, 36, 30, 24, 18, 12, 9, 6, 3, 1]
HOR_COLS = [f"p_{h}h" for h in HORIZONS]

# Prices are the mean of all taker trades inside +/- this window of a horizon.
WIN_HALF_H = 0.5

# Price ranges defining the two tails (paper Section IV-B).
LONGSHOT_LO, LONGSHOT_HI = 0.0, 0.10
FAVORITE_LO, FAVORITE_HI = 0.75, 0.95

# City -> station directory holding its CLOB artifact.
NODES: dict[str, str] = {
    "NYC": "KLGA",
    "LONDON": "LONDON",
    "CHICAGO": "CHICAGO",
    "DALLAS": "DALLAS",
    "MIAMI": "MIAMI",
    "DENVER": "DENVER",
    "SEATTLE": "SEATTLE",
    "ATLANTA": "ATLANTA",
    "TOKYO": "TOKYO",
    "SHANGHAI": "SHANGHAI",
    "SEOUL": "SEOUL",
    "TORONTO": "TORONTO",
}

# Two nodes are excluded from the analysis as outliers (paper Section IV).
OUTLIERS = ["DENVER", "DALLAS"]

# Trade fee model: Fee = Theta * C * p * (1 - p).
FEE_RATE = 0.05
NOTIONAL = 10.0
