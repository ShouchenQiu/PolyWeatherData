"""Polymarket taker fee curve primitives."""
from __future__ import annotations

import math


def taker_fee_usdc(
    *,
    shares: float,
    price: float,
    fees_enabled: bool,
    fee_rate: float | None,
    fee_exponent: float | None = 1.0,
) -> float:
    """Return the protocol taker fee, rounded to five USDC decimals."""
    if not fees_enabled:
        return 0.0
    shares = float(shares)
    price = float(price)
    rate = float(fee_rate or 0.0)
    exponent = float(1.0 if fee_exponent is None else fee_exponent)
    if shares < 0:
        raise ValueError("shares must be non-negative")
    if not 0.0 <= price <= 1.0:
        raise ValueError("price must be between zero and one")
    if rate < 0 or exponent < 0:
        raise ValueError("fee rate and exponent must be non-negative")
    fee = shares * rate * math.pow(price * (1.0 - price), exponent)
    return round(fee, 5)
