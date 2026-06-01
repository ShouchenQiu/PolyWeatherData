"""Core domain types shared across strategy, exchange, and risk modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectionalSignal:
    """Strategy output representing a directional mispricing for one bucket.

    Contains both YES and NO token IDs so TradingLoop can arbitrate
    SELL YES vs BUY NO after evaluating inventory.

    direction='bullish' means YES is underpriced (model_prob > reference_price).
    direction='bearish' means YES is overpriced (reference_price > model_prob).
    edge is always positive (abs of model_prob - reference_price).
    reference_price is the YES BUY VWAP from compute_quote at signal time.
    """
    yes_token_id:    str
    no_token_id:     str
    direction:       str    # "bullish" | "bearish"
    model_prob:      float
    reference_price: float
    edge:            float
    bucket_lo_f:     float = float("-inf")
    bucket_hi_f:     float = float("inf")
    condition_id:    str   = ""


@dataclass
class OrderIntent:
    """Execution/risk input produced by TradingLoop after directional arbitration.

    token_id and side are resolved from DirectionalSignal + inventory state.
    executable_price is the VWAP from compute_quote at the time of creation.
    size_usd is None before RiskPipeline.compute_size() resolves it.
    """
    token_id:         str
    side:             str           # "BUY" | "SELL"
    size_usd:         float | None
    executable_price: float
    edge:             float
    bucket_lo_f:      float = float("-inf")
    bucket_hi_f:      float = float("inf")
    condition_id:     str   = ""
