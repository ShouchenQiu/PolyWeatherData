"""Core domain types shared across strategy, exchange, and risk modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """A trading signal produced by a strategy.

    Fields
    ------
    token_id:
        Polymarket CLOB token ID (YES or NO share).
    side:
        ``"BUY"`` or ``"SELL"``.
    mc_prob:
        Model-derived probability for the YES outcome (0..1).
    clob_price:
        CLOB mid-price at signal generation time (0..1).
    edge:
        ``mc_prob - clob_price``; positive means the model thinks the
        current price is too low.
    bucket_lo_f:
        Lower bound of the temperature bucket in degF.
    bucket_hi_f:
        Upper bound of the temperature bucket in degF.
    condition_id:
        Polymarket condition/event ID.  Optional metadata.
    """
    token_id:     str
    side:         str           # "BUY" | "SELL"
    mc_prob:      float
    clob_price:   float
    edge:         float
    bucket_lo_f:      float        = float("-inf")
    bucket_hi_f:      float        = float("inf")
    condition_id:     str          = ""
    executable_price: float | None = None


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


@dataclass
class Order:
    """Represents a placed or simulated trade order.

    Fields
    ------
    order_id:
        Exchange-assigned or locally-generated order identifier.
    token_id:
        Polymarket CLOB token ID.
    side:
        ``"BUY"`` or ``"SELL"``.
    size_usd:
        Nominal USDC size of the order.
    price:
        Order price (0..1, Polymarket binary market convention).
    status:
        ``"pending"``, ``"filled"``, or ``"cancelled"``.
    filled_price:
        Actual fill price (set after fill confirmation).
    """
    order_id:     str
    token_id:     str
    side:         str
    size_usd:     float
    price:        float
    status:       str   = "pending"
    filled_price: Optional[float] = None
