"""L2 VWAP pricing primitive.

Leaf module: must not import from any other project package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class BookLevel(TypedDict):
    """A single price level in an L2 order book."""

    price: float
    size: float


class NormalizedBook(TypedDict):
    """Normalized L2 order book shape expected by ``compute_quote``.

    Both lists may be empty (compute_quote handles this defensively).
    compute_quote sorts the levels internally, so callers need not pre-sort.
    """

    bids: list[BookLevel]
    asks: list[BookLevel]


@dataclass
class Quote:
    """Result of walking L2 book depth for a given side and size."""

    vwap_price: float
    max_fillable_usd: float
    filled_shares: float
    slippage_bps: float
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    complete_fill: bool


@dataclass(frozen=True)
class BookMetrics:
    """Top-of-book metrics for display, mark prices, and observability."""

    best_bid: float | None
    best_ask: float | None
    mid_price: float | None
    spread: float | None
    depth_bid: float
    depth_ask: float

    def as_record(self) -> dict[str, float | None]:
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "vwap_price": None,
            "spread": self.spread,
            "depth_bid": self.depth_bid,
            "depth_ask": self.depth_ask,
        }


def compute_book_metrics(book: NormalizedBook) -> BookMetrics:
    """Extract display/mark metrics from a normalized L2 book.

    Uses the same sorting and mark-price semantics as ``compute_quote``:
    best bid is highest bid, best ask is lowest ask, and mid_price exists
    only when both sides are present.
    """
    bids: list[dict] = sorted(
        book.get("bids", []),
        key=lambda x: float(x["price"]),
        reverse=True,
    )
    asks: list[dict] = sorted(
        book.get("asks", []),
        key=lambda x: float(x["price"]),
    )

    best_bid: float | None = float(bids[0]["price"]) if bids else None
    best_ask: float | None = float(asks[0]["price"]) if asks else None
    mid_price: float | None = (
        (best_bid + best_ask) / 2.0
        if best_bid is not None and best_ask is not None
        else None
    )
    spread: float | None = (
        best_ask - best_bid
        if best_bid is not None and best_ask is not None
        else None
    )
    return BookMetrics(
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread=spread,
        depth_bid=sum(float(level.get("size", 0.0)) for level in bids),
        depth_ask=sum(float(level.get("size", 0.0)) for level in asks),
    )


def compute_quote(book: NormalizedBook, side: str, size_usd: float) -> Quote:
    """Walk L2 depth for the requested side and target size.

    Parameters
    ----------
    book:
        Dict with ``"bids"`` and ``"asks"`` keys, each a list of
        ``{"price": float, "size": float}`` dicts where size is in shares.
        Both lists are sorted defensively inside this function (asks
        ascending, bids descending) so callers need not pre-sort.
    side:
        ``"BUY"`` (consume asks ascending) or ``"SELL"`` (consume bids
        descending).
    size_usd:
        Target trade size in USD.

    Returns
    -------
    Quote
        When depth is insufficient to fill ``size_usd``, ``complete_fill``
        is ``False`` and ``max_fillable_usd < size_usd``.  Callers must
        skip or cancel in that case; no partial fills are executed in v1.

    Raises
    ------
    ValueError
        If ``side`` is not ``"BUY"`` or ``"SELL"``, or if ``size_usd <= 0``.
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if size_usd <= 0:
        raise ValueError(f"size_usd must be positive, got {size_usd!r}")

    bids: list[dict] = sorted(
        book.get("bids", []),
        key=lambda x: float(x["price"]),
        reverse=True,
    )
    asks: list[dict] = sorted(
        book.get("asks", []),
        key=lambda x: float(x["price"]),
    )
    metrics = compute_book_metrics(book)
    best_bid = metrics.best_bid
    best_ask = metrics.best_ask
    mid_price = metrics.mid_price

    levels = asks if side == "BUY" else bids

    remaining = size_usd
    filled_shares = 0.0
    total_cost = 0.0

    for level in levels:
        price = float(level["price"])
        avail_shares = float(level["size"])
        usd_at_level = price * avail_shares
        usd_to_spend = min(remaining, usd_at_level)
        shares_taken = usd_to_spend / price

        filled_shares += shares_taken
        total_cost += usd_to_spend
        remaining -= usd_to_spend

        if remaining <= 1e-9:
            break

    max_fillable_usd = total_cost
    complete_fill = remaining <= 1e-9

    vwap_price = total_cost / filled_shares if filled_shares > 0 else 0.0

    if side == "BUY" and best_ask is not None and filled_shares > 0:
        slippage_bps = (vwap_price - best_ask) / best_ask * 10_000.0
    elif side == "SELL" and best_bid is not None and filled_shares > 0:
        slippage_bps = (best_bid - vwap_price) / best_bid * 10_000.0
    else:
        slippage_bps = 0.0

    return Quote(
        vwap_price=vwap_price,
        max_fillable_usd=max_fillable_usd,
        filled_shares=filled_shares,
        slippage_bps=slippage_bps,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        complete_fill=complete_fill,
    )
