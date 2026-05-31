"""CLOB orderbook fetcher.

Wraps the polymarket-client SDK's PublicClient to return a normalized L2
order book that the rest of the codebase can consume without importing the
SDK directly.

polymarket-client is an optional dependency; the module imports cleanly
without it.  OrderbookFetcher.__init__ accepts any object exposing
``get_order_book(*, token_id)``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OrderbookReader(Protocol):
    """Structural interface for anything that can return an L2 order book.

    Any object implementing ``get_book(token_id)`` satisfies this Protocol.
    Used as the type annotation for the ``orderbook`` parameter in
    TradingLoop and SniperLoop so the concrete SDK class is not imported
    outside ``data/``.
    """

    def get_book(self, token_id: str) -> dict: ...


def _order_list(entries: list | None) -> list[dict]:
    """Normalise a list of book-level objects to plain dicts with float fields."""
    if not entries:
        return []
    out = []
    for e in entries:
        if isinstance(e, dict):
            out.append({"price": float(e["price"]), "size": float(e.get("size", 0))})
        else:
            # Dataclass / SDK object with .price and .size attributes
            out.append({"price": float(e.price), "size": float(e.size)})
    return out


class OrderbookFetcher:
    """Fetch a normalized L2 order book for a CLOB token.

    Parameters
    ----------
    public_client:
        A ``polymarket_client.PublicClient`` instance (or any object exposing
        ``get_order_book(*, token_id)``).
    """

    def __init__(self, public_client: Any) -> None:
        self._client = public_client

    def get_book(self, token_id: str) -> dict:
        """Fetch and normalize the live orderbook for *token_id*.

        Returns
        -------
        {
            "token_id":  str,
            "bids":      list[{"price": float, "size": float}],  # descending
            "asks":      list[{"price": float, "size": float}],  # ascending
            "best_bid":  float | None,
            "best_ask":  float | None,
            "mid_price": float | None,  # (best_bid + best_ask) / 2, or None
        }
        """
        raw = self._client.get_order_book(token_id=token_id)

        raw_bids = raw.bids if hasattr(raw, "bids") else raw.get("bids", [])
        raw_asks = raw.asks if hasattr(raw, "asks") else raw.get("asks", [])

        # Sort defensively: new SDK returns bids ascending (lowest first).
        # Normalize to descending for bids, ascending for asks.
        bids = sorted(_order_list(raw_bids), key=lambda x: x["price"], reverse=True)
        asks = sorted(_order_list(raw_asks), key=lambda x: x["price"])

        best_bid: float | None = bids[0]["price"] if bids else None
        best_ask: float | None = asks[0]["price"] if asks else None
        mid_price: float | None = (
            (best_bid + best_ask) / 2.0
            if best_bid is not None and best_ask is not None
            else None
        )

        return {
            "token_id":  token_id,
            "bids":      bids,
            "asks":      asks,
            "best_bid":  best_bid,
            "best_ask":  best_ask,
            "mid_price": mid_price,
        }

    def get_mid_price(self, token_id: str) -> float | None:
        """Return only the mid-price (convenience wrapper)."""
        return self.get_book(token_id)["mid_price"]

