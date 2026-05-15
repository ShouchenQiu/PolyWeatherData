"""CLOB orderbook fetcher.

Wraps the py_clob_client SDK's get_order_book() method and returns a
plain dict that the rest of the codebase can consume without importing
the SDK directly.

py_clob_client is an optional dependency; the module imports cleanly
without it.  OrderbookFetcher.__init__ will raise ImportError at runtime
if the package is missing.
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


def _mid(bids: list, asks: list) -> float | None:
    """Compute mid-price from best bid / best ask.

    Parameters
    ----------
    bids / asks:
        Lists of OrderSummary objects (or plain dicts) with a ``price``
        attribute / key.  Bids are sorted descending, asks ascending by
        the SDK.
    """
    best_bid = best_ask = None
    if bids:
        try:
            best_bid = float(bids[0].price)
        except AttributeError:
            best_bid = float(bids[0]["price"])
    if asks:
        try:
            best_ask = float(asks[0].price)
        except AttributeError:
            best_ask = float(asks[0]["price"])

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    if best_bid is not None:
        return best_bid
    if best_ask is not None:
        return best_ask
    return None


def _order_list(entries: list | None) -> list[dict]:
    """Normalise a list of OrderSummary objects to plain dicts."""
    if not entries:
        return []
    out = []
    for e in entries:
        if isinstance(e, dict):
            out.append({"price": float(e["price"]), "size": float(e.get("size", 0))})
        else:
            # OrderSummary dataclass
            out.append({"price": float(e.price), "size": float(e.size)})
    return out


class OrderbookFetcher:
    """Fetch best-bid / best-ask and mid-price for a CLOB token.

    Parameters
    ----------
    clob_client:
        A ``py_clob_client.ClobClient`` instance (or any object exposing
        a ``get_order_book(token_id)`` method that returns an
        ``OrderBookSummary``).
    """

    def __init__(self, clob_client: Any) -> None:
        self._client = clob_client

    def get_book(self, token_id: str) -> dict:
        """Fetch the live orderbook for *token_id*.

        Returns
        -------
        {
            "token_id":  str,
            "bids":      list[{"price": float, "size": float}],
            "asks":      list[{"price": float, "size": float}],
            "mid_price": float | None,
        }
        """
        raw = self._client.get_order_book(token_id)
        bids = _order_list(raw.bids)
        asks = _order_list(raw.asks)
        return {
            "token_id":  token_id,
            "bids":      bids,
            "asks":      asks,
            "mid_price": _mid(raw.bids, raw.asks),
        }

    def get_mid_price(self, token_id: str) -> float | None:
        """Return only the mid-price (convenience wrapper)."""
        return self.get_book(token_id)["mid_price"]
