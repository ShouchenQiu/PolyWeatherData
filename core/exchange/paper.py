"""Paper exchange: simulates order execution without sending real orders.

Uses real CLOB mid-prices (from OrderbookFetcher or injected book) to
simulate fills.  All state is held in memory.
"""
from __future__ import annotations

import uuid
from typing import Any

from core.exchange.base import BaseExchange


class PaperExchange(BaseExchange):
    """Simulate order fills using real CLOB book prices.

    Parameters
    ----------
    book_fetcher:
        Optional object exposing ``get_book(token_id) -> dict`` to pull
        live mid-prices.  If None, orders are filled at the submitted
        price immediately.
    """

    def __init__(self, book_fetcher: Any = None) -> None:
        self._fetcher   = book_fetcher
        # {order_id: order_dict}
        self._orders:    dict[str, dict] = {}
        # {order_id: position_dict}
        self._positions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # BaseExchange interface
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side:     str,
        size_usd: float,
        price:    float,
    ) -> str:
        """Simulate immediate fill at mid-price (or submitted price)."""
        fill_price = self._get_fill_price(token_id, price)
        order_id   = str(uuid.uuid4())

        order = {
            "order_id":    order_id,
            "token_id":    token_id,
            "side":        side.upper(),
            "size_usd":    size_usd,
            "price":       price,
            "fill_price":  fill_price,
            "status":      "filled",
        }
        self._orders[order_id] = order

        # Track net position per token
        self._apply_position(order)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order["status"] == "pending":
            order["status"] = "cancelled"
            return True
        return False

    def get_positions(self) -> list[dict]:
        """Return currently open simulated positions."""
        return [
            pos for pos in self._positions.values()
            if pos.get("size_usd", 0) > 0
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_fill_price(self, token_id: str, submitted_price: float) -> float:
        """Return the price at which this order would be filled."""
        if self._fetcher is not None:
            try:
                book = self._fetcher.get_book(token_id)
                mid  = book.get("mid_price")
                if mid is not None:
                    return float(mid)
            except Exception:
                pass
        return submitted_price

    def _apply_position(self, order: dict) -> None:
        token_id = order["token_id"]
        side     = order["side"]
        size_usd = order["size_usd"]

        pos = self._positions.setdefault(token_id, {
            "token_id": token_id,
            "size_usd": 0.0,
            "side":     side,
            "avg_price": 0.0,
        })

        if side == "BUY":
            old_size  = pos["size_usd"]
            new_size  = old_size + size_usd
            old_avg   = pos["avg_price"]
            pos["avg_price"] = (
                (old_avg * old_size + order["fill_price"] * size_usd) / new_size
                if new_size > 0 else order["fill_price"]
            )
            pos["size_usd"] = new_size
            pos["side"]     = "BUY"
        else:  # SELL
            pos["size_usd"] = max(0.0, pos["size_usd"] - size_usd)

    def get_order(self, order_id: str) -> dict | None:
        """Return the order record by ID."""
        return self._orders.get(order_id)
