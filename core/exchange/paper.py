"""Paper exchange: simulates order execution without sending real orders.

Uses real CLOB mid-prices (from OrderbookFetcher or injected book) to
simulate fills.  All state is held in memory.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.exchange.base import BaseExchange

if TYPE_CHECKING:
    from core.alerts import AlertBus


class OrderbookFetcher:
    """Thin wrapper around a read-only (L1) ClobClient.

    Exposes only ``get_book(token_id) -> dict`` so that PaperExchange can
    fetch live mid-prices without holding any L2 credentials.

    Parameters
    ----------
    clob_client:
        A ``py_clob_client.ClobClient`` constructed with chain_id only
        (no ApiCreds).
    """

    def __init__(self, clob_client: Any) -> None:
        self._client = clob_client

    def get_book(self, token_id: str) -> dict:
        """Return the order book dict for *token_id*.

        Delegates directly to ``clob_client.get_order_book()``.  Any
        exception raised by the underlying client propagates to the caller.
        """
        return self._client.get_order_book(token_id)


class PaperExchange(BaseExchange):
    """Simulate order fills using real CLOB book prices.

    Parameters
    ----------
    book_fetcher:
        Optional object exposing ``get_book(token_id) -> dict`` to pull
        live mid-prices.  If None, orders are filled at the submitted
        price immediately.
    """

    def __init__(
        self,
        book_fetcher: Any = None,
        alert_bus: "AlertBus | None" = None,
    ) -> None:
        self._fetcher   = book_fetcher
        self._alert_bus = alert_bus
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
        self._apply_position(order)

        logger.debug(
            "[PAPER] {} {} ${:.2f} @ {:.4f} (fill={:.4f}) id={}",
            side.upper(), token_id, size_usd, price, fill_price, order_id,
        )
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
            except Exception as exc:
                logger.error("[PAPER] get_book() failed for {}: {}", token_id, exc)
                self._emit_order_failure(
                    token_id=token_id,
                    reason=str(exc),
                )
        return submitted_price

    def _emit_order_failure(self, token_id: str, reason: str) -> None:
        """Emit an order_failure alert if an AlertBus is wired."""
        if self._alert_bus is None:
            return
        from core.alerts import AlertEvent
        self._alert_bus.emit(AlertEvent(
            alert_type  = "order_failure",
            station     = None,
            message     = f"PaperExchange order book fetch failed for {token_id}: {reason}",
            occurred_at = datetime.now(tz=timezone.utc),
            context     = {"token_id": token_id, "reason": reason},
        ))

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
