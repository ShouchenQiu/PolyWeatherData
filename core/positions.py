"""Position tracker backed by the SQLite orders table."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Order, Signal
    from db.schema import DbClient


class PositionTracker:
    """Persist and query open trading positions via DbClient.

    Parameters
    ----------
    db:
        DbClient instance (SQLite backend).
    """

    def __init__(self, db: "DbClient") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, signal: "Signal", order: "Order") -> None:
        """Record a newly placed order.

        Parameters
        ----------
        signal:
            The signal that triggered the order (used to store the
            signal in the DB and link it to the order).
        order:
            Order dataclass populated by the exchange adapter.
        """
        now = datetime.now(timezone.utc).isoformat()

        signal_id = self._db.insert_signal(
            timestamp  = now,
            token_id   = signal.token_id,
            mc_prob    = signal.mc_prob,
            clob_price = signal.clob_price,
            edge       = signal.edge,
            side       = signal.side,
        )

        self._db.insert_order(
            order_id  = order.order_id,
            signal_id = signal_id,
            size_usd  = order.size_usd,
            price     = order.price,
            status    = order.status,
        )

    def fill(self, order_id: str, filled_price: float) -> None:
        """Mark an order as filled.

        Parameters
        ----------
        order_id:
            Exchange order identifier.
        filled_price:
            Actual fill price (stored for PnL accounting).
        """
        filled_at = datetime.now(timezone.utc).isoformat()
        self._db.update_order_status(
            order_id  = order_id,
            status    = "filled",
            filled_at = filled_at,
        )

    def close(self, order_id: str) -> None:
        """Mark an order as closed (unwound / settled)."""
        self._db.update_order_status(
            order_id = order_id,
            status   = "closed",
        )

    def get_open(self) -> list[dict]:
        """Return all orders with status ``'pending'``.

        Returns
        -------
        List of dicts with keys: order_id, signal_id, status,
        size_usd, price, filled_at.
        """
        rows = self._db.get_open_orders()
        return [dict(r) for r in rows]

    def total_exposure(self) -> float:
        """Sum of size_usd across all open (pending) positions."""
        return sum(pos["size_usd"] for pos in self.get_open())
