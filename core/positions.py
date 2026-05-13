"""Position tracker backed by the SQLite orders table."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    # Used by ForecastLoop -- kept for compatibility until F03 extends
    # ------------------------------------------------------------------

    def open(self, sig: Any, order: Any) -> None:
        """Record a newly placed order using the lifecycle schema."""
        now = datetime.now(timezone.utc).isoformat()
        self._db._conn.execute(
            """
            INSERT INTO orders
                (order_id, token_id, condition_id, side, size_usd, price,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                order.order_id,
                order.token_id,
                sig.condition_id,
                order.side,
                order.size_usd,
                order.price,
                now,
            ),
        )
        self._db._conn.commit()

    def total_exposure(self) -> float:
        """Sum of size_usd across all pending positions."""
        cur = self._db._conn.execute(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders WHERE status='pending'"
        )
        return float(cur.fetchone()[0])
