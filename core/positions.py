"""Position tracker backed by the SQLite orders table."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.schema import DbClient


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # Write -- order lifecycle
    # ------------------------------------------------------------------

    def record_order(
        self,
        order_id: str,
        token_id: str,
        condition_id: str,
        side: str,
        size_usd: float,
        price: float,
        created_at: str | None = None,
    ) -> None:
        """Insert a pending order row. All nullable columns are NULL."""
        self._db._conn.execute(
            """
            INSERT INTO orders
                (order_id, token_id, condition_id, side, size_usd, price,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (order_id, token_id, condition_id, side, size_usd, price,
             created_at or _utc_now()),
        )
        self._db._conn.commit()

    def mark_filled(
        self,
        order_id: str,
        fill_price: float,
        filled_at: str | None = None,
    ) -> None:
        """Transition status to 'filled', set fill_price and filled_at."""
        self._db._conn.execute(
            "UPDATE orders SET status='filled', fill_price=?, filled_at=?"
            " WHERE order_id=?",
            (fill_price, filled_at or _utc_now(), order_id),
        )
        self._db._conn.commit()

    def mark_closed(
        self,
        order_id: str,
        closed_at: str | None = None,
    ) -> None:
        """Transition status to 'closed', set closed_at."""
        self._db._conn.execute(
            "UPDATE orders SET status='closed', closed_at=? WHERE order_id=?",
            (closed_at or _utc_now(), order_id),
        )
        self._db._conn.commit()

    def mark_cancelled(
        self,
        order_id: str,
        cancelled_at: str | None = None,
    ) -> None:
        """Transition status to 'cancelled', set cancelled_at."""
        self._db._conn.execute(
            "UPDATE orders SET status='cancelled', cancelled_at=? WHERE order_id=?",
            (cancelled_at or _utc_now(), order_id),
        )
        self._db._conn.commit()

    def mark_settled(
        self,
        order_id: str,
        settled_payout: float,
    ) -> None:
        """Transition status to 'settled', write settled_payout."""
        self._db._conn.execute(
            "UPDATE orders SET status='settled', settled_payout=? WHERE order_id=?",
            (settled_payout, order_id),
        )
        self._db._conn.commit()

    # ------------------------------------------------------------------
    # Read -- exposure queries
    # ------------------------------------------------------------------

    def open_positions_by_token(self) -> dict[str, int]:
        """Count pending|filled rows per token_id.

        Used by RiskPipeline for duplicate-order prevention.
        """
        cur = self._db._conn.execute(
            "SELECT token_id, COUNT(*) AS cnt FROM orders"
            " WHERE status IN ('pending', 'filled')"
            " GROUP BY token_id"
        )
        return {row["token_id"]: row["cnt"] for row in cur.fetchall()}

    def total_exposure(self) -> float:
        """Sum of size_usd for all pending|filled orders."""
        cur = self._db._conn.execute(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders"
            " WHERE status IN ('pending', 'filled')"
        )
        return float(cur.fetchone()[0])

    def exposure_by_condition(self, condition_id: str) -> float:
        """Sum of size_usd for pending|filled orders matching condition_id."""
        cur = self._db._conn.execute(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders"
            " WHERE status IN ('pending', 'filled') AND condition_id=?",
            (condition_id,),
        )
        return float(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # ForecastLoop compatibility shim -- delegates to record_order
    # ------------------------------------------------------------------

    def open(self, sig: object, order: object) -> None:
        """Compatibility shim used by ForecastLoop until V11 migrates it."""
        self.record_order(
            order_id=order.order_id,  # type: ignore[attr-defined]
            token_id=order.token_id,  # type: ignore[attr-defined]
            condition_id=sig.condition_id,  # type: ignore[attr-defined]
            side=order.side,  # type: ignore[attr-defined]
            size_usd=order.size_usd,  # type: ignore[attr-defined]
            price=order.price,  # type: ignore[attr-defined]
        )
