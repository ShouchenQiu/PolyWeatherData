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
        is_paper: bool = False,
        executable_price: float | None = None,
        fee_usd: float = 0.0,
    ) -> None:
        """Insert a pending order row. All nullable columns are NULL."""
        self._db.execute(
            """
            INSERT INTO orders
                (order_id, token_id, condition_id, side, size_usd, price,
                 status, created_at, is_paper, executable_price, fee_usd)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (order_id, token_id, condition_id, side, size_usd, price,
             created_at or _utc_now(), int(is_paper), executable_price,
             fee_usd),
        )

    def mark_filled(
        self,
        order_id: str,
        fill_price: float,
        filled_at: str | None = None,
    ) -> None:
        """Transition status to 'filled'; write fill_price, filled_at, and
        share/cash-flow accounting columns.

        BUY fills: positive signed_shares, negative cash_flow_usd (outflow).
        SELL fills: negative signed_shares, positive cash_flow_usd (inflow).
        """
        rows = self._db.query(
            "SELECT side, size_usd, fee_usd FROM orders WHERE order_id=?",
            (order_id,),
        )
        if not rows:
            return
        side     = rows[0]["side"]
        size_usd = float(rows[0]["size_usd"])
        fee_usd = float(rows[0]["fee_usd"] or 0.0)

        filled_shares = size_usd / fill_price if fill_price > 0 else 0.0
        if side == "BUY":
            signed_shares = filled_shares
            cash_flow_usd = -(size_usd + fee_usd)
        else:
            signed_shares = -filled_shares
            cash_flow_usd = size_usd - fee_usd

        self._db.execute(
            "UPDATE orders SET status='filled', fill_price=?, filled_at=?,"
            " filled_shares=?, signed_shares=?, cash_flow_usd=?"
            " WHERE order_id=?",
            (fill_price, filled_at or _utc_now(),
             filled_shares, signed_shares, cash_flow_usd,
             order_id),
        )

    def mark_closed(
        self,
        order_id: str,
        closed_at: str | None = None,
    ) -> None:
        """Transition status to 'closed', set closed_at."""
        self._db.execute(
            "UPDATE orders SET status='closed', closed_at=? WHERE order_id=?",
            (closed_at or _utc_now(), order_id),
        )

    def mark_cancelled(
        self,
        order_id: str,
        cancelled_at: str | None = None,
    ) -> None:
        """Transition status to 'cancelled', set cancelled_at."""
        self._db.execute(
            "UPDATE orders SET status='cancelled', cancelled_at=? WHERE order_id=?",
            (cancelled_at or _utc_now(), order_id),
        )

    def mark_settled(
        self,
        order_id: str,
        settled_payout: float,
    ) -> None:
        """Transition status to 'settled', write settled_payout."""
        self._db.execute(
            "UPDATE orders SET status='settled', settled_payout=? WHERE order_id=?",
            (settled_payout, order_id),
        )

    # ------------------------------------------------------------------
    # Read -- exposure queries
    # ------------------------------------------------------------------

    def gross_open_notional_usd(self) -> float:
        """Sum of size_usd for all pending|filled orders (both BUY and SELL).

        This is a gross notional figure used by the risk gate:
        ``current_exposure + proposed_size_usd <= max_total_usd``.

        It intentionally double-counts opposite-side orders (e.g. a pending
        SELL does not reduce this figure) so that the risk gate remains
        conservative while a SELL order is in-flight.  Use
        ``position_size_usd()`` or ``position_shares()`` for net exposure.
        """
        rows = self._db.query(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders"
            " WHERE status IN ('pending', 'filled')"
        )
        return float(rows[0][0])

    def net_cash_flow_usd(self, is_paper: bool | None = None) -> float:
        """Return cumulative filled-order cash flow for one execution mode."""
        paper_sql = "" if is_paper is None else " AND is_paper=?"
        params: tuple = () if is_paper is None else (int(is_paper),)
        rows = self._db.query(
            "SELECT COALESCE(SUM(cash_flow_usd), 0.0) FROM orders"
            " WHERE status='filled' AND cash_flow_usd IS NOT NULL"
            f"{paper_sql}",
            params,
        )
        return float(rows[0][0])

    def exposure_by_condition(self, condition_id: str) -> float:
        """Sum of size_usd for pending|filled orders matching condition_id."""
        rows = self._db.query(
            "SELECT COALESCE(SUM(size_usd), 0.0) FROM orders"
            " WHERE status IN ('pending', 'filled') AND condition_id=?",
            (condition_id,),
        )
        return float(rows[0][0])

    def position_shares(self, token_id: str, is_paper: bool | None = None) -> float:
        """Net signed token shares for *token_id* across all filled orders.

        Sums ``signed_shares`` for all rows where the column has been
        written (i.e. post-fill) and the order is not cancelled.
        Returns 0.0 when no fills exist for the token.

        Parameters
        ----------
        is_paper:
            When not None, restricts the query to paper (True) or live
            (False) orders only.  When None (default), both are included.
        """
        paper_sql = "" if is_paper is None else " AND is_paper=?"
        params: tuple = (token_id,) if is_paper is None else (token_id, int(is_paper))
        rows = self._db.query(
            "SELECT COALESCE(SUM(signed_shares), 0.0) FROM orders"
            " WHERE token_id=? AND signed_shares IS NOT NULL"
            f" AND status = 'filled'{paper_sql}",
            params,
        )
        return float(rows[0][0])

    def position_size_usd(self, token_id: str, is_paper: bool | None = None) -> float:
        """Net USDC cost-basis for *token_id*: sum of BUY size_usd minus SELL
        size_usd across all filled rows.

        Used for exposure reporting; SELL sizing must use
        ``position_shares()`` instead.

        Parameters
        ----------
        is_paper:
            When not None, restricts to paper (True) or live (False) rows.
        """
        paper_sql = "" if is_paper is None else " AND is_paper=?"
        params: tuple = (token_id,) if is_paper is None else (token_id, int(is_paper))
        rows = self._db.query(
            "SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN size_usd"
            "                        WHEN side='SELL' THEN -size_usd"
            "                        ELSE 0 END), 0.0)"
            " FROM orders"
            f" WHERE token_id=? AND status='filled'{paper_sql}",
            params,
        )
        return float(rows[0][0])
