"""PnL calculation primitives.

PnLCalculator is a pure-computation class that reads from the DB and
does not import from loops/, model/, or data/.

Semantics
---------
realized_pnl:
    Cash flows from CLOSED and SETTLED orders only, plus settlement
    payouts for settled rows with positive signed_shares.  Open ('filled')
    orders do not contribute to realized PnL.

unrealized_pnl:
    For each token_id in mark_prices, sum
        mark_price * net_signed_shares + net_cash_flow_usd
    across all 'filled' (open) rows for that token.  Closed and settled
    rows are excluded because their contribution is already realized.

total_pnl:
    realized_pnl + unrealized_pnl
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.schema import DbClient


class PnLCalculator:
    """Compute realized and unrealized PnL from order cash flows and shares.

    Parameters
    ----------
    db:
        DbClient instance (SQLite backend).
    """

    def __init__(self, db: "DbClient") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_clause(is_paper: bool | None) -> tuple[str, list]:
        if is_paper is None:
            return ("", [])
        return (" AND is_paper=?", [int(is_paper)])

    @staticmethod
    def _cid_clause(condition_id: str | None) -> tuple[str, list]:
        if condition_id is None:
            return ("", [])
        return (" AND condition_id=?", [condition_id])

    # ------------------------------------------------------------------
    # Realized PnL
    # ------------------------------------------------------------------

    def realized_pnl(
        self,
        condition_id: str | None = None,
        is_paper: bool | None = None,
    ) -> float:
        """Sum cash flows from closed/settled orders plus settlement payouts.

        Open ('filled') orders are excluded -- their contribution is
        captured in unrealized_pnl.

        Parameters
        ----------
        condition_id:
            Restrict to a single event; ``None`` means all events.
        is_paper:
            Paper/live filter; ``None`` means both.
        """
        cid_sql, cid_p   = self._cid_clause(condition_id)
        paper_sql, pap_p = self._paper_clause(is_paper)

        # Cash flows from closed and settled orders
        cf_rows = self._db.query(
            "SELECT COALESCE(SUM(cash_flow_usd), 0.0) FROM orders"
            " WHERE status IN ('closed', 'settled')"
            " AND cash_flow_usd IS NOT NULL"
            + cid_sql + paper_sql,
            tuple(cid_p + pap_p),
        )
        cash_flow = float(cf_rows[0][0])

        # Settlement payouts for settled rows with net-long signed shares
        sp_rows = self._db.query(
            "SELECT COALESCE(SUM(settled_payout), 0.0) FROM orders"
            " WHERE status='settled'"
            " AND signed_shares > 0"
            " AND settled_payout IS NOT NULL"
            + cid_sql + paper_sql,
            tuple(cid_p + pap_p),
        )
        settlement = float(sp_rows[0][0])

        return cash_flow + settlement

    # ------------------------------------------------------------------
    # Unrealized PnL
    # ------------------------------------------------------------------

    def unrealized_pnl(
        self,
        mark_prices: dict[str, float],
        is_paper: bool | None = None,
    ) -> float:
        """Mark-to-market unrealized PnL for open ('filled') positions.

        For each token_id in *mark_prices* that has 'filled' orders:
            unrealized = mark_price * net_signed_shares + net_cash_flow_usd

        Parameters
        ----------
        mark_prices:
            ``{token_id: mid_or_mark_price}`` mapping.
        is_paper:
            Paper/live filter; ``None`` means both.
        """
        if not mark_prices:
            return 0.0

        paper_sql, pap_p = self._paper_clause(is_paper)

        rows = self._db.query(
            "SELECT token_id,"
            "       COALESCE(SUM(signed_shares), 0.0)  AS net_shares,"
            "       COALESCE(SUM(cash_flow_usd), 0.0)  AS net_cash"
            " FROM orders"
            " WHERE status='filled'"
            " AND signed_shares IS NOT NULL"
            + paper_sql
            + " GROUP BY token_id",
            tuple(pap_p),
        )

        total = 0.0
        for row in rows:
            if isinstance(row, dict):
                token_id   = row["token_id"]
                net_shares = float(row["net_shares"])
                net_cash   = float(row["net_cash"])
            else:
                token_id, net_shares, net_cash = row[0], float(row[1]), float(row[2])
            if token_id in mark_prices:
                total += mark_prices[token_id] * net_shares + net_cash
        return total

    # ------------------------------------------------------------------
    # Convenience aggregates
    # ------------------------------------------------------------------

    def total_pnl(
        self,
        mark_prices: dict[str, float],
        is_paper: bool | None = None,
    ) -> float:
        """Realized + unrealized PnL."""
        return self.realized_pnl(is_paper=is_paper) + self.unrealized_pnl(
            mark_prices, is_paper=is_paper
        )

    def pnl_by_event(
        self,
        mark_prices: dict[str, float],
        is_paper: bool | None = None,
    ) -> dict[str, float]:
        """Per-event breakdown of total PnL.

        Returns ``{condition_id: total_pnl}`` for every condition that has
        at least one non-pending, non-cancelled fill.
        """
        paper_sql, pap_p = self._paper_clause(is_paper)
        cid_rows = self._db.query(
            "SELECT DISTINCT condition_id FROM orders"
            " WHERE (cash_flow_usd IS NOT NULL OR status='filled')"
            + paper_sql,
            tuple(pap_p),
        )
        result: dict[str, float] = {}
        for row in cid_rows:
            cid = row["condition_id"] if isinstance(row, dict) else row[0]
            result[cid] = (
                self.realized_pnl(condition_id=cid, is_paper=is_paper)
                + self._unrealized_for_condition(cid, mark_prices, is_paper)
            )
        return result

    def _unrealized_for_condition(
        self,
        condition_id: str,
        mark_prices: dict[str, float],
        is_paper: bool | None,
    ) -> float:
        """Unrealized PnL restricted to a single condition_id."""
        paper_sql, pap_p = self._paper_clause(is_paper)

        rows = self._db.query(
            "SELECT token_id,"
            "       COALESCE(SUM(signed_shares), 0.0)  AS net_shares,"
            "       COALESCE(SUM(cash_flow_usd), 0.0)  AS net_cash"
            " FROM orders"
            " WHERE status='filled'"
            " AND signed_shares IS NOT NULL"
            " AND condition_id=?"
            + paper_sql
            + " GROUP BY token_id",
            (condition_id, *pap_p),
        )
        total = 0.0
        for row in rows:
            if isinstance(row, dict):
                token_id   = row["token_id"]
                net_shares = float(row["net_shares"])
                net_cash   = float(row["net_cash"])
            else:
                token_id, net_shares, net_cash = row[0], float(row[1]), float(row[2])
            if token_id in mark_prices:
                total += mark_prices[token_id] * net_shares + net_cash
        return total
