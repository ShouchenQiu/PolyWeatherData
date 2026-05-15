"""SettlementResolver: compute per-order payouts from observed temperature.

Given a condition_id and the observed max temperature, resolves the YES/NO
payout for every filled/closed order on that event.  All DB access and
payout rules are encapsulated here so SettlementLoop only handles scheduling
and observation loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from db.schema import DbClient


@dataclass(frozen=True)
class OrderSettlement:
    """Payout decision for one order."""

    order_id: str
    payout: float


class SettlementResolver:
    """Compute bucket payouts for all filled/closed orders of one condition.

    Payout rules:
      - YES token in winning bucket  -> payout = size_usd
      - YES token in losing bucket   -> payout = 0.0
      - NO token in winning bucket   -> payout = 0.0
      - NO token in losing bucket    -> payout = size_usd

    An order whose token_id has no matching market entry gets payout = 0.0.

    Parameters
    ----------
    db:
        DbClient for querying markets and orders.
    """

    def __init__(self, db: "DbClient") -> None:
        self._db = db

    def resolve_from_obs(
        self, condition_id: str, actual_max_temp: float
    ) -> list[OrderSettlement]:
        """Return payout decisions for all filled/closed orders on *condition_id*.

        Parameters
        ----------
        condition_id:
            Polymarket condition ID for the event.
        actual_max_temp:
            Observed daily maximum temperature in degrees Fahrenheit.
        """
        token_market: dict[str, tuple[float, float, str]] = {}
        for mkt in self._db.get_markets_for_event(condition_id):
            lo = mkt.get("bucket_lo_f")
            hi = mkt.get("bucket_hi_f")
            if lo is not None and hi is not None:
                token_market[mkt["token_id"]] = (
                    float(lo),
                    float(hi),
                    (mkt.get("outcome") or "YES").upper(),
                )

        results: list[OrderSettlement] = []
        for order in self._db.get_orders_for_settlement(condition_id):
            order_id = order["order_id"]
            token_id = order["token_id"]
            size_usd = float(order.get("size_usd") or 0.0)

            entry = token_market.get(token_id)
            if entry is not None:
                lo, hi, outcome = entry
                in_bucket = lo <= actual_max_temp < hi
                payout = size_usd if (outcome == "YES") == in_bucket else 0.0
            else:
                logger.warning(
                    "SettlementResolver: token_id {} has no market entry; payout=0",
                    token_id,
                )
                payout = 0.0

            results.append(OrderSettlement(order_id=order_id, payout=payout))

        return results
