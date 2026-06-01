"""SettlementResolver: compute per-order payouts from observed temperature.

Given a condition_id and the observed max temperature, resolves the YES/NO
payout for every filled/closed order on that event.  All DB access and
payout rules are encapsulated here so SettlementLoop only handles scheduling
and observation loading.

Payout semantics (V2F05 / signed_shares):
    Each winning share redeems for $1.00.  Payout uses ``signed_shares``:
    BUY orders have positive signed_shares, SELL orders have negative
    signed_shares.  Payout = max(0, signed_shares) for winning rows;
    losing rows get 0.0.  Rows recorded before V2F05 (signed_shares IS
    NULL) fall back to ``filled_shares`` (always positive, same as V1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from db.schema import DbClient


@dataclass(frozen=True)
class OrderSettlement:
    """Payout decision for one order.

    ``token_id`` and ``signed_shares`` are carried so that
    ``SettlementLoop`` can compute net position per token for redeem-task
    gating (only net-long tokens trigger a redeem task).
    """

    order_id:      str
    token_id:      str
    payout:        float
    signed_shares: float = 0.0


class SettlementResolver:
    """Compute bucket payouts for all filled/closed orders of one condition.

    Payout rules:
      - YES token in winning bucket  -> payout = max(0, signed_shares)
      - YES token in losing bucket   -> payout = 0.0
      - NO token in winning bucket   -> payout = 0.0
      - NO token in losing bucket    -> payout = max(0, signed_shares)

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
            order_id     = order["order_id"]
            token_id     = order["token_id"]
            size_usd     = float(order.get("size_usd") or 0.0)

            signed_raw = order.get("signed_shares")
            filled_raw = order.get("filled_shares")
            if signed_raw is not None:
                signed = float(signed_raw)
            elif filled_raw is not None:
                # Pre-V2F05 row: signed_shares not recorded; use filled_shares
                # (always positive, same as V1).  Logged at debug level since
                # this is common for historical rows.
                logger.debug(
                    "SettlementResolver: order {} has no signed_shares; "
                    "using filled_shares={:.4f} (always-positive fallback)",
                    order_id, float(filled_raw),
                )
                signed = float(filled_raw)
            else:
                logger.warning(
                    "SettlementResolver: order {} has no signed_shares or "
                    "filled_shares; using size_usd={:.4f} as fallback",
                    order_id, size_usd,
                )
                signed = size_usd

            entry = token_market.get(token_id)
            if entry is not None:
                lo, hi, outcome = entry
                in_bucket = lo <= actual_max_temp < hi
                is_winner = (outcome == "YES") == in_bucket
                payout = max(0.0, signed) if is_winner else 0.0
            else:
                logger.warning(
                    "SettlementResolver: token_id {} has no market entry; payout=0",
                    token_id,
                )
                payout = 0.0

            results.append(OrderSettlement(
                order_id      = order_id,
                token_id      = token_id,
                payout        = payout,
                signed_shares = signed,
            ))

        return results
