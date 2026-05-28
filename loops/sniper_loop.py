"""SniperLoop: event-driven trading triggered by real-time METAR observations.

Consumes the SniperManager queue.  For each new METAR observation it
parses the actual observed temperature from the METAR text, identifies
the true temperature bucket, then places:
  - BUY YES on the matching bucket token
  - BUY NO  on every non-matching bucket token
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

# T-group: T(sign_t)(ddd)(sign_d)(ddd)  -- tenths-of-degree Celsius
_RE_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
# Standard METAR temp/dew field, e.g. "20/14" or "M03/M07"
_RE_TEMP_SLASH = re.compile(r"\b(M?\d{2})/(M?\d{2})\b")


def _parse_metar_temp_f(metar_text: str) -> float | None:
    """Return the observed temperature in degF parsed from a METAR string.

    Tries the precision T-group first (e.g. T02000139 = 20.0 C / 13.9 C).
    Falls back to the integer temp/dew field (e.g. 20/14 = 20 C).
    Returns None when no temperature can be found.
    """
    m = _RE_T_GROUP.search(metar_text)
    if m:
        sign, digits = m.group(1), m.group(2)
        val_c = int(digits) / 10.0
        if sign == "1":
            val_c = -val_c
        return val_c * 9.0 / 5.0 + 32.0

    m = _RE_TEMP_SLASH.search(metar_text)
    if m:
        t_str = m.group(1)
        sign  = -1 if t_str.startswith("M") else 1
        val_c = sign * int(t_str.lstrip("M"))
        return val_c * 9.0 / 5.0 + 32.0

    return None

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
    from core.risk.base import BaseRiskModel
    from db.schema import DbClient
    from snipers.sniper_manager import SniperManager


class SniperLoop:
    """Consume METAR observations and fire ground-truth BUY order cycles.

    Parameters
    ----------
    sniper_mgr:
        SniperManager whose queue yields (metar_text, source, timestamp).
    orderbook:
        Duck-typed fetcher with ``get_book(token_id) -> dict``.
    risk:
        BaseRiskModel implementation.
    exchange:
        BaseExchange implementation.
    positions:
        PositionTracker backed by the DB.
    db:
        DbClient for querying active events and markets.
    """

    def __init__(
        self,
        sniper_mgr: "SniperManager",
        orderbook:  Any,
        risk:       "BaseRiskModel",
        exchange:   "BaseExchange",
        positions:  "PositionTracker",
        db:         "DbClient",
    ) -> None:
        self.sniper_mgr = sniper_mgr
        self.orderbook  = orderbook
        self.risk       = risk
        self.exchange   = exchange
        self.positions  = positions
        self.db         = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the full sniper subsystem until cancelled.

        Starts source polling via SniperManager and consumes the resulting
        queue in parallel.  The loop exits only when the task is cancelled
        or a source task fails.
        """
        logger.info("SniperLoop started")
        try:
            await asyncio.gather(
                self._consume_queue(),
                self.sniper_mgr.start_all(),
            )
        finally:
            stop_all = getattr(self.sniper_mgr, "stop_all", None)
            if stop_all is not None:
                await stop_all()

    async def _consume_queue(self) -> None:
        """Consume queued METAR observations and process each independently."""
        while True:
            metar_text, source, timestamp = await self.sniper_mgr.queue.get()
            logger.debug("SniperLoop: new METAR from {} at {}", source, timestamp)
            try:
                await self._handle_observation(metar_text, source, timestamp)
            except Exception:
                logger.exception("SniperLoop: error handling METAR from {}", source)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_observation(
        self,
        metar_text: str,
        source:     str,
        timestamp:  Any,
    ) -> None:
        """Parse ground-truth temperature and place BUY orders on true/false buckets."""
        from core.types import Signal

        actual_temp_f = _parse_metar_temp_f(metar_text)
        if actual_temp_f is None:
            logger.warning(
                "SniperLoop: could not parse temperature from METAR ({}): {}",
                source, metar_text,
            )
            return

        logger.debug(
            "SniperLoop: actual_temp_f={:.1f} from {}", actual_temp_f, source
        )

        now_date = datetime.now(timezone.utc).date().isoformat()
        if hasattr(self.db, "get_tradeable_events"):
            active_events = self.db.get_tradeable_events(now_date)
        else:
            active_events = self.db.get_active_events()

        for event in active_events:
            condition_id = event["condition_id"]
            markets      = self.db.get_markets_for_event(condition_id)
            if not markets:
                continue

            # Build per-bucket lookup: {(lo_f, hi_f): {"YES": token_id, "NO": token_id}}
            bucket_tokens: dict[tuple[float, float], dict[str, str]] = {}
            for mkt in markets:
                raw_lo = mkt.get("bucket_lo_f")
                raw_hi = mkt.get("bucket_hi_f")
                if raw_lo is None or raw_hi is None:
                    logger.warning(
                        "SniperLoop: market {} has missing bucket bounds, skipping",
                        mkt.get("token_id"),
                    )
                    continue
                try:
                    lo = float(raw_lo)
                    hi = float(raw_hi)
                except (TypeError, ValueError):
                    logger.warning(
                        "SniperLoop: market {} has invalid bucket bounds ({}, {}), skipping",
                        mkt.get("token_id"), raw_lo, raw_hi,
                    )
                    continue
                if lo >= hi:
                    logger.warning(
                        "SniperLoop: market {} has non-increasing bucket ({}, {}), skipping",
                        mkt.get("token_id"), lo, hi,
                    )
                    continue
                out = str(mkt.get("outcome", "")).upper()
                if out not in ("YES", "NO"):
                    continue
                key = (lo, hi)
                bucket_tokens.setdefault(key, {})
                bucket_tokens[key][out] = str(mkt["token_id"])

            if not bucket_tokens:
                continue

            exposure = self.positions.total_exposure()

            for (lo_f, hi_f), tokens in bucket_tokens.items():
                is_true_bucket = lo_f <= actual_temp_f < hi_f
                outcome_key    = "YES" if is_true_bucket else "NO"
                token_id       = tokens.get(outcome_key)
                if token_id is None:
                    logger.debug(
                        "SniperLoop: no {} token for bucket ({}, {})",
                        outcome_key, lo_f, hi_f,
                    )
                    continue

                # Fetch best ask from orderbook; fall back to 0.95
                if self.orderbook is None:
                    price = 0.95
                else:
                    try:
                        book  = self.orderbook.get_book(token_id)
                        price = book.get("best_ask") or book.get("mid_price") or 0.95
                    except Exception:
                        price = 0.95

                sig = Signal(
                    token_id     = token_id,
                    side         = "BUY",
                    mc_prob      = 1.0,
                    clob_price   = price,
                    edge         = 1.0 - price,
                    bucket_lo_f  = lo_f,
                    bucket_hi_f  = hi_f,
                    condition_id = condition_id,
                )

                if not self.risk.approve_signal(sig, exposure):
                    continue
                size = self.risk.compute_size(sig, exposure)
                if size <= 0:
                    continue

                logger.info(
                    "SniperLoop: placing BUY {} token {} @ {:.3f} size={:.2f}",
                    outcome_key, token_id, price, size,
                )
                order_id = self.exchange.place_order(token_id, "BUY", size, price)
                if not order_id:
                    logger.error(
                        "SniperLoop: empty order_id for token {}; order not recorded",
                        token_id,
                    )
                    continue
                self.positions.record_order(
                    order_id, token_id, condition_id, "BUY", size, price
                )
                exposure += size
