"""SettlementLoop: daily post-market settlement of expired events."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange
    from core.exchange.settlement import Settler
    from db.schema import DbClient
    from events.lifecycle import EventLifecycle


class SettlementLoop:
    """Settle expired events once per day after market close.

    Checks for events in ``pending_settle`` status and calls
    Settler.settle() on each one.  Successfully settled events are
    marked ``settled`` in the DB.

    Parameters
    ----------
    settler:
        Settler instance (tries Relayer, falls back to 0.99 SELL).
    exchange:
        Exchange adapter used by the Settler to place close orders.
    lifecycle:
        EventLifecycle for advancing event states.
    db:
        DbClient for querying and updating event records.
    interval_seconds:
        How often to run the settlement check.  Default 3600 (1 h)
        so that events are settled within an hour of expiry.
    """

    def __init__(
        self,
        settler:          "Settler",
        exchange:         "BaseExchange",
        lifecycle:        "EventLifecycle",
        db:               "DbClient",
        interval_seconds: float = 3_600.0,
    ) -> None:
        self._settler          = settler
        self._exchange         = exchange
        self._lifecycle        = lifecycle
        self._db               = db
        self._interval_seconds = interval_seconds

    async def run(self) -> None:
        """Periodically advance lifecycles and settle expired events."""
        import pandas as pd

        logger.info("SettlementLoop started (interval={}s)", self._interval_seconds)
        while True:
            now = pd.Timestamp.utcnow()
            try:
                events = self._db.get_active_events()
                for event in events:
                    cid        = event["condition_id"]
                    old_status = event.get("status")
                    new_status = self._lifecycle.advance(event, now)
                    if new_status != old_status:
                        logger.info(
                            "Event {} status: {} -> {}", cid, old_status, new_status
                        )
                        self._db.update_event_status(cid, new_status)
                    if new_status == "pending_settle":
                        logger.info("Settling event {}", cid)
                        try:
                            result = await asyncio.to_thread(
                                self._settler.settle, event, self._exchange
                            )
                            logger.info("Event {} settled via {}", cid, result)
                            self._db.update_event_status(cid, "settled")
                        except Exception:
                            logger.exception("Failed to settle event {}", cid)
            except Exception:
                logger.exception("SettlementLoop: unhandled error in run()")
            await asyncio.sleep(self._interval_seconds)
