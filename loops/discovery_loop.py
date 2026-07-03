"""DiscoveryLoop: hourly Gamma API polling for new weather markets."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from db.schema import DbClient
    from events.discovery import EventDiscovery
    from events.lifecycle import EventLifecycle


class DiscoveryLoop:
    """Run EventDiscovery.discover() once per hour.

    After each discover() call, queries the DB for all non-terminal events and
    calls EventLifecycle.advance() on each.  Status changes are written back to
    the DB and logged.  (V2V01)

    Parameters
    ----------
    discovery:
        EventDiscovery instance (handles Gamma API + DB upsert).
    interval_seconds:
        How often to poll for new events.  Default is 3600 (1 h).
    db:
        Optional DbClient.  Required for lifecycle advance pass; if None, the
        advance pass is skipped.
    lifecycle:
        Optional EventLifecycle.  Required for lifecycle advance pass; if None,
        the advance pass is skipped.
    """

    def __init__(
        self,
        discovery:        "EventDiscovery",
        interval_seconds: float = 3_600.0,
        db:               "DbClient | None"        = None,
        lifecycle:        "EventLifecycle | None"  = None,
    ) -> None:
        if (db is None) != (lifecycle is None):
            raise ValueError(
                "DiscoveryLoop: 'db' and 'lifecycle' must both be provided "
                "or both be None; partial construction silently disables the "
                "lifecycle advance pass"
            )
        self._discovery        = discovery
        self._interval_seconds = interval_seconds
        self._db               = db
        self._lifecycle        = lifecycle

    async def run(self) -> None:
        """Poll Gamma API on a fixed interval until cancelled."""
        logger.info("DiscoveryLoop started (interval={}s)", self._interval_seconds)
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                logger.debug("DiscoveryLoop: running discover()")
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        executor,
                        self._discovery.discover,
                    )
                    logger.debug("DiscoveryLoop: discover() completed")
                except Exception:
                    logger.exception("DiscoveryLoop: unhandled error in discover()")

                if self._db is not None and self._lifecycle is not None:
                    await self._advance_lifecycle(executor)

                # A zero interval is useful in tests but must still yield for
                # cancellation before another worker call is submitted.
                await asyncio.sleep(max(self._interval_seconds, 0.01))

    async def _advance_lifecycle(
        self,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Advance lifecycle state for all non-terminal events in the DB."""
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        try:
            events = await asyncio.get_running_loop().run_in_executor(
                executor,
                self._db.get_non_terminal_events,
            )
        except Exception:
            logger.exception("DiscoveryLoop: failed to query non-terminal events")
            return

        for ev in events:
            condition_id = ev["condition_id"]
            old_status   = ev.get("status", "")
            try:
                new_status = self._lifecycle.advance(ev, now_utc)
            except Exception:
                logger.exception(
                    "DiscoveryLoop: lifecycle.advance failed for {}", condition_id
                )
                continue

            if new_status != old_status:
                logger.info(
                    "lifecycle: {} {} -> {}", condition_id, old_status, new_status
                )
                try:
                    self._db.update_event_status(condition_id, new_status)
                except Exception:
                    logger.exception(
                        "DiscoveryLoop: failed to write status for {}", condition_id
                    )
