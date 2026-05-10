"""DiscoveryLoop: daily Gamma API polling for new weather markets."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from events.discovery import EventDiscovery


class DiscoveryLoop:
    """Run EventDiscovery.discover() once per day.

    Parameters
    ----------
    discovery:
        EventDiscovery instance (handles Gamma API + DB upsert).
    interval_seconds:
        How often to poll for new events.  Default is 86400 (24 h).
    """

    def __init__(
        self,
        discovery:        "EventDiscovery",
        interval_seconds: float = 86_400.0,
    ) -> None:
        self._discovery        = discovery
        self._interval_seconds = interval_seconds

    async def run(self) -> None:
        """Poll Gamma API on a fixed interval until cancelled."""
        logger.info("DiscoveryLoop started (interval={}s)", self._interval_seconds)
        while True:
            logger.debug("DiscoveryLoop: running discover()")
            try:
                await asyncio.to_thread(self._discovery.discover)
                logger.debug("DiscoveryLoop: discover() completed")
            except Exception:
                logger.exception("DiscoveryLoop: unhandled error in discover()")
            await asyncio.sleep(self._interval_seconds)
