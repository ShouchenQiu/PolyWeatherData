"""SniperManager: runs multiple sniper sources concurrently and deduplicates."""
from __future__ import annotations

import asyncio
from datetime import datetime

from snipers.base import BaseSniperSource


class SniperManager:
    """Orchestrates one or more BaseSniperSource instances.

    Each source calls _on_observation() whenever it detects a new
    METAR.  The manager deduplicates across sources using metar_text
    as the key, and places unique observations onto an asyncio.Queue.
    Consumers such as SniperLoop own queue processing and trading.

    Queue items are 3-tuples: (metar_text, source, timestamp).

    Parameters
    ----------
    sources:
        List of BaseSniperSource instances to run concurrently.
    """

    def __init__(self, sources: list[BaseSniperSource]) -> None:
        self._sources  = sources
        self.seen_obs: set[str]   = set()
        self.queue: asyncio.Queue = asyncio.Queue()

    async def start_all(self) -> None:
        """Launch all sources concurrently via asyncio.gather."""
        await asyncio.gather(
            *(src.start(self._on_observation) for src in self._sources)
        )

    async def stop_all(self) -> None:
        """Signal every source to stop."""
        await asyncio.gather(*(src.stop() for src in self._sources))

    def _on_observation(
        self,
        metar_text: str,
        source:     str,
        timestamp:  datetime,
    ) -> None:
        """Deduplicate and enqueue the observation.

        The deduplication key is the stripped METAR text.  If the same
        observation arrives from both AWC and NOAA it is enqueued only
        once.
        """
        key = metar_text.strip()
        if key in self.seen_obs:
            return
        self.seen_obs.add(key)
        self.queue.put_nowait((key, source, timestamp))
