"""Abstract base class for real-time METAR sniper sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class BaseSniperSource(ABC):
    """Base class for polling-based METAR observation sources.

    Subclasses implement their own polling strategy and call the
    on_observation callback whenever a previously-unseen METAR is
    detected.

    Callback signature:
        on_observation(metar_text: str, source: str, timestamp: datetime)
    """

    @abstractmethod
    async def start(self, on_observation: Callable) -> None:
        """Start the polling loop.

        Runs until stop() is called.  Each time a new observation is
        detected, on_observation(metar_text, source, timestamp) is
        invoked synchronously from the event loop thread.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Signal the polling loop to exit gracefully."""
