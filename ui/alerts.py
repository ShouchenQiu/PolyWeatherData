"""AlertBus handlers and shared staleness state for the PolyWeatherBot UI.

Separates testable logic from Streamlit-specific wiring in app.py.
All objects in this module are safe to import in tests without a running
Streamlit context.
"""
from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Callable, Awaitable

from loguru import logger

from core.alerts import AlertBus, AlertEvent


class StalenessState:
    """Thread-safe container tracking data-staleness state for the UI.

    Written by an asyncio AlertBus handler (the bot's event loop thread);
    read by Streamlit on each render request (Streamlit's own thread).
    A threading.Lock protects all reads and writes.

    Fields
    ------
    is_stale:
        True once a ``data_stale`` alert has been received and not reset.
    stale_since:
        UTC timestamp of the first unresolved ``data_stale`` alert.
    station:
        Station ID from the alert, or None if not applicable.
    """

    def __init__(self) -> None:
        self._lock        = Lock()
        self._is_stale    = False
        self._stale_since: datetime | None = None
        self._station:    str | None = None

    def mark_stale(self, occurred_at: datetime, station: str | None) -> None:
        """Record a staleness event.  Idempotent: re-marking while already
        stale updates the timestamp and station to the latest event."""
        with self._lock:
            self._is_stale    = True
            self._stale_since = occurred_at
            self._station     = station

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return self._is_stale

    @property
    def stale_since(self) -> datetime | None:
        with self._lock:
            return self._stale_since

    @property
    def station(self) -> str | None:
        with self._lock:
            return self._station


def make_data_stale_handler(
    state: StalenessState,
) -> Callable[[AlertEvent], Awaitable[None]]:
    """Return an async AlertBus handler that marks *state* stale on
    ``data_stale`` events.  Non-matching alert types are silently ignored."""

    async def handler(event: AlertEvent) -> None:
        if event.alert_type == "data_stale":
            state.mark_stale(event.occurred_at, event.station)

    return handler


async def telegram_stub_handler(event: AlertEvent) -> None:
    """Telegram stub: logs at DEBUG level and does nothing else.

    Full Telegram routing is deferred (see Deferred Register in
    docs/issues/proposed-slices.md).
    """
    logger.debug(
        "Telegram not yet implemented (alert_type={})", event.alert_type
    )


def register_ui_handlers(bus: AlertBus, state: StalenessState) -> None:
    """Register all UI alert handlers on *bus*.

    Wires:
    - ``make_data_stale_handler(state)`` for staleness state tracking.
    - ``telegram_stub_handler`` for stub Telegram fanout.
    """
    bus.subscribe(make_data_stale_handler(state))
    bus.subscribe(telegram_stub_handler)
