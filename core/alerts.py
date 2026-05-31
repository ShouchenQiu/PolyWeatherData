"""Alert bus for broadcasting operational events to subscribed handlers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal

from loguru import logger


@dataclass
class AlertEvent:
    """Payload emitted by the AlertBus.

    Fields
    ------
    alert_type:
        Domain: ``"data_stale" | "order_failure" | "risk_rejected" | "book_fetch_failure"``.
    station:
        Station ID relevant to the alert, or None if not applicable.
    message:
        Human-readable description shown in the UI.
    occurred_at:
        UTC timestamp when the condition was detected.
    context:
        Alert-specific structured data (e.g. token_id, order_id, reason).
        Consumed by downstream subscribers such as Telegram or dashboard.
    """

    alert_type:  Literal["data_stale", "order_failure", "risk_rejected", "book_fetch_failure", "discovery_warn"]
    station:     str | None
    message:     str
    occurred_at: datetime
    context:     dict[str, Any] = field(default_factory=dict)


Handler = Callable[[AlertEvent], Awaitable[None]]


class AlertBus:
    """Fanout bus that schedules async alert handlers via asyncio tasks.

    All handlers are fire-and-forget; exceptions are logged at ERROR level
    and do not propagate to the caller or to other handlers.
    """

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        """Register an async handler to receive future alerts."""
        self._handlers.append(handler)

    def emit(self, event: AlertEvent) -> None:
        """Schedule all handlers for *event* via asyncio.create_task.

        Must be called from within a running asyncio event loop.

        Raises
        ------
        RuntimeError
            If there is no running event loop.
        """
        loop = asyncio.get_running_loop()  # raises RuntimeError when none
        for handler in self._handlers:
            task = loop.create_task(self._call(handler, event))
            task.add_done_callback(lambda t: None)  # prevent GC warning

    def emit_nowait(self, event: AlertEvent) -> None:
        """Emit *event* from sync context.

        If a running event loop exists, schedules tasks normally.  Otherwise
        dispatches handlers synchronously via ``asyncio.run`` on the first
        available handler, which is suitable for background threads and
        discovery polling (which runs in an executor thread).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            for handler in self._handlers:
                task = loop.create_task(self._call(handler, event))
                task.add_done_callback(lambda t: None)
        else:
            for handler in self._handlers:
                asyncio.run(self._call(handler, event))

    @staticmethod
    async def _call(handler: Handler, event: AlertEvent) -> None:
        try:
            await handler(event)
        except Exception:
            logger.error(
                "AlertBus handler {} raised on event {}",
                getattr(handler, "__name__", repr(handler)),
                event.alert_type,
            )
