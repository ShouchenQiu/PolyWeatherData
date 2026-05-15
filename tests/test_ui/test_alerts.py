"""Tests for ui/alerts.py -- staleness state and alert handlers."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone

import pytest
from loguru import logger

from core.alerts import AlertBus, AlertEvent
from ui.alerts import (
    StalenessState,
    make_data_stale_handler,
    register_ui_handlers,
    telegram_stub_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


def _evt(alert_type: str = "data_stale", station: str | None = "KLGA") -> AlertEvent:
    return AlertEvent(
        alert_type=alert_type,  # type: ignore[arg-type]
        station=station,
        message="test",
        occurred_at=_NOW,
    )


# ---------------------------------------------------------------------------
# StalenessState
# ---------------------------------------------------------------------------


def test_staleness_state_initially_not_stale() -> None:
    state = StalenessState()
    assert state.is_stale is False
    assert state.stale_since is None
    assert state.station is None


def test_staleness_state_mark_stale() -> None:
    state = StalenessState()
    state.mark_stale(_NOW, "KLGA")
    assert state.is_stale is True
    assert state.stale_since == _NOW
    assert state.station == "KLGA"


def test_staleness_state_mark_stale_station_none() -> None:
    state = StalenessState()
    state.mark_stale(_NOW, None)
    assert state.is_stale is True
    assert state.station is None


# ---------------------------------------------------------------------------
# make_data_stale_handler
# ---------------------------------------------------------------------------


def test_data_stale_handler_marks_state_stale() -> None:
    """data_stale event causes state.is_stale to become True."""
    state   = StalenessState()
    handler = make_data_stale_handler(state)

    asyncio.run(handler(_evt("data_stale", "KLGA")))

    assert state.is_stale is True
    assert state.stale_since == _NOW
    assert state.station == "KLGA"


def test_data_stale_handler_ignores_non_stale_events() -> None:
    """Handler does not touch state for non-data_stale alert types."""
    state   = StalenessState()
    handler = make_data_stale_handler(state)

    asyncio.run(handler(_evt("risk_rejected", "KLGA")))

    assert state.is_stale is False


# ---------------------------------------------------------------------------
# telegram_stub_handler
# ---------------------------------------------------------------------------


def test_telegram_stub_logs_debug(capsys) -> None:
    """telegram_stub_handler logs at DEBUG level and does not raise."""
    records: list[str] = []

    def capture(message):
        records.append(message)

    logger.add(capture, level="DEBUG", format="{message}")
    try:
        asyncio.run(telegram_stub_handler(_evt("data_stale")))
    finally:
        logger.remove()

    assert any("Telegram not yet implemented" in r for r in records)


def test_telegram_stub_handler_does_not_raise() -> None:
    """telegram_stub_handler completes without raising for any known alert_type."""
    for alert_type in ("data_stale", "order_failure", "risk_rejected", "book_fetch_failure"):
        asyncio.run(telegram_stub_handler(_evt(alert_type)))  # must not raise


# ---------------------------------------------------------------------------
# register_ui_handlers
# ---------------------------------------------------------------------------


def test_register_ui_handlers_wires_staleness_via_bus() -> None:
    """After register_ui_handlers, a data_stale emit updates the state."""
    bus   = AlertBus()
    state = StalenessState()
    register_ui_handlers(bus, state)

    async def run() -> None:
        bus.emit(_evt("data_stale", "KLGA"))
        await asyncio.sleep(0)  # let tasks execute

    asyncio.run(run())
    assert state.is_stale is True


def test_register_ui_handlers_wires_telegram_via_bus() -> None:
    """After register_ui_handlers, any emit fires telegram stub without exception."""
    bus   = AlertBus()
    state = StalenessState()
    register_ui_handlers(bus, state)

    async def run() -> None:
        bus.emit(_evt("order_failure", None))
        await asyncio.sleep(0)

    asyncio.run(run())  # must not raise
