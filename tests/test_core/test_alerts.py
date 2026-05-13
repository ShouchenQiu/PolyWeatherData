"""Tests for core/alerts.py -- AlertEvent and AlertBus."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from core.alerts import AlertBus, AlertEvent


# ---------------------------------------------------------------------------
# AlertEvent construction
# ---------------------------------------------------------------------------


def test_alert_event_required_fields() -> None:
    evt = AlertEvent(
        alert_type="data_stale",
        station="KLGA",
        message="HRRR data is stale",
        occurred_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert evt.alert_type == "data_stale"
    assert evt.station == "KLGA"
    assert evt.message == "HRRR data is stale"
    assert evt.context == {}


def test_alert_event_context_kwarg() -> None:
    evt = AlertEvent(
        alert_type="order_failure",
        station=None,
        message="Order rejected",
        occurred_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        context={"order_id": "ord-001", "reason": "no fill"},
    )
    assert evt.context["order_id"] == "ord-001"


def test_alert_event_station_can_be_none() -> None:
    evt = AlertEvent(
        alert_type="risk_rejected",
        station=None,
        message="Risk cap exceeded",
        occurred_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert evt.station is None


# ---------------------------------------------------------------------------
# AlertBus.subscribe + emit
# ---------------------------------------------------------------------------


def test_emit_calls_handler() -> None:
    """Single handler is invoked with the correct AlertEvent."""
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(event: AlertEvent) -> None:
        received.append(event)

    bus.subscribe(handler)
    evt = AlertEvent(
        alert_type="data_stale",
        station="KLGA",
        message="stale",
        occurred_at=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc),
    )

    async def run() -> None:
        bus.emit(evt)
        await asyncio.sleep(0)  # yield so tasks execute

    asyncio.run(run())
    assert len(received) == 1
    assert received[0] is evt


def test_emit_calls_all_handlers() -> None:
    """All subscribed handlers are called."""
    bus = AlertBus()
    log: list[str] = []

    async def h1(event: AlertEvent) -> None:
        log.append("h1")

    async def h2(event: AlertEvent) -> None:
        log.append("h2")

    bus.subscribe(h1)
    bus.subscribe(h2)
    evt = AlertEvent(
        alert_type="data_stale",
        station=None,
        message="x",
        occurred_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )

    async def run() -> None:
        bus.emit(evt)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert set(log) == {"h1", "h2"}


def test_failing_handler_does_not_block_other_handlers() -> None:
    """If h1 raises, h2 still runs."""
    bus = AlertBus()
    log: list[str] = []

    async def bad(event: AlertEvent) -> None:
        raise RuntimeError("boom")

    async def good(event: AlertEvent) -> None:
        log.append("good")

    bus.subscribe(bad)
    bus.subscribe(good)
    evt = AlertEvent(
        alert_type="order_failure",
        station=None,
        message="fail",
        occurred_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )

    async def run() -> None:
        bus.emit(evt)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert "good" in log


def test_emit_outside_event_loop_raises() -> None:
    """emit() raises RuntimeError when called outside a running event loop."""
    bus = AlertBus()
    evt = AlertEvent(
        alert_type="data_stale",
        station=None,
        message="x",
        occurred_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
    )
    with pytest.raises(RuntimeError):
        bus.emit(evt)


def test_emit_accepted_alert_types() -> None:
    """All three defined alert_type values can be emitted without error."""
    bus = AlertBus()
    received: list[str] = []

    async def handler(event: AlertEvent) -> None:
        received.append(event.alert_type)

    bus.subscribe(handler)

    async def run() -> None:
        for atype in ("data_stale", "order_failure", "risk_rejected"):
            bus.emit(AlertEvent(
                alert_type=atype,
                station=None,
                message=atype,
                occurred_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
            ))
        await asyncio.sleep(0)

    asyncio.run(run())
    assert set(received) == {"data_stale", "order_failure", "risk_rejected"}
