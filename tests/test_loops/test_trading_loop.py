"""Tests for loops/trading_loop.py -- TradingLoop F05 skeleton."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from config.settings import Settings
from core.alerts import AlertBus, AlertEvent
from core.probability_store import BucketPayload, ProbabilityStore
from loops.trading_loop import TradingLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_loop(**kwargs) -> TradingLoop:
    prob_store = kwargs.pop("prob_store", ProbabilityStore())
    alert_bus = kwargs.pop("alert_bus", AlertBus())
    settings = kwargs.pop("settings", Settings())
    return TradingLoop(prob_store=prob_store, alert_bus=alert_bus, settings=settings)


async def _write_bucket(
    prob_store: ProbabilityStore,
    condition_id: str,
    day_offset: int,
    last_tick_utc: datetime,
) -> None:
    payload = BucketPayload(
        mc_prob=0.6,
        quantiles_by_fxx={1: {"p10": 60.0, "p50": 65.0, "p90": 70.0}},
        last_tick_utc=last_tick_utc,
    )
    await prob_store.write(condition_id, day_offset, {(60.0, 70.0): payload})


# ---------------------------------------------------------------------------
# Instantiation + cadence
# ---------------------------------------------------------------------------

def test_instantiation_with_fake_dependencies() -> None:
    loop = _make_loop()
    assert isinstance(loop, TradingLoop)


def test_interval_s_reads_from_settings() -> None:
    settings = _make_settings(trading_loop_interval_s=120)
    loop = _make_loop(settings=settings)
    assert loop.interval_s == 120


def test_interval_s_default_is_300() -> None:
    loop = _make_loop(settings=Settings())
    assert loop.interval_s == 300


# ---------------------------------------------------------------------------
# Staleness gate
# ---------------------------------------------------------------------------

def test_tick_returns_false_and_emits_alert_when_stale() -> None:
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings = _make_settings(staleness_threshold_min=65)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=settings)

    old_tick = datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)  # 120 min later

    async def run() -> bool:
        await _write_bucket(prob_store, "cid-1", 0, old_tick)
        result = await loop.tick("cid-1", 0, station="KLGA", now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result is False
    assert len(received) == 1
    assert received[0].alert_type == "data_stale"


def test_tick_returns_true_no_alert_when_fresh() -> None:
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings = _make_settings(staleness_threshold_min=65)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=settings)

    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    fresh_tick = datetime(2026, 7, 4, 11, 30, tzinfo=timezone.utc)  # 30 min ago

    async def run() -> bool:
        await _write_bucket(prob_store, "cid-1", 0, fresh_tick)
        result = await loop.tick("cid-1", 0, station="KLGA", now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result is True
    assert len(received) == 0


def test_tick_returns_false_and_emits_alert_when_no_data() -> None:
    """No data in ProbabilityStore -> data_stale alert emitted."""
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=Settings())

    async def run() -> bool:
        result = await loop.tick("cid-missing", 0, station="KLGA")
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result is False
    assert len(received) == 1
    assert received[0].alert_type == "data_stale"


def test_data_stale_alert_carries_station() -> None:
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings = _make_settings(staleness_threshold_min=65)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=settings)

    old_tick = datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    async def run() -> None:
        await _write_bucket(prob_store, "cid-1", 0, old_tick)
        await loop.tick("cid-1", 0, station="KJFK", now=now)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert received[0].station == "KJFK"


def test_data_stale_alert_context_has_age_min() -> None:
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings = _make_settings(staleness_threshold_min=65)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=settings)

    old_tick = datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)  # 120 min gap

    async def run() -> None:
        await _write_bucket(prob_store, "cid-1", 0, old_tick)
        await loop.tick("cid-1", 0, now=now)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert "age_min" in received[0].context
    assert abs(received[0].context["age_min"] - 120.0) < 0.01


def test_exactly_at_threshold_is_not_stale() -> None:
    """age_min == threshold -> NOT stale (spec uses strictly-greater-than)."""
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings = _make_settings(staleness_threshold_min=65)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=settings)

    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    tick = now - timedelta(minutes=65)  # exactly at threshold

    async def run() -> bool:
        await _write_bucket(prob_store, "cid-1", 0, tick)
        result = await loop.tick("cid-1", 0, now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result is True
    assert len(received) == 0


def test_tick_returns_false_and_emits_alert_when_empty_bucket_map() -> None:
    """Empty bucket map (key exists, value is {}) treated as missing data."""
    prob_store = ProbabilityStore()
    bus = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    loop = TradingLoop(prob_store=prob_store, alert_bus=bus, settings=Settings())

    async def run() -> bool:
        # Write an empty bucket map directly via internal API to simulate the edge case
        async with prob_store._lock:
            prob_store._data[("cid-empty", 0)] = {}
        result = await loop.tick("cid-empty", 0, station="KLGA")
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result is False
    assert len(received) == 1
    assert received[0].alert_type == "data_stale"


def test_no_order_submission_method_exists() -> None:
    """TradingLoop skeleton must not expose submit_order or place_order."""
    loop = _make_loop()
    assert not hasattr(loop, "submit_order")
    assert not hasattr(loop, "place_order")
