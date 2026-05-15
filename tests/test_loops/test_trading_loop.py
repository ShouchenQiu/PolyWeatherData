"""Tests for loops/trading_loop.py -- TradingLoop V11 full wiring."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from config.stations import STATIONS
from core.alerts import AlertBus, AlertEvent
from core.positions import PositionTracker
from core.probability_store import BucketPayload, ProbabilityStore
from core.risk.fixed_fraction import FixedFraction
from core.risk.pipeline import RiskPipeline
from core.strategy.edge import EdgeStrategy
from core.strategy.multi import MultiStrategy
from core.types import Signal
from db.schema import DbClient
from loops.trading_loop import TradingLoop


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    client = DbClient(":memory:")
    station = next(iter(STATIONS.values()))
    client.upsert_station(
        station.station_id, station.lat, station.lon, station.timezone
    )
    yield client
    client.close()


def _make_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_strategy() -> MultiStrategy:
    return MultiStrategy([EdgeStrategy(min_edge=0.04)])


def _make_risk() -> RiskPipeline:
    return RiskPipeline([FixedFraction()])


def _make_exchange() -> MagicMock:
    exc = MagicMock()
    exc.place_order.return_value = "oid-001"
    return exc


def _make_loop(db: DbClient, **overrides) -> TradingLoop:
    return TradingLoop(
        prob_store = overrides.pop("prob_store", ProbabilityStore()),
        alert_bus  = overrides.pop("alert_bus",  AlertBus()),
        settings   = overrides.pop("settings",   Settings()),
        strategy   = overrides.pop("strategy",   _make_strategy()),
        risk       = overrides.pop("risk",        _make_risk()),
        exchange   = overrides.pop("exchange",    _make_exchange()),
        positions  = overrides.pop("positions",   PositionTracker(db)),
        db         = overrides.pop("db",          db),
        orderbook  = overrides.pop("orderbook",   None),
    )


# ---------------------------------------------------------------------------
# Instantiation + cadence
# ---------------------------------------------------------------------------

def test_instantiation_with_valid_dependencies(db) -> None:
    loop = _make_loop(db)
    assert isinstance(loop, TradingLoop)


def test_interval_s_reads_from_settings(db) -> None:
    loop = _make_loop(db, settings=_make_settings(trading_loop_interval_s=120))
    assert loop.interval_s == 120


def test_interval_s_default_is_300(db) -> None:
    loop = _make_loop(db, settings=Settings())
    assert loop.interval_s == 300


# ---------------------------------------------------------------------------
# Constructor type enforcement  (V11)
# ---------------------------------------------------------------------------

def test_constructor_rejects_bare_base_strategy(db) -> None:
    """TypeError when strategy is not a MultiStrategy."""
    with pytest.raises(TypeError, match="MultiStrategy"):
        TradingLoop(
            prob_store = ProbabilityStore(),
            alert_bus  = AlertBus(),
            settings   = Settings(),
            strategy   = EdgeStrategy(),
            risk       = _make_risk(),
            exchange   = _make_exchange(),
            positions  = PositionTracker(db),
            db         = db,
        )


def test_constructor_rejects_bare_base_risk_model(db) -> None:
    """TypeError when risk is not a RiskPipeline."""
    with pytest.raises(TypeError, match="RiskPipeline"):
        TradingLoop(
            prob_store = ProbabilityStore(),
            alert_bus  = AlertBus(),
            settings   = Settings(),
            strategy   = _make_strategy(),
            risk       = FixedFraction(),
            exchange   = _make_exchange(),
            positions  = PositionTracker(db),
            db         = db,
        )


# ---------------------------------------------------------------------------
# Staleness gate  (ADR-0006 -- preserved from F05)
# ---------------------------------------------------------------------------

def test_tick_emits_data_stale_and_returns_empty_when_stale(db) -> None:
    """Stale data: no submission, data_stale alert emitted."""
    prob_store = ProbabilityStore()
    bus        = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings   = _make_settings(staleness_threshold_min=65)
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    old_tick   = now - timedelta(minutes=120)

    db.upsert_event("cid-stale", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-st", "cid-stale", "slug-s", 60.0, 70.0, "YES")

    exchange = MagicMock()
    loop     = TradingLoop(
        prob_store = prob_store,
        alert_bus  = bus,
        settings   = settings,
        strategy   = _make_strategy(),
        risk       = _make_risk(),
        exchange   = exchange,
        positions  = PositionTracker(db),
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.6, quantiles_by_fxx={},
                                last_tick_utc=old_tick)
        await prob_store.write("cid-stale", 0, {(60.0, 70.0): payload})
        result = await loop.tick(now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result == []
    assert any(e.alert_type == "data_stale" for e in received)
    exchange.place_order.assert_not_called()


def test_tick_no_alert_when_fresh_no_signals(db) -> None:
    """Fresh data with no signals: no alert, nothing submitted."""
    prob_store = ProbabilityStore()
    bus        = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    settings   = _make_settings(staleness_threshold_min=65)
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    fresh_tick = now - timedelta(minutes=30)

    db.upsert_event("cid-fresh", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-f", "cid-fresh", "slug-f", 60.0, 70.0, "YES")

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = bus,
        settings   = settings,
        strategy   = _make_strategy(),
        risk       = _make_risk(),
        exchange   = MagicMock(),
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = None,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.6, quantiles_by_fxx={},
                                last_tick_utc=fresh_tick)
        await prob_store.write("cid-fresh", 0, {(60.0, 70.0): payload})
        result = await loop.tick(now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result == []
    assert received == []


# ---------------------------------------------------------------------------
# Signal pipeline  (V11)
# ---------------------------------------------------------------------------

def test_tick_risk_rejected_emits_alert(db) -> None:
    """RiskPipeline rejection triggers risk_rejected alert; no exchange call."""
    prob_store = ProbabilityStore()
    bus        = AlertBus()
    received: list[AlertEvent] = []

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-r", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-r", "cid-r", "slug-r", 60.0, 70.0, "YES")

    sig = Signal(token_id="tok-r", side="BUY", mc_prob=0.8, clob_price=0.5,
                 edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-r")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = False
    exchange      = MagicMock()

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = bus,
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = PositionTracker(db),
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-r", 0, {(60.0, 70.0): payload})
        result = await loop.tick(now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result == []
    assert any(e.alert_type == "risk_rejected" for e in received)
    exchange.place_order.assert_not_called()


def test_tick_successful_submit_calls_record_order(db) -> None:
    """Approved signal: exchange order_id forwarded to PositionTracker.record_order."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-s", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-s", "cid-s", "slug-s", 60.0, 70.0, "YES")

    sig = Signal(token_id="tok-s", side="BUY", mc_prob=0.8, clob_price=0.5,
                 edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-s")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange      = MagicMock()
    exchange.place_order.return_value = "oid-123"
    mock_positions = MagicMock(spec=PositionTracker)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-s", 0, {(60.0, 70.0): payload})
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-123"]
    mock_positions.record_order.assert_called_once()
    kw = mock_positions.record_order.call_args.kwargs
    assert kw["order_id"]     == "oid-123"
    assert kw["token_id"]     == "tok-s"
    assert kw["condition_id"] == "cid-s"


def test_tick_failed_submit_no_record_order(db) -> None:
    """Exchange raises: record_order not called; exception logged, not re-raised."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-fe", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-fe", "cid-fe", "slug-fe", 60.0, 70.0, "YES")

    sig = Signal(token_id="tok-fe", side="BUY", mc_prob=0.8, clob_price=0.5,
                 edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-fe")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange      = MagicMock()
    exchange.place_order.side_effect = RuntimeError("exchange offline")
    mock_positions = MagicMock(spec=PositionTracker)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-fe", 0, {(60.0, 70.0): payload})
        return await loop.tick(now=now)   # must not raise

    result = asyncio.run(run())
    assert result == []
    mock_positions.record_order.assert_not_called()


def test_tick_yes_and_no_tokens_both_submitted(db) -> None:
    """Strategy signals for both YES and NO tokens each produce an exchange call."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-yn", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-yes", "cid-yn", "slug-yes", 60.0, 70.0, "YES")
    db.upsert_market("tok-no",  "cid-yn", "slug-no",  60.0, 70.0, "NO")

    yes_sig = Signal(token_id="tok-yes", side="BUY", mc_prob=0.8, clob_price=0.5,
                     edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-yn")
    no_sig  = Signal(token_id="tok-no",  side="BUY", mc_prob=0.2, clob_price=0.15,
                     edge=0.05, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-yn")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [yes_sig, no_sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange      = MagicMock()
    exchange.place_order.side_effect = ["oid-yes", "oid-no"]
    mock_positions = MagicMock(spec=PositionTracker)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-yn", 0, {(60.0, 70.0): payload})
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert set(result) == {"oid-yes", "oid-no"}
    assert exchange.place_order.call_count        == 2
    assert mock_positions.record_order.call_count == 2


# ---------------------------------------------------------------------------
# Fix 1 -- real exposure and open_positions forwarded to RiskPipeline
# ---------------------------------------------------------------------------

def test_tick_passes_real_exposure_and_positions_to_risk(db) -> None:
    """approve_signal receives live total_exposure and open_positions from PositionTracker."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-exp", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-exp", "cid-exp", "slug-exp", 60.0, 70.0, "YES")

    # Pre-seed one pending order using the real market + event so FK is satisfied.
    positions = PositionTracker(db)
    positions.record_order(
        order_id="pre-oid", token_id="tok-exp",
        condition_id="cid-exp", side="BUY",
        size_usd=50.0, price=0.5,
    )

    sig = Signal(token_id="tok-exp", side="BUY", mc_prob=0.8, clob_price=0.5,
                 edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-exp")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = "oid-new"

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-exp", 0, {(60.0, 70.0): payload})
        return await loop.tick(now=now)

    asyncio.run(run())

    approve_kw = mock_risk.approve_signal.call_args.kwargs
    assert approve_kw["current_exposure"] == pytest.approx(50.0)
    assert approve_kw["open_positions"] == {"tok-exp": 1}

    size_kw = mock_risk.compute_size.call_args.kwargs
    assert size_kw["current_exposure"] == pytest.approx(50.0)
    assert size_kw["open_positions"] == {"tok-exp": 1}


# ---------------------------------------------------------------------------
# Fix 3 -- empty order_id returned by exchange must not be recorded
# ---------------------------------------------------------------------------

def test_tick_empty_order_id_not_recorded(db) -> None:
    """Exchange returns empty string: record_order not called; order not in result."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-empty", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-empty", "cid-empty", "slug-e", 60.0, 70.0, "YES")

    sig = Signal(token_id="tok-empty", side="BUY", mc_prob=0.8, clob_price=0.5,
                 edge=0.3, bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-empty")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = ""   # rejected / no order id
    mock_positions = MagicMock(spec=PositionTracker)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
    )

    async def run() -> list[str]:
        payload = BucketPayload(mc_prob=0.7, quantiles_by_fxx={},
                                last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-empty", 0, {(60.0, 70.0): payload})
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == []
    mock_positions.record_order.assert_not_called()


# ---------------------------------------------------------------------------
# Candidate A -- TradingLoop.run() drives tick at interval_s
# ---------------------------------------------------------------------------

def test_run_calls_tick_and_exits_on_cancel(db) -> None:
    """run() calls tick() then sleeps; CancelledError propagates cleanly."""
    loop = _make_loop(db, settings=_make_settings(trading_loop_interval_s=99))
    tick_calls: list[int] = []
    original_tick = loop.tick

    async def _patched_tick(**kw):
        result = await original_tick(**kw)
        tick_calls.append(1)
        raise asyncio.CancelledError

    loop.tick = _patched_tick

    async def run_one() -> None:
        try:
            await loop.run()
        except asyncio.CancelledError:
            pass

    asyncio.run(run_one())
    assert len(tick_calls) == 1
