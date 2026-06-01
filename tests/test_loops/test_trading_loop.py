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
from core.probability_store import ProbabilitySnapshot, ProbabilityStore
from core.risk.fixed_fraction import FixedFraction
from core.risk.pipeline import RiskPipeline
from core.strategy.edge import EdgeStrategy
from core.strategy.multi import MultiStrategy
from core.types import DirectionalSignal
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


def _make_orderbook(ask: float = 0.50) -> MagicMock:
    """Return a mock orderbook that returns a deep L2 book at *ask* price."""
    book = MagicMock()
    bid  = max(0.01, ask - 0.02)
    book.get_book.return_value = {
        "bids": [{"price": bid,  "size": 1000.0}],
        "asks": [{"price": ask,  "size": 1000.0}],
    }
    return book


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
        orderbook  = overrides.pop("orderbook",   _make_orderbook()),
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
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.6},
                                       quantiles_by_fxx={}, last_tick_utc=old_tick)
        await prob_store.write("cid-stale", 0, snapshot)
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
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.6},
                                       quantiles_by_fxx={}, last_tick_utc=fresh_tick)
        await prob_store.write("cid-fresh", 0, snapshot)
        result = await loop.tick(now=now)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    assert result == []
    assert received == []


def test_tick_orderbook_unavailable_rejects_signal(db) -> None:
    """When no orderbook is wired, signals are rejected with orderbook_unavailable."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-nobook", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-nobook", "cid-nobook", "sl-nobook", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-nobook", no_token_id="", direction="bullish",
        model_prob=0.80, reference_price=0.50, edge=0.30,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-nobook")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    exchange  = MagicMock()

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = None,
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-nobook", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == []
    exchange.place_order.assert_not_called()
    mock_risk.approve_signal.assert_not_called()

    row = db._conn.execute(
        "SELECT reject_reason FROM signals WHERE yes_token_id=?",
        ("tok-nobook",),
    ).fetchone()
    assert row is not None
    assert row["reject_reason"] == "orderbook_unavailable"


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

    sig = DirectionalSignal(
        yes_token_id="tok-r", no_token_id="", direction="bullish",
        model_prob=0.8, reference_price=0.5, edge=0.3,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-r")
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
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-r", 0, snapshot)
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

    sig = DirectionalSignal(
        yes_token_id="tok-s", no_token_id="", direction="bullish",
        model_prob=0.8, reference_price=0.5, edge=0.3,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-s")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange      = MagicMock()
    exchange.place_order.return_value = "oid-123"
    positions = PositionTracker(db)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-s", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-123"]
    row = db._conn.execute(
        "SELECT token_id, condition_id FROM orders WHERE order_id=?",
        ("oid-123",),
    ).fetchone()
    assert row is not None
    assert row["token_id"]     == "tok-s"
    assert row["condition_id"] == "cid-s"


def test_tick_failed_submit_no_record_order(db) -> None:
    """Exchange raises: record_order not called; exception logged, not re-raised."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-fe", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-fe", "cid-fe", "slug-fe", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-fe", no_token_id="", direction="bullish",
        model_prob=0.8, reference_price=0.5, edge=0.3,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-fe")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk     = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange      = MagicMock()
    exchange.place_order.side_effect = RuntimeError("exchange offline")
    mock_positions = MagicMock(spec=PositionTracker)
    mock_positions.gross_open_notional_usd.return_value = 0.0
    mock_positions.position_shares.return_value = 0.0

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-fe", 0, snapshot)
        return await loop.tick(now=now)   # must not raise

    result = asyncio.run(run())
    assert result == []
    mock_positions.record_order.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 1 -- real exposure forwarded to RiskPipeline
# ---------------------------------------------------------------------------

def test_tick_passes_real_exposure_to_risk(db) -> None:
    """approve_signal receives live gross_open_notional_usd from PositionTracker."""
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

    sig = DirectionalSignal(
        yes_token_id="tok-exp", no_token_id="", direction="bullish",
        model_prob=0.8, reference_price=0.5, edge=0.3,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-exp")
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
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-exp", 0, snapshot)
        return await loop.tick(now=now)

    asyncio.run(run())

    approve_kw = mock_risk.approve_signal.call_args.kwargs
    assert approve_kw["current_exposure"] == pytest.approx(50.0)

    size_kw = mock_risk.compute_size.call_args.kwargs
    assert size_kw["current_exposure"] == pytest.approx(50.0)


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

    sig = DirectionalSignal(
        yes_token_id="tok-empty", no_token_id="", direction="bullish",
        model_prob=0.8, reference_price=0.5, edge=0.3,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-empty")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = ""   # rejected / no order id
    mock_positions = MagicMock(spec=PositionTracker)
    mock_positions.gross_open_notional_usd.return_value = 0.0
    mock_positions.position_shares.return_value = 0.0

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-empty", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == []
    mock_positions.record_order.assert_not_called()



# ---------------------------------------------------------------------------
# V2V08 -- directional arbitration paths
# ---------------------------------------------------------------------------

def _deep_book(ask: float, bid: float | None = None) -> dict:
    if bid is None:
        bid = max(0.01, ask - 0.02)
    return {
        "bids": [{"price": bid,  "size": 1000.0}],
        "asks": [{"price": ask,  "size": 1000.0}],
    }


def test_tick_bullish_signal_buys_yes_token(db) -> None:
    """Bullish DirectionalSignal -> TradingLoop submits BUY order for YES token."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-b1", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-yes-b1", "cid-b1", "slug-yes-b1", 60.0, 70.0, "YES")
    db.upsert_market("tok-no-b1",  "cid-b1", "slug-no-b1",  60.0, 70.0, "NO")

    sig = DirectionalSignal(
        yes_token_id="tok-yes-b1", no_token_id="tok-no-b1", direction="bullish",
        model_prob=0.75, reference_price=0.50, edge=0.25,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-b1")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = "oid-yes-b1"
    positions = PositionTracker(db)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-b1", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-yes-b1"]
    call_kw = exchange.place_order.call_args.kwargs
    assert call_kw["token_id"] == "tok-yes-b1"
    assert call_kw["side"]     == "BUY"


def test_tick_bearish_without_yes_position_buys_no_token(db) -> None:
    """Bearish signal + no YES inventory -> BUY NO token."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-b2", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-yes-b2", "cid-b2", "slug-yes-b2", 60.0, 70.0, "YES")
    db.upsert_market("tok-no-b2",  "cid-b2", "slug-no-b2",  60.0, 70.0, "NO")

    sig = DirectionalSignal(
        yes_token_id="tok-yes-b2", no_token_id="tok-no-b2", direction="bearish",
        model_prob=0.30, reference_price=0.55, edge=0.25,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-b2")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = "oid-no-b2"
    positions = PositionTracker(db)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-b2", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-no-b2"]
    call_kw = exchange.place_order.call_args.kwargs
    assert call_kw["token_id"] == "tok-no-b2"
    assert call_kw["side"]     == "BUY"


def test_tick_bearish_with_yes_position_sells_yes_token(db) -> None:
    """Bearish signal + existing YES inventory -> SELL YES token."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-b3", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-yes-b3", "cid-b3", "slug-yes-b3", 60.0, 70.0, "YES")
    db.upsert_market("tok-no-b3",  "cid-b3", "slug-no-b3",  60.0, 70.0, "NO")

    sig = DirectionalSignal(
        yes_token_id="tok-yes-b3", no_token_id="tok-no-b3", direction="bearish",
        model_prob=0.30, reference_price=0.55, edge=0.25,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-b3")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 15.0
    exchange = MagicMock()
    exchange.place_order.return_value = "oid-sell-b3"
    positions = PositionTracker(db)
    # Seed YES inventory so PositionTracker.position_shares() returns > 0.
    db._conn.execute(
        "INSERT INTO orders(order_id,token_id,condition_id,side,size_usd,price,signed_shares)"
        " VALUES(?,?,?,?,?,?,?)",
        ("seed-yes-b3", "tok-yes-b3", "cid-b3", "BUY", 30.0, 0.55, 54.5),
    )
    db._conn.commit()

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
        orderbook  = _make_orderbook(ask=0.60),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-b3", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-sell-b3"]
    call_kw = exchange.place_order.call_args.kwargs
    assert call_kw["token_id"] == "tok-yes-b3"
    assert call_kw["side"]     == "SELL"


# ---------------------------------------------------------------------------
# V2V08 -- re-quote rejection
# ---------------------------------------------------------------------------

def test_tick_requote_skips_when_stale_edge(db) -> None:
    """Re-quote VWAP moves close to model_prob so edge < min_edge: order not placed.

    The market book shows ask=0.57 at tick time. Initial depth check passes
    (complete_fill). Re-quote at actual size also completes, but edge
    model_prob(0.60) - vwap(0.57) = 0.03 < min_edge(0.04) -> stale_edge skip.
    """
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    settings   = _make_settings(min_edge=0.04, quote_size_usd=50.0)

    db.upsert_event("cid-rq1", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-rq1", "cid-rq1", "sl-rq1", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-rq1", no_token_id="", direction="bullish",
        model_prob=0.60, reference_price=0.50, edge=0.10,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-rq1")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    mock_positions = MagicMock(spec=PositionTracker)
    mock_positions.gross_open_notional_usd.return_value = 0.0
    mock_positions.position_shares.return_value = 0.0

    # Book has ask=0.57 (1000 shares). Both initial check and re-quote use it.
    # vwap=0.57; edge=0.60-0.57=0.03 < min_edge=0.04 -> stale_edge rejection.
    mock_orderbook = MagicMock()
    mock_orderbook.get_book.return_value = _deep_book(0.57)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = settings,
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
        orderbook  = mock_orderbook,
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-rq1", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == []
    exchange.place_order.assert_not_called()


def test_tick_requote_skips_when_insufficient_depth(db) -> None:
    """Re-quote at actual size has complete_fill=False: order not placed.

    The market book has only 90 shares at ask=0.50 (worth $45).
    quote_size_usd=$20 fills fine (needs 40 shares). But compute_size
    returns $50 which needs 100 shares -> complete_fill=False on re-quote.
    """
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    settings   = _make_settings(min_edge=0.04, quote_size_usd=20.0)

    db.upsert_event("cid-rq2", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-rq2", "cid-rq2", "sl-rq2", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-rq2", no_token_id="", direction="bullish",
        model_prob=0.70, reference_price=0.50, edge=0.20,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-rq2")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 50.0  # larger than available depth ($45)
    exchange = MagicMock()
    mock_positions = MagicMock(spec=PositionTracker)
    mock_positions.gross_open_notional_usd.return_value = 0.0
    mock_positions.position_shares.return_value = 0.0

    # 90 shares at ask=0.50 -> $45 available.
    # Initial check at $20 (40 shares needed, 90 available): complete_fill=True.
    # Re-quote at $50 (100 shares needed, 90 available): complete_fill=False.
    mock_orderbook = MagicMock()
    mock_orderbook.get_book.return_value = {
        "bids": [{"price": 0.48, "size": 90.0}],
        "asks": [{"price": 0.50, "size": 90.0}],
    }

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = settings,
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = mock_positions,
        db         = db,
        orderbook  = mock_orderbook,
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-rq2", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == []
    exchange.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# V2V08 -- per-tick local exposure update
# ---------------------------------------------------------------------------

def test_tick_local_exposure_updates_after_each_order(db) -> None:
    """Second signal's risk call sees exposure increased by the first order."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-exp2", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-e1", "cid-exp2", "sl-e1", 60.0, 70.0, "YES")
    db.upsert_market("tok-e2", "cid-exp2", "sl-e2", 70.0, 80.0, "YES")

    sig1 = DirectionalSignal(
        yes_token_id="tok-e1", no_token_id="", direction="bullish",
        model_prob=0.70, reference_price=0.50, edge=0.20,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-exp2")
    sig2 = DirectionalSignal(
        yes_token_id="tok-e2", no_token_id="", direction="bullish",
        model_prob=0.80, reference_price=0.55, edge=0.25,
        bucket_lo_f=70.0, bucket_hi_f=80.0, condition_id="cid-exp2")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig1, sig2]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 25.0
    exchange = MagicMock()
    exchange.place_order.side_effect = ["oid-e1", "oid-e2"]

    # Seed $10 of pre-existing open exposure in a third market (tok-e0).
    db.upsert_market("tok-e0", "cid-exp2", "sl-e0", 55.0, 60.0, "YES")
    db._conn.execute(
        "INSERT INTO orders(order_id,token_id,condition_id,side,size_usd,"
        "price,status,is_paper) VALUES(?,?,?,?,?,?,?,?)",
        ("seed-exp", "tok-e0", "cid-exp2", "BUY", 10.0, 0.50, "pending", 0),
    )
    db._conn.commit()
    positions = PositionTracker(db)

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = positions,
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(
            buckets={(60.0, 70.0): 0.7, (70.0, 80.0): 0.8},
            quantiles_by_fxx={},
            last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-exp2", 0, snapshot)
        return await loop.tick(now=now)

    asyncio.run(run())

    calls = mock_risk.approve_signal.call_args_list
    assert len(calls) == 2
    # First call sees initial exposure = 10.0
    assert calls[0].kwargs["current_exposure"] == pytest.approx(10.0)
    # Second call sees 10.0 + 25.0 = 35.0 (updated after first order)
    assert calls[1].kwargs["current_exposure"] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# V2V08 -- signal DB lifecycle
# ---------------------------------------------------------------------------

def test_tick_signal_written_to_db_before_arbitration(db) -> None:
    """Pending signal row is written to DB before risk evaluation."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-db1", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-db1", "cid-db1", "sl-db1", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-db1", no_token_id="", direction="bullish",
        model_prob=0.70, reference_price=0.50, edge=0.20,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-db1")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = False  # rejected so no order placed

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = MagicMock(),
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> None:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-db1", 0, snapshot)
        await loop.tick(now=now)

    asyncio.run(run())

    row = db._conn.execute(
        "SELECT direction, mc_prob, edge FROM signals WHERE yes_token_id=?",
        ("tok-db1",),
    ).fetchone()
    assert row is not None
    assert row["direction"] == "bullish"
    assert abs(row["mc_prob"] - 0.70) < 1e-9


def test_tick_signal_decision_updated_after_rejection(db) -> None:
    """After risk rejection, signal row has approved=0 and reject_reason set."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-db2", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-db2", "cid-db2", "sl-db2", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-db2", no_token_id="", direction="bullish",
        model_prob=0.70, reference_price=0.50, edge=0.20,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-db2")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = False

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = MagicMock(),
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> None:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-db2", 0, snapshot)
        await loop.tick(now=now)

    asyncio.run(run())

    row = db._conn.execute(
        "SELECT approved, reject_reason FROM signals WHERE yes_token_id=?",
        ("tok-db2",),
    ).fetchone()
    assert row is not None
    assert row["approved"]      == 0
    assert row["reject_reason"] == "risk_rejected"


def test_tick_signal_order_linkage(db) -> None:
    """After a successful order, signal row has approved=1 and order_id linked."""
    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-db3", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-db3", "cid-db3", "sl-db3", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-db3", no_token_id="", direction="bullish",
        model_prob=0.70, reference_price=0.50, edge=0.20,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-db3")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 20.0
    exchange = MagicMock()
    exchange.place_order.return_value = "oid-linked"

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = exchange,
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = _make_orderbook(),
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-db3", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert result == ["oid-linked"]

    row = db._conn.execute(
        "SELECT approved, order_id, computed_size FROM signals WHERE yes_token_id=?",
        ("tok-db3",),
    ).fetchone()
    assert row is not None
    assert row["approved"]      == 1
    assert row["order_id"]      == "oid-linked"
    assert abs(row["computed_size"] - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# V2V09 -- paper exchange integration (is_paper + executable_price persistence)
# ---------------------------------------------------------------------------

def test_tick_paper_order_recorded_with_is_paper_and_executable_price(db) -> None:
    """TradingLoop records is_paper=True, executable_price, status='filled',
    fill_price, and filled_shares for paper orders (V2V09 acceptance criteria).

    A two-level ask book is used so that the VWAP is demonstrably not a
    round single-price number, proving that compute_quote() contributed to
    the fill_price persisted in the DB.
    """
    from core.exchange.paper import PaperExchange
    from core.pricing.quote import compute_quote

    # Two ask levels with ample depth: $10 at 0.50, $200 at 0.60.
    # Initial probe (quote_size_usd=50.0) fills entirely within this range.
    # Actual size (15.0): fully consumes L1 ($10), then $5 at 0.60
    # VWAP = 15.0 / (20 + 8.333...) ~= 0.5294
    two_level_book = {
        "bids": [{"price": "0.48", "size": "1000"}],
        "asks": [
            {"price": "0.50", "size": "20"},
            {"price": "0.60", "size": "1000"},
        ],
    }
    expected_quote = compute_quote(two_level_book, "BUY", 15.0)
    assert expected_quote.complete_fill
    expected_vwap  = expected_quote.vwap_price

    prob_store = ProbabilityStore()
    station_id = next(iter(STATIONS))
    now        = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    db.upsert_event("cid-paper", "q", "slug", "2026-07-04",
                    station_id, status="active")
    db.upsert_market("tok-paper", "cid-paper", "sl-paper", 60.0, 70.0, "YES")

    sig = DirectionalSignal(
        yes_token_id="tok-paper", no_token_id="", direction="bullish",
        model_prob=0.75, reference_price=0.50, edge=0.25,
        bucket_lo_f=60.0, bucket_hi_f=70.0, condition_id="cid-paper")
    mock_strategy = MagicMock(spec=MultiStrategy)
    mock_strategy.compute_signals.return_value = [sig]
    mock_risk = MagicMock(spec=RiskPipeline)
    mock_risk.approve_signal.return_value = True
    mock_risk.compute_size.return_value   = 15.0

    mock_orderbook = MagicMock()
    mock_orderbook.get_book.return_value = two_level_book
    paper_exchange = PaperExchange()

    loop = TradingLoop(
        prob_store = prob_store,
        alert_bus  = AlertBus(),
        settings   = Settings(),
        strategy   = mock_strategy,
        risk       = mock_risk,
        exchange   = paper_exchange,
        positions  = PositionTracker(db),
        db         = db,
        orderbook  = mock_orderbook,
        is_paper   = True,
    )

    async def run() -> list[str]:
        snapshot = ProbabilitySnapshot(buckets={(60.0, 70.0): 0.7},
                                       quantiles_by_fxx={},
                                       last_tick_utc=now - timedelta(minutes=5))
        await prob_store.write("cid-paper", 0, snapshot)
        return await loop.tick(now=now)

    result = asyncio.run(run())
    assert len(result) == 1

    row = db._conn.execute(
        "SELECT is_paper, executable_price, status, fill_price, filled_shares"
        " FROM orders WHERE order_id=?",
        (result[0],),
    ).fetchone()
    assert row is not None
    assert row["is_paper"]         == 1
    assert row["status"]           == "filled"
    assert row["executable_price"] == pytest.approx(expected_vwap, rel=1e-6)
    assert row["fill_price"]       == pytest.approx(expected_vwap, rel=1e-6)
    assert row["filled_shares"]    == pytest.approx(15.0 / expected_vwap, rel=1e-4)


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
