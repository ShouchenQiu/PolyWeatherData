"""Tests for Phase 10 components: DataManager, ForecastLoop, and DB additions."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.stations import STATIONS
from core.exchange.paper import PaperExchange
from core.positions import PositionTracker
from core.risk.fixed_fraction import FixedFraction
from core.strategy.edge import EdgeStrategy
from core.types import Signal
from data.manager import DataManager
from db.schema import DbClient
from loops.forecast_loop import ForecastLoop


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def db():
    client = DbClient(":memory:")
    station = next(iter(STATIONS.values()))
    client.upsert_station(
        station.station_id, station.lat, station.lon, station.timezone
    )
    yield client
    client.close()


@pytest.fixture()
def station():
    return next(iter(STATIONS.values()))


# -----------------------------------------------------------------------
# DataManager
# -----------------------------------------------------------------------

def _make_hrrr_df():
    return pd.DataFrame({
        "init_time":  [pd.Timestamp("2026-03-25 12:00:00+00:00")],
        "fxx":        [1],
        "valid_time": [pd.Timestamp("2026-03-25 13:00:00+00:00")],
        "TMP_2m":     [280.0],
    })


def test_data_manager_hrrr_fallback_on_incomplete_data(tmp_path, station):
    """fetch_hrrr falls back when first source returns non-empty but all-NaN TMP_2m."""
    import numpy as np
    incomplete_src = MagicMock()
    incomplete_src.fetch.return_value = {station.station_id: pd.DataFrame({
        "init_time":  [pd.Timestamp("2026-03-25 12:00:00+00:00")] * 3,
        "fxx":        [0, 1, 2],
        "valid_time": [
            pd.Timestamp("2026-03-25 12:00:00+00:00"),
            pd.Timestamp("2026-03-25 13:00:00+00:00"),
            pd.Timestamp("2026-03-25 14:00:00+00:00"),
        ],
        "TMP_2m":     [np.nan, np.nan, np.nan],
    })}
    good_src = MagicMock()
    good_src.fetch.return_value = {station.station_id: _make_hrrr_df()}

    store = MagicMock()
    store.append.return_value = 1
    mgr = DataManager([incomplete_src, good_src], [], store, store)
    df = mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)

    incomplete_src.fetch.assert_called_once()
    good_src.fetch.assert_called_once()
    assert df["TMP_2m"].iloc[0] == pytest.approx(280.0)


def test_data_manager_uses_first_successful_source(tmp_path, station):
    """fetch_hrrr tries sources in order; succeeds on the first that returns data."""
    good_src = MagicMock()
    good_src.fetch.return_value = {station.station_id: _make_hrrr_df()}
    bad_src  = MagicMock()
    bad_src.fetch.side_effect  = RuntimeError("no data")

    store  = MagicMock()
    store.append.return_value = 1
    mgr    = DataManager([good_src, bad_src], [], store, store)
    df     = mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)
    assert len(df) == 1
    good_src.fetch.assert_called_once()
    bad_src.fetch.assert_not_called()


def test_data_manager_falls_back_to_second_source(tmp_path, station):
    """fetch_hrrr falls back to second source when first raises."""
    bad_src  = MagicMock()
    bad_src.fetch.side_effect   = RuntimeError("offline")
    good_src = MagicMock()
    good_src.fetch.return_value = {station.station_id: _make_hrrr_df()}

    store = MagicMock()
    store.append.return_value = 1
    mgr   = DataManager([bad_src, good_src], [], store, store)
    df    = mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)
    assert not df.empty
    bad_src.fetch.assert_called_once()
    good_src.fetch.assert_called_once()


def test_data_manager_raises_when_all_sources_fail(tmp_path, station):
    """fetch_hrrr raises RuntimeError if every source fails."""
    src = MagicMock()
    src.fetch.side_effect = RuntimeError("always fails")
    store = MagicMock()
    mgr   = DataManager([src], [], store, store)
    with pytest.raises(RuntimeError, match="All HRRR sources failed"):
        mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)


def test_data_manager_appends_to_store_on_success(tmp_path, station):
    """Successful fetch triggers ParquetStore.append."""
    src   = MagicMock()
    src.fetch.return_value = {station.station_id: _make_hrrr_df()}
    store = MagicMock()
    store.append.return_value = 1
    mgr   = DataManager([src], [], store, store)
    mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)
    store.append.assert_called_once()


def test_data_manager_obs_fallback_on_incomplete_data(station):
    """fetch_iem falls back when first source returns non-empty but all-NaN tmpc."""
    import numpy as np
    incomplete = MagicMock()
    incomplete.fetch.return_value = pd.DataFrame({
        "valid_time":      [pd.Timestamp("2026-03-25 12:00:00+00:00")],
        "raw_tmpc_tenths": [np.nan],
    })
    good = MagicMock()
    good.fetch.return_value = pd.DataFrame({
        "valid_time":      [pd.Timestamp("2026-03-25 12:00:00+00:00")],
        "raw_tmpc_tenths": [5.0],
    })
    store = MagicMock()
    store.append.return_value = 1
    mgr = DataManager([], [incomplete, good], store, store)
    df = mgr.fetch_iem(
        station,
        pd.Timestamp("2026-03-25 11:00:00+00:00"),
        pd.Timestamp("2026-03-25 12:00:00+00:00"),
    )
    incomplete.fetch.assert_called_once()
    good.fetch.assert_called_once()
    assert df["raw_tmpc_tenths"].iloc[0] == pytest.approx(5.0)


def test_data_manager_fetch_iem_fallback(station):
    """fetch_iem tries IEM first; falls back to NOAA file source."""
    bad  = MagicMock()
    bad.fetch.side_effect = RuntimeError("IEM down")
    good = MagicMock()
    good.fetch.return_value = pd.DataFrame({
        "valid_time":      [pd.Timestamp("2026-03-25 12:00:00+00:00")],
        "raw_tmpc_tenths": [12.8],
    })
    store = MagicMock()
    store.append.return_value = 1
    mgr  = DataManager([], [bad, good], store, store)
    df   = mgr.fetch_iem(
        station,
        pd.Timestamp("2026-03-25 11:00:00+00:00"),
        pd.Timestamp("2026-03-25 12:00:00+00:00"),
    )
    assert not df.empty


def test_data_manager_fetch_iem_raises_when_all_fail(station):
    src   = MagicMock()
    src.fetch.side_effect = RuntimeError("down")
    store = MagicMock()
    mgr   = DataManager([], [src], store, store)
    with pytest.raises(RuntimeError, match="All obs sources failed"):
        mgr.fetch_iem(
            station,
            pd.Timestamp("2026-03-25 11:00:00+00:00"),
            pd.Timestamp("2026-03-25 12:00:00+00:00"),
        )


# -----------------------------------------------------------------------
# DbClient new methods
# -----------------------------------------------------------------------

def test_get_markets_for_event(db):
    db.upsert_event("cid1", "q", "slug", "2026-04-01", next(iter(STATIONS)))
    db.upsert_market("tok1", "cid1", "slug-yes", 60.0, 70.0, "YES")
    db.upsert_market("tok2", "cid1", "slug-no",  60.0, 70.0, "NO")
    rows = db.get_markets_for_event("cid1")
    assert len(rows) == 2
    token_ids = {r["token_id"] for r in rows}
    assert {"tok1", "tok2"} == token_ids


def test_get_markets_for_event_empty(db):
    db.upsert_event("cid1", "q", "slug", "2026-04-01", next(iter(STATIONS)))
    assert db.get_markets_for_event("cid1") == []


def test_update_event_status(db):
    db.upsert_event("cid1", "q", "slug", "2026-04-01", next(iter(STATIONS)))
    db.update_event_status("cid1", "active")
    rows = db.get_active_events()
    assert len(rows) == 1
    assert rows[0]["status"] == "active"


# -----------------------------------------------------------------------
# ForecastLoop.tick
# -----------------------------------------------------------------------

def _seed_event_and_market(db):
    """Pre-populate DB with one active event and one YES market."""
    station_id = next(iter(STATIONS))
    db.upsert_event("cid1", "q", "slug", "2026-04-01", station_id,
                    status="active")
    db.upsert_market("tok1", "cid1", "slug-60-70", 60.0, 70.0, "YES")


def test_forecast_loop_tick_no_events(db, station):
    """tick() with no active events returns empty list."""
    loop = ForecastLoop(
        pipeline  = MagicMock(),
        orderbook = MagicMock(),
        strategy  = EdgeStrategy(min_edge=0.04),
        risk      = FixedFraction(position_usd=20.0),
        exchange  = PaperExchange(),
        positions = PositionTracker(db),
        db        = db,
        station   = station,
    )
    result = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert result == []


def test_forecast_loop_tick_pipeline_failure_is_swallowed(db, station):
    """tick() survives if pipeline.run() raises; returns empty list."""
    _seed_event_and_market(db)
    broken_pipeline = MagicMock()
    broken_pipeline.run.side_effect = RuntimeError("model unavailable")
    loop = ForecastLoop(
        pipeline  = broken_pipeline,
        orderbook = MagicMock(),
        strategy  = EdgeStrategy(min_edge=0.04),
        risk      = FixedFraction(position_usd=20.0),
        exchange  = PaperExchange(),
        positions = PositionTracker(db),
        db        = db,
        station   = station,
    )
    result = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert result == []


def test_forecast_loop_tick_places_order_when_signal_approved(db, station):
    """tick() places an order when the strategy returns a signal above threshold."""
    _seed_event_and_market(db)

    # Pipeline returns probs that lead to a BUY signal: mc=0.70, clob=0.50 -> edge=0.2
    pipeline   = MagicMock()
    pipeline.run.return_value = {
        "probs": {(60.0, 70.0): 0.70},
        "p50_high_f": 65.0,
    }

    # Orderbook returns mid_price=0.50
    orderbook  = MagicMock()
    orderbook.get_book.return_value = {"mid_price": 0.50, "bids": [], "asks": []}

    exchange   = PaperExchange()
    positions  = PositionTracker(db)

    loop = ForecastLoop(
        pipeline  = pipeline,
        orderbook = orderbook,
        strategy  = EdgeStrategy(min_edge=0.04),
        risk      = FixedFraction(position_usd=20.0, max_total_usd=500.0),
        exchange  = exchange,
        positions = positions,
        db        = db,
        station   = station,
    )
    orders = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert len(orders) == 1
    assert orders[0].side     == "BUY"
    assert orders[0].token_id == "tok1"


def test_forecast_loop_tick_no_order_below_edge_threshold(db, station):
    """tick() produces no orders when edge < min_edge."""
    _seed_event_and_market(db)

    pipeline = MagicMock()
    # edge = 0.53 - 0.50 = 0.03 < min_edge=0.04
    pipeline.run.return_value = {
        "probs": {(60.0, 70.0): 0.53},
        "p50_high_f": 65.0,
    }
    orderbook = MagicMock()
    orderbook.get_book.return_value = {"mid_price": 0.50, "bids": [], "asks": []}

    loop = ForecastLoop(
        pipeline  = pipeline,
        orderbook = orderbook,
        strategy  = EdgeStrategy(min_edge=0.04),
        risk      = FixedFraction(position_usd=20.0, max_total_usd=500.0),
        exchange  = PaperExchange(),
        positions = PositionTracker(db),
        db        = db,
        station   = station,
    )
    orders = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert orders == []


def test_forecast_loop_tick_risk_rejection_blocks_order(db, station):
    """tick() respects risk cap: no order when total exposure is already at cap."""
    _seed_event_and_market(db)

    pipeline = MagicMock()
    pipeline.run.return_value = {
        "probs": {(60.0, 70.0): 0.70},
        "p50_high_f": 65.0,
    }
    orderbook = MagicMock()
    orderbook.get_book.return_value = {"mid_price": 0.50, "bids": [], "asks": []}

    # position_usd=20, max_total_usd=10: first trade exceeds cap
    loop = ForecastLoop(
        pipeline  = pipeline,
        orderbook = orderbook,
        strategy  = EdgeStrategy(min_edge=0.04),
        risk      = FixedFraction(position_usd=20.0, max_total_usd=10.0),
        exchange  = PaperExchange(),
        positions = PositionTracker(db),
        db        = db,
        station   = station,
    )
    orders = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert orders == []
