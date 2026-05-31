"""Tests for Phase 10 components: DataManager, ForecastLoop, and DB additions."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.stations import STATIONS
from core.probability_store import ProbabilityStore
from data.manager import DataManager
from data.obs.noaa_file import NoaaFileSource
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
    """Return a complete HRRR DataFrame covering fxx 1..48 (decoder window)."""
    init = pd.Timestamp("2026-03-25 12:00:00+00:00")
    fxx_vals = list(range(1, 49))
    return pd.DataFrame({
        "init_time":  [init] * 48,
        "fxx":        fxx_vals,
        "valid_time": [init + pd.Timedelta(hours=f) for f in fxx_vals],
        "TMP_2m":     [280.0] * 48,
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
    # After Cleaner.clean_hrrr, TMP_2m is converted from Kelvin to Celsius
    assert df["TMP_2m"].iloc[0] == pytest.approx(280.0 - 273.15, abs=0.01)


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
    assert len(df) == 48
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
    """fetch_iem falls back when first source returns non-empty but all-NaN raw temp."""
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
        "obs_U":           [-3.0],
        "obs_V":           [4.0],
        "skyc1":           [0.0],
        "p01i":            [0.0],
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


def test_data_manager_fetch_hrrr_stores_cleaned_data(station):
    """fetch_hrrr applies Cleaner before storing: TMP_2m in Celsius, wind features added."""
    init = pd.Timestamp("2026-03-25 12:00:00+00:00")
    fxx_vals = list(range(1, 49))
    raw_src = MagicMock()
    raw_src.fetch.return_value = {station.station_id: pd.DataFrame({
        "init_time":  [init] * 48,
        "fxx":        fxx_vals,
        "valid_time": [init + pd.Timedelta(hours=f) for f in fxx_vals],
        "TMP_2m":     [280.0] * 48,   # Kelvin from source
        "UGRD_10m":   [10.0]  * 48,   # m/s from source
        "VGRD_10m":   [5.0]   * 48,   # m/s from source
    })}
    store = MagicMock()
    store.append.return_value = 1
    mgr = DataManager([raw_src], [], store, store)
    df = mgr.fetch_hrrr(pd.Timestamp("2026-03-25 12:00:00+00:00"), station)

    assert df["TMP_2m"].iloc[0] == pytest.approx(280.0 - 273.15, abs=0.01)
    assert df["UGRD_10m"].iloc[0] == pytest.approx(10.0)
    assert "WS_10m" in df.columns
    assert df["WS_10m"].iloc[0] == pytest.approx((10.0 ** 2 + 5.0 ** 2) ** 0.5)
    appended_df = store.append.call_args[0][1]
    assert appended_df["TMP_2m"].iloc[0] == pytest.approx(280.0 - 273.15, abs=0.01)
    assert "WS_10m" in appended_df.columns


def test_data_manager_fetch_iem_stores_cleaned_data(station):
    """fetch_iem applies Cleaner before storing canonical raw obs columns."""
    raw_src = MagicMock()
    raw_src.fetch.return_value = pd.DataFrame({
        "valid_time":      [pd.Timestamp("2026-03-25 12:51:00+00:00")],
        "raw_tmpc_tenths": [8.5],
        "raw_dwpc_tenths": [3.2],
        "obs_U": [3.0],
        "obs_V": [4.0],
        "skyc1": [0.0],
        "p01i": [0.0],
    })
    store = MagicMock()
    store.append.return_value = 1
    mgr = DataManager([], [raw_src], store, store)
    df = mgr.fetch_iem(
        station,
        pd.Timestamp("2026-03-25 12:50:00+00:00"),
        pd.Timestamp("2026-03-25 12:52:00+00:00"),
    )

    assert "raw_tmpc_tenths" in df.columns
    assert df["raw_tmpc_tenths"].iloc[0] == pytest.approx(8.5)
    assert "tmpc" not in df.columns
    assert "obs_WS" in df.columns
    assert df["obs_WS"].iloc[0] == pytest.approx(5.0)
    assert df["valid_time"].iloc[0] == pd.Timestamp("2026-03-25 13:00:00+00:00")
    appended_df = store.append.call_args[0][1]
    assert "raw_tmpc_tenths" in appended_df.columns
    assert "tmpc" not in appended_df.columns
    assert appended_df["valid_time"].iloc[0] == pd.Timestamp(
        "2026-03-25 13:00:00+00:00"
    )


def test_data_manager_noaa_tick_fallback_stores_representative_hour(station):
    """NOAA tick fallback accepts H-1 :51 and stores it as the current hour."""
    bad = MagicMock()
    bad.fetch.side_effect = RuntimeError("IEM down")
    noaa = NoaaFileSource()
    txt = (
        "2026/03/25 13:51\n"
        "KLGA 251351Z 22012KT 10SM OVC055 14/03 A2979 "
        "RMK AO2 T01390028\n"
    )
    store = MagicMock()
    store.append.return_value = 1
    mgr = DataManager([], [bad, noaa], store, store)

    with patch.object(noaa._session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = txt
        mock_resp.raise_for_status = lambda: None
        mock_get.return_value = mock_resp

        df = mgr.fetch_latest_iem_tick(
            station,
            pd.Timestamp("2026-03-25 14:00:00+00:00"),
        )

    assert df["valid_time"].iloc[0] == pd.Timestamp("2026-03-25 14:00:00+00:00")
    appended_df = store.append.call_args[0][1]
    assert appended_df["valid_time"].iloc[0] == pd.Timestamp(
        "2026-03-25 14:00:00+00:00"
    )


def test_data_manager_fetch_iem_fallback(station):
    """fetch_iem tries IEM first; falls back to NOAA file source."""
    bad  = MagicMock()
    bad.fetch.side_effect = RuntimeError("IEM down")
    good = MagicMock()
    good.fetch.return_value = pd.DataFrame({
        "valid_time":      [pd.Timestamp("2026-03-25 12:00:00+00:00")],
        "raw_tmpc_tenths": [12.8],
        "obs_U":           [-3.0],
        "obs_V":           [4.0],
        "skyc1":           [0.0],
        "p01i":            [0.0],
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
    db.upsert_market("tok1", "cid1", "klga-60-70f", 60.0, 70.0, "YES")
    db.update_event_status("cid1", "active")
    rows = db.get_active_events()
    assert len(rows) == 1
    assert rows[0]["status"] == "active"


# -----------------------------------------------------------------------
# ForecastLoop.tick
# -----------------------------------------------------------------------

def _seed_event_and_market(db):
    """Pre-populate DB with one active event and one YES market.

    event_end_date is set to the same day as the test init_time (2026-03-25)
    so day_offset == 0 and the event is processed by ForecastLoop.tick().
    """
    station_id = next(iter(STATIONS))
    db.upsert_event("cid1", "q", "slug", "2026-03-25", station_id,
                    status="active")
    db.upsert_market("tok1", "cid1", "slug-60-70", 60.0, 70.0, "YES")


def test_forecast_loop_tick_no_events(db, station):
    """tick() with no active events returns empty list."""
    loop = ForecastLoop(
        pipeline   = MagicMock(),
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )
    result = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert result == []


def test_forecast_loop_tick_pipeline_failure_is_swallowed(db, station):
    """tick() survives if pipeline.run() raises; returns empty list."""
    _seed_event_and_market(db)
    broken_pipeline = MagicMock()
    broken_pipeline.run.side_effect = RuntimeError("model unavailable")
    loop = ForecastLoop(
        pipeline   = broken_pipeline,
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )
    result = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert result == []


def test_forecast_loop_tick_writes_prob_store(db, station):
    """tick() writes bucket probs and quantiles_by_fxx to ProbabilityStore."""
    _seed_event_and_market(db)
    pipeline = MagicMock()
    pipeline.run.return_value = {
        "probs":     {(60.0, 70.0): 0.70},
        "p50_high_f": 65.0,
        "quantiles": {fxx: {"p10": 1.0, "p50": 5.0, "p90": 9.0}
                      for fxx in range(1, 49)},
    }
    prob_store = ProbabilityStore()
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = prob_store,
        db         = db,
        station    = station,
    )
    written = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert written == ["cid1"]
    snapshot = asyncio.run(prob_store.read("cid1", 0))
    assert (60.0, 70.0) in snapshot.buckets
    assert abs(snapshot.buckets[(60.0, 70.0)] - 0.70) < 1e-9
    assert snapshot.quantiles_by_fxx[1]["p50"] == pytest.approx(5.0)


def test_forecast_loop_tick_skips_event_outside_day_offset(db, station):
    """Events whose end date is > 2 days from init_time are silently skipped."""
    station_id = next(iter(STATIONS))
    db.upsert_event("far_cid", "q", "slug", "2026-05-01", station_id,
                    status="active")
    db.upsert_market("far_tok", "far_cid", "slug-60-70", 60.0, 70.0, "YES")
    loop = ForecastLoop(
        pipeline   = MagicMock(),
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )
    written = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert written == []


# -----------------------------------------------------------------------
# ForecastLoop.run  (V10 -- wall-clock UTC hour alignment)
# -----------------------------------------------------------------------

def test_forecast_loop_run_aligns_to_next_utc_hour(db, station):
    """run() sleeps until the next wall-clock UTC hour boundary (within 5 s)."""
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch

    slept_for: list[float] = []

    async def _spy_sleep(secs: float) -> None:
        slept_for.append(secs)
        raise asyncio.CancelledError

    pipeline = MagicMock()
    pipeline.run.return_value = {"probs": {}, "quantiles": {}}
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )

    with patch("asyncio.sleep", new=_spy_sleep):
        try:
            asyncio.run(loop.run())
        except (asyncio.CancelledError, SystemExit):
            pass

    assert len(slept_for) == 1
    now       = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    expected  = (next_hour - now).total_seconds()
    assert abs(slept_for[0] - expected) < 5.0


def test_forecast_loop_run_calls_tick_then_sleeps(db, station):
    """run() calls tick() once before sleeping."""
    from unittest.mock import patch, AsyncMock

    call_order: list[str] = []

    async def _spy_sleep(secs: float) -> None:
        call_order.append("sleep")
        raise asyncio.CancelledError

    pipeline = MagicMock()
    pipeline.run.return_value = {"probs": {}, "quantiles": {}}
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )
    original_tick = loop.tick

    async def _spy_tick(*args, **kwargs):
        call_order.append("tick")
        return await original_tick(*args, **kwargs)

    loop.tick = _spy_tick

    with patch("asyncio.sleep", new=_spy_sleep):
        try:
            asyncio.run(loop.run())
        except (asyncio.CancelledError, SystemExit):
            pass

    assert call_order == ["tick", "sleep"]


# -----------------------------------------------------------------------
# ForecastLoop.tick -- all 3 day_offsets  (V10 acceptance criterion)
# -----------------------------------------------------------------------

def test_forecast_loop_tick_writes_all_3_day_offsets(db, station):
    """tick() writes to ProbabilityStore for day_offsets 0, 1, and 2."""
    station_id = next(iter(STATIONS))
    init_time  = pd.Timestamp("2026-03-25 12:00:00+00:00")
    today      = init_time.date()

    for day_off, cid in enumerate(("cid-d0", "cid-d1", "cid-d2")):
        end_date = str(today + pd.Timedelta(days=day_off))
        db.upsert_event(cid, "q", "slug", end_date, station_id, status="active")
        db.upsert_market(f"tok-{day_off}", cid, "slug-60-70", 60.0, 70.0, "YES")

    pipeline = MagicMock()
    pipeline.run.return_value = {
        "probs":     {(60.0, 70.0): 0.60},
        "quantiles": {},
    }
    prob_store = ProbabilityStore()
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = prob_store,
        db         = db,
        station    = station,
    )

    written = asyncio.run(loop.tick(init_time))

    assert set(written) == {"cid-d0", "cid-d1", "cid-d2"}
    for day_off, cid in enumerate(("cid-d0", "cid-d1", "cid-d2")):
        snapshot = asyncio.run(prob_store.read(cid, day_off))
        assert (60.0, 70.0) in snapshot.buckets, f"day_offset {day_off} not written"


# -----------------------------------------------------------------------
# V10 fixes -- outcome case-insensitivity and pipeline last_tick_utc
# -----------------------------------------------------------------------

def test_forecast_loop_tick_outcome_case_insensitive(db, station):
    """tick() includes buckets for markets whose outcome is 'yes' (lowercase)."""
    station_id = next(iter(STATIONS))
    db.upsert_event("cid-ci", "q", "slug", "2026-03-25", station_id, status="active")
    db.upsert_market("tok-ci", "cid-ci", "slug-60-70", 60.0, 70.0, "yes")  # lowercase

    pipeline = MagicMock()
    pipeline.run.return_value = {"probs": {(60.0, 70.0): 0.55}, "quantiles": {}}
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = ProbabilityStore(),
        db         = db,
        station    = station,
    )
    written = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    assert written == ["cid-ci"]


def test_forecast_loop_tick_pipeline_last_tick_utc_used(db, station):
    """tick() stores pipeline-returned last_tick_utc in ProbabilitySnapshot."""
    from datetime import datetime, timezone
    _seed_event_and_market(db)
    expected_ts = datetime(2026, 3, 25, 11, 45, tzinfo=timezone.utc)

    pipeline = MagicMock()
    pipeline.run.return_value = {
        "probs":         {(60.0, 70.0): 0.65},
        "quantiles":     {},
        "last_tick_utc": expected_ts,
    }
    prob_store = ProbabilityStore()
    loop = ForecastLoop(
        pipeline   = pipeline,
        prob_store = prob_store,
        db         = db,
        station    = station,
    )
    asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))
    snapshot = asyncio.run(prob_store.read("cid1", 0))
    assert snapshot.last_tick_utc == expected_ts


# -----------------------------------------------------------------------
# V2V04: inference readiness gate in ForecastLoop.tick
# -----------------------------------------------------------------------

def test_forecast_loop_skips_event_when_not_ready(db, station):
    """tick() must not call pipeline.run() when DataManager reports not-ready."""
    from data.manager import ReadinessStatus

    _seed_event_and_market(db)

    not_ready_status = ReadinessStatus(
        ready=False, reason="cold_start", first_eligible_init=None
    )
    data_manager = MagicMock()
    data_manager.inference_readiness.return_value = not_ready_status

    pipeline = MagicMock()
    loop = ForecastLoop(
        pipeline     = pipeline,
        prob_store   = ProbabilityStore(),
        db           = db,
        station      = station,
        data_manager = data_manager,
    )

    written = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))

    pipeline.run.assert_not_called()
    assert written == []


def test_forecast_loop_runs_pipeline_when_data_manager_none(db, station):
    """tick() runs pipeline normally when data_manager is None (backward-compat)."""
    _seed_event_and_market(db)

    pipeline = MagicMock()
    pipeline.run.return_value = {"probs": {(60.0, 70.0): 0.70}, "quantiles": {}}
    loop = ForecastLoop(
        pipeline     = pipeline,
        prob_store   = ProbabilityStore(),
        db           = db,
        station      = station,
        data_manager = None,
    )

    written = asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))

    pipeline.run.assert_called_once()
    assert written == ["cid1"]


def test_forecast_loop_tick_passes_target_date_to_pipeline(db, station):
    """tick() forwards event_end_date as target_date kwarg to pipeline.run()."""
    _seed_event_and_market(db)

    pipeline = MagicMock()
    pipeline.run.return_value = {"probs": {}, "quantiles": {}}
    loop = ForecastLoop(
        pipeline     = pipeline,
        prob_store   = ProbabilityStore(),
        db           = db,
        station      = station,
        data_manager = None,
    )

    asyncio.run(loop.tick(pd.Timestamp("2026-03-25 12:00:00+00:00")))

    _, kwargs = pipeline.run.call_args
    assert kwargs.get("target_date") == "2026-03-25"
