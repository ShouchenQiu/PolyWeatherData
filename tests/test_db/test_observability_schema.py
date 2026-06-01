from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from db.schema import DbClient


def _seed_market(db: DbClient) -> None:
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-obs", "Q", "event-slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-yes", "cid-obs", "market-yes", 60.0, 70.0, "YES")


def test_observability_tables_created() -> None:
    db = DbClient(":memory:")
    names = {
        row["name"]
        for row in db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    }
    assert {
        "loop_ticks",
        "forecast_snapshots",
        "forecast_bucket_probs",
        "forecast_quantiles",
        "market_price_snapshots",
        "pnl_snapshots",
    } <= names


def test_record_loop_tick_round_trip() -> None:
    db = DbClient(":memory:")
    tick_id = db.record_loop_tick(
        loop_name="ForecastLoop",
        tick_utc="2026-07-04T12:00:00+00:00",
        status="success",
        duration_ms=125.5,
        events_processed=2,
        signals_processed=0,
        orders_submitted=0,
    )

    row = db.query("SELECT * FROM loop_ticks WHERE tick_id=?", (tick_id,))[0]
    assert row["loop_name"] == "ForecastLoop"
    assert row["status"] == "success"
    assert row["events_processed"] == 2


def test_forecast_snapshot_bucket_and_quantile_round_trip() -> None:
    db = DbClient(":memory:")
    _seed_market(db)

    snapshot_id = db.record_forecast_snapshot(
        condition_id="cid-obs",
        station_id="KLGA",
        init_time="2026-07-04T12:00:00+00:00",
        event_end_date="2026-07-04",
        day_offset=0,
        tick_utc="2026-07-04T12:01:00+00:00",
        status="success",
        p50_high_f=65.0,
    )
    db.record_forecast_bucket_prob(
        snapshot_id=snapshot_id,
        token_id="tok-yes",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        mc_prob=0.72,
    )
    db.record_forecast_quantile(
        snapshot_id=snapshot_id,
        fxx=1,
        p10=61.0,
        p50=65.0,
        p90=69.0,
    )

    bucket = db.query(
        "SELECT * FROM forecast_bucket_probs WHERE snapshot_id=?", (snapshot_id,)
    )[0]
    quantile = db.query(
        "SELECT * FROM forecast_quantiles WHERE snapshot_id=?", (snapshot_id,)
    )[0]
    assert bucket["token_id"] == "tok-yes"
    assert bucket["mc_prob"] == pytest.approx(0.72)
    assert quantile["fxx"] == 1
    assert quantile["p50"] == pytest.approx(65.0)


def test_forecast_bucket_prob_requires_snapshot_parent() -> None:
    db = DbClient(":memory:")
    _seed_market(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.record_forecast_bucket_prob(
            snapshot_id=999,
            token_id="tok-yes",
            bucket_lo_f=60.0,
            bucket_hi_f=70.0,
            mc_prob=0.5,
        )


def test_record_market_price_and_pnl_snapshot_round_trip() -> None:
    db = DbClient(":memory:")
    _seed_market(db)
    tick = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc).isoformat()

    db.record_market_price_snapshot(
        token_id="tok-yes",
        condition_id="cid-obs",
        tick_utc=tick,
        best_bid=0.48,
        best_ask=0.52,
        mid_price=0.50,
        spread=0.04,
        depth_bid=100.0,
        depth_ask=120.0,
    )
    db.record_pnl_snapshot(
        tick_utc=tick,
        is_paper=True,
        realized_pnl=1.0,
        unrealized_pnl=-0.5,
        total_pnl=0.5,
        gross_exposure=25.0,
    )

    price = db.query("SELECT * FROM market_price_snapshots")[0]
    pnl = db.query("SELECT * FROM pnl_snapshots")[0]
    assert price["mid_price"] == pytest.approx(0.50)
    assert pnl["is_paper"] == 1
    assert pnl["total_pnl"] == pytest.approx(0.5)
