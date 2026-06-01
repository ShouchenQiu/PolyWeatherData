from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from core.observability import ObservabilityRecorder
from core.positions import PositionTracker
from core.probability_store import ProbabilitySnapshot
from db.schema import DbClient


def _seed(db: DbClient) -> None:
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-o", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-o", "cid-o", "slug-yes", 60.0, 70.0, "YES")


def test_record_forecast_success_writes_snapshot_family() -> None:
    db = DbClient(":memory:")
    _seed(db)
    recorder = ObservabilityRecorder(db)
    snapshot = ProbabilitySnapshot(
        buckets={(60.0, 70.0): 0.71},
        quantiles_by_fxx={1: {"p10": 61.0, "p50": 65.0, "p90": 69.0}},
        last_tick_utc=datetime(2026, 7, 4, 12, 1, tzinfo=timezone.utc),
    )

    snapshot_id = recorder.record_forecast_success(
        event={"condition_id": "cid-o", "station_id": "KLGA", "event_end_date": "2026-07-04"},
        init_utc=pd.Timestamp("2026-07-04T12:00:00Z"),
        day_offset=0,
        markets=[{"token_id": "tok-o", "bucket_lo_f": 60.0, "bucket_hi_f": 70.0, "outcome": "YES"}],
        result={"p50_high_f": 65.0},
        snapshot=snapshot,
    )

    bucket = db.query("SELECT * FROM forecast_bucket_probs WHERE snapshot_id=?", (snapshot_id,))[0]
    quantile = db.query("SELECT * FROM forecast_quantiles WHERE snapshot_id=?", (snapshot_id,))[0]
    assert bucket["token_id"] == "tok-o"
    assert bucket["mc_prob"] == pytest.approx(0.71)
    assert quantile["p50"] == pytest.approx(65.0)


def test_record_market_book_returns_mark_price_and_writes_snapshot() -> None:
    db = DbClient(":memory:")
    _seed(db)
    recorder = ObservabilityRecorder(db)

    mark = recorder.record_market_book(
        token_id="tok-o",
        condition_id="cid-o",
        tick_utc=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc),
        book={
            "bids": [{"price": 0.48, "size": 10.0}],
            "asks": [{"price": 0.52, "size": 12.0}],
        },
    )

    row = db.query("SELECT * FROM market_price_snapshots")[0]
    assert mark == pytest.approx(0.50)
    assert row["mid_price"] == pytest.approx(0.50)


def test_record_trading_tick_writes_pnl_and_loop_tick() -> None:
    db = DbClient(":memory:")
    _seed(db)
    positions = PositionTracker(db)
    positions.record_order("ord-o", "tok-o", "cid-o", "BUY", 10.0, 0.5, is_paper=True)

    ObservabilityRecorder(db).record_trading_tick(
        tick_utc=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc),
        duration_ms=10.0,
        events_processed=1,
        signals_processed=2,
        orders_submitted=0,
        positions=positions,
        is_paper=True,
        mark_prices={},
    )

    pnl = db.query("SELECT * FROM pnl_snapshots")[0]
    tick = db.query("SELECT * FROM loop_ticks WHERE loop_name='TradingLoop'")[0]
    assert pnl["gross_exposure"] == pytest.approx(10.0)
    assert tick["signals_processed"] == 2
