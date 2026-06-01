from __future__ import annotations

from datetime import datetime, timezone

from db.schema import DbClient
from ui import queries


def test_overview_metrics_empty_db() -> None:
    db = DbClient(":memory:")
    metrics = queries.overview_metrics(db, is_paper=True)
    assert metrics["active_events"] == 0
    assert metrics["open_orders"] == 0
    assert metrics["total_pnl"] == 0.0


def test_recent_signal_decisions_uses_current_schema() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-q", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-q", "cid-q", "slug-yes", 60.0, 70.0, "YES")
    db.record_signal(
        signal_id="sig-q",
        yes_token_id="tok-q",
        no_token_id="",
        direction="bullish",
        mc_prob=0.80,
        reference_price=0.50,
        edge=0.30,
        tick_utc="2026-07-04T12:00:00+00:00",
    )
    db.update_signal_decision(
        signal_id="sig-q",
        final_token_id="tok-q",
        final_side="BUY",
        executable_price=0.51,
        approved=1,
        reject_reason=None,
        computed_size=20.0,
    )

    df = queries.recent_signal_decisions(db)

    assert len(df) == 1
    assert set(queries.RECENT_SIGNAL_DECISION_COLUMNS) <= set(df.columns)
    assert "timestamp" not in df.columns
    assert "clob_price" not in df.columns
    assert "side" not in df.columns


def test_events_overview_filters_by_station_and_status() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_station("KORD", 41.98, -87.90, "America/Chicago")
    db.upsert_event("cid-1", "Q1", "slug-1", "2026-07-04", "KLGA", "active")
    db.upsert_event("cid-2", "Q2", "slug-2", "2026-07-04", "KORD", "settled")

    df = queries.events_overview(db, station_id="KLGA", status="active")

    assert len(df) == 1
    assert df.iloc[0]["condition_id"] == "cid-1"


def test_forecast_snapshots_join_bucket_probs() -> None:
    db = DbClient(":memory:")
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    db.upsert_event("cid-f", "Q", "slug", "2026-07-04", "KLGA", "active")
    db.upsert_market("tok-f", "cid-f", "slug-yes", 60.0, 70.0, "YES")
    sid = db.record_forecast_snapshot(
        condition_id="cid-f",
        station_id="KLGA",
        init_time="2026-07-04T12:00:00+00:00",
        event_end_date="2026-07-04",
        day_offset=0,
        tick_utc="2026-07-04T12:01:00+00:00",
        status="success",
        p50_high_f=65.0,
    )
    db.record_forecast_bucket_prob(sid, "tok-f", 60.0, 70.0, 0.73)

    df = queries.forecast_bucket_history(db, condition_id="cid-f")

    assert tuple(df.columns) == queries.FORECAST_BUCKET_HISTORY_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["mc_prob"] == 0.73
    assert df.iloc[0]["p50_high_f"] == 65.0


def test_pnl_history_filters_paper_mode() -> None:
    db = DbClient(":memory:")
    tick = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc).isoformat()
    db.record_pnl_snapshot(tick, True, 1.0, 2.0, 3.0, 10.0)
    db.record_pnl_snapshot(tick, False, 4.0, 5.0, 9.0, 20.0)

    df = queries.pnl_history(db, is_paper=True)

    assert len(df) == 1
    assert df.iloc[0]["is_paper"] == 1
    assert df.iloc[0]["total_pnl"] == 3.0


def test_empty_query_results_still_return_contract_columns() -> None:
    db = DbClient(":memory:")

    assert tuple(queries.recent_signal_decisions(db).columns) == (
        queries.RECENT_SIGNAL_DECISION_COLUMNS
    )
    assert tuple(queries.forecast_bucket_history(db).columns) == (
        queries.FORECAST_BUCKET_HISTORY_COLUMNS
    )
    assert tuple(queries.market_ladder(db).columns) == queries.MARKET_LADDER_COLUMNS
