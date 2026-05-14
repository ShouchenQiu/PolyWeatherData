"""Tests for core/positions.py -- PositionTracker F03 lifecycle methods."""
from __future__ import annotations

import pytest

from db.schema import DbClient
from core.positions import PositionTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    client = DbClient(":memory:")
    client.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    client.upsert_event(
        condition_id="cid-1",
        question="High temp",
        event_slug="high-temp-nyc",
        event_end_date="2026-07-04",
        station_id="KLGA",
    )
    client.upsert_market(
        token_id="tok-yes",
        condition_id="cid-1",
        market_slug="high-60-70f",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        outcome="YES",
    )
    client.upsert_market(
        token_id="tok-yes-2",
        condition_id="cid-1",
        market_slug="high-70-80f",
        bucket_lo_f=70.0,
        bucket_hi_f=80.0,
        outcome="YES",
    )
    yield client
    client.close()


@pytest.fixture()
def tracker(db):
    return PositionTracker(db)


# ---------------------------------------------------------------------------
# record_order
# ---------------------------------------------------------------------------

def test_record_order_inserts_pending_row(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    row = db._conn.execute(
        "SELECT status, fill_price, filled_at, closed_at, cancelled_at, settled_payout"
        " FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row is not None
    assert row["status"] == "pending"
    assert row["fill_price"] is None
    assert row["filled_at"] is None
    assert row["closed_at"] is None
    assert row["cancelled_at"] is None
    assert row["settled_payout"] is None


def test_record_order_created_at_defaults_to_utc(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    row = db._conn.execute(
        "SELECT created_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["created_at"] is not None


def test_record_order_accepts_explicit_created_at(tracker, db):
    tracker.record_order(
        "ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55,
        created_at="2026-07-04T12:00:00+00:00",
    )
    row = db._conn.execute(
        "SELECT created_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["created_at"] == "2026-07-04T12:00:00+00:00"


# ---------------------------------------------------------------------------
# mark_filled
# ---------------------------------------------------------------------------

def test_mark_filled_sets_status_and_fill_price(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.57)
    row = db._conn.execute(
        "SELECT status, fill_price, filled_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["status"] == "filled"
    assert abs(row["fill_price"] - 0.57) < 1e-9
    assert row["filled_at"] is not None


def test_mark_filled_accepts_explicit_filled_at(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.57, filled_at="2026-07-04T12:05:00+00:00")
    row = db._conn.execute(
        "SELECT filled_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["filled_at"] == "2026-07-04T12:05:00+00:00"


# ---------------------------------------------------------------------------
# mark_closed
# ---------------------------------------------------------------------------

def test_mark_closed_sets_status_and_closed_at(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.57)
    tracker.mark_closed("ord-1")
    row = db._conn.execute(
        "SELECT status, closed_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["status"] == "closed"
    assert row["closed_at"] is not None


def test_mark_closed_accepts_explicit_closed_at(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_closed("ord-1", closed_at="2026-07-04T18:00:00+00:00")
    row = db._conn.execute(
        "SELECT closed_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["closed_at"] == "2026-07-04T18:00:00+00:00"


# ---------------------------------------------------------------------------
# mark_cancelled
# ---------------------------------------------------------------------------

def test_mark_cancelled_sets_status(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    row = db._conn.execute(
        "SELECT status, cancelled_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["status"] == "cancelled"
    assert row["cancelled_at"] is not None


def test_mark_cancelled_accepts_explicit_cancelled_at(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1", cancelled_at="2026-07-04T13:00:00+00:00")
    row = db._conn.execute(
        "SELECT cancelled_at FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["cancelled_at"] == "2026-07-04T13:00:00+00:00"


# ---------------------------------------------------------------------------
# mark_settled
# ---------------------------------------------------------------------------

def test_mark_settled_sets_status_and_payout(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.57)
    tracker.mark_closed("ord-1")
    tracker.mark_settled("ord-1", settled_payout=20.0)
    row = db._conn.execute(
        "SELECT status, settled_payout FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["status"] == "settled"
    assert abs(row["settled_payout"] - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Full lifecycle: pending -> filled -> closed -> settled
# ---------------------------------------------------------------------------

def test_full_lifecycle(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    assert db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()["status"] == "pending"

    tracker.mark_filled("ord-1", 0.57)
    assert db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()["status"] == "filled"

    tracker.mark_closed("ord-1")
    assert db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()["status"] == "closed"

    tracker.mark_settled("ord-1", 20.0)
    assert db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()["status"] == "settled"


# ---------------------------------------------------------------------------
# Cancellation path
# ---------------------------------------------------------------------------

def test_cancellation_path(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    assert db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-1'"
    ).fetchone()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# open_positions_by_token
# ---------------------------------------------------------------------------

def test_open_positions_by_token_counts_pending_and_filled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-2", 0.57)
    result = tracker.open_positions_by_token()
    assert result.get("tok-yes", 0) == 2


def test_open_positions_by_token_excludes_cancelled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    result = tracker.open_positions_by_token()
    assert result.get("tok-yes", 0) == 0


def test_open_positions_by_token_excludes_closed(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_closed("ord-1")
    result = tracker.open_positions_by_token()
    assert result.get("tok-yes", 0) == 0


def test_open_positions_by_token_mixed_tokens(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.record_order("ord-2", "tok-yes-2", "cid-1", "BUY", 20.0, 0.55)
    result = tracker.open_positions_by_token()
    assert result.get("tok-yes", 0) == 1
    assert result.get("tok-yes-2", 0) == 1


def test_open_positions_by_token_empty_when_no_orders(tracker):
    assert tracker.open_positions_by_token() == {}


# ---------------------------------------------------------------------------
# total_exposure (pending + filled only)
# ---------------------------------------------------------------------------

def test_total_exposure_sums_pending_and_filled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "BUY", 15.0, 0.55)
    tracker.mark_filled("ord-2", 0.57)
    assert abs(tracker.total_exposure() - 35.0) < 1e-9


def test_total_exposure_excludes_cancelled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    assert tracker.total_exposure() == 0.0


def test_total_exposure_excludes_settled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_settled("ord-1", 20.0)
    assert tracker.total_exposure() == 0.0


def test_total_exposure_zero_when_no_orders(tracker):
    assert tracker.total_exposure() == 0.0


# ---------------------------------------------------------------------------
# exposure_by_condition
# ---------------------------------------------------------------------------

def test_exposure_by_condition_sums_matching_rows(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.record_order("ord-2", "tok-yes-2", "cid-1", "BUY", 15.0, 0.55)
    assert abs(tracker.exposure_by_condition("cid-1") - 35.0) < 1e-9


def test_exposure_by_condition_excludes_cancelled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    assert tracker.exposure_by_condition("cid-1") == 0.0


def test_exposure_by_condition_zero_for_other_condition(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    assert tracker.exposure_by_condition("cid-other") == 0.0
