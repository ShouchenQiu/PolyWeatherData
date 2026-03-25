"""Tests for core/positions.py -- PositionTracker."""
from __future__ import annotations

import pytest

from db.schema import DbClient
from core.positions import PositionTracker
from core.types import Order, Signal


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def db():
    client = DbClient(":memory:")
    # Pre-seed required foreign-key rows
    client.upsert_station("KLGA", 40.77, -73.87, "America/New_York")
    client.upsert_event(
        condition_id   = "cond-1",
        question       = "High temp",
        event_slug     = "high-temp-nyc",
        event_end_date = "2025-03-20",
        station_id     = "KLGA",
    )
    client.upsert_market(
        token_id     = "tok-yes-1",
        condition_id = "cond-1",
        market_slug  = "high-60-70f",
        bucket_lo_f  = 60.0,
        bucket_hi_f  = 70.0,
        outcome      = "Yes",
    )
    yield client
    client.close()


@pytest.fixture()
def tracker(db):
    return PositionTracker(db)


def _signal() -> Signal:
    return Signal(
        token_id     = "tok-yes-1",
        side         = "BUY",
        mc_prob      = 0.65,
        clob_price   = 0.50,
        edge         = 0.15,
        bucket_lo_f  = 60.0,
        bucket_hi_f  = 70.0,
        condition_id = "cond-1",
    )


def _order(order_id: str = "ord-001") -> Order:
    return Order(
        order_id = order_id,
        token_id = "tok-yes-1",
        side     = "BUY",
        size_usd = 20.0,
        price    = 0.50,
        status   = "pending",
    )


# -----------------------------------------------------------------------
# open
# -----------------------------------------------------------------------

def test_open_stores_signal_and_order(tracker):
    tracker.open(_signal(), _order())
    open_pos = tracker.get_open()
    assert len(open_pos) == 1
    assert open_pos[0]["order_id"] == "ord-001"


def test_open_order_status_pending(tracker):
    tracker.open(_signal(), _order())
    pos = tracker.get_open()
    assert pos[0]["status"] == "pending"


# -----------------------------------------------------------------------
# fill
# -----------------------------------------------------------------------

def test_fill_sets_filled_status(tracker, db):
    tracker.open(_signal(), _order())
    tracker.fill("ord-001", 0.52)
    # After fill, order is no longer in "pending" set
    open_pos = tracker.get_open()
    assert open_pos == []

    # Verify in raw DB
    cur = db._conn.execute(
        "SELECT status, filled_at FROM orders WHERE order_id='ord-001'"
    )
    row = cur.fetchone()
    assert row["status"] == "filled"
    assert row["filled_at"] is not None


# -----------------------------------------------------------------------
# close
# -----------------------------------------------------------------------

def test_close_updates_status(tracker, db):
    tracker.open(_signal(), _order())
    tracker.close("ord-001")
    cur = db._conn.execute(
        "SELECT status FROM orders WHERE order_id='ord-001'"
    )
    row = cur.fetchone()
    assert row["status"] == "closed"


# -----------------------------------------------------------------------
# total_exposure
# -----------------------------------------------------------------------

def test_total_exposure_sums_pending_sizes(tracker):
    tracker.open(_signal(), _order("ord-001"))
    tracker.open(_signal(), _order("ord-002"))
    assert abs(tracker.total_exposure() - 40.0) < 1e-9


def test_total_exposure_zero_when_no_open(tracker):
    assert tracker.total_exposure() == 0.0


def test_total_exposure_excludes_filled(tracker):
    tracker.open(_signal(), _order("ord-001"))
    tracker.open(_signal(), _order("ord-002"))
    tracker.fill("ord-001", 0.50)
    assert abs(tracker.total_exposure() - 20.0) < 1e-9
