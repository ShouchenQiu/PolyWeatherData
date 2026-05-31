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
# gross_open_notional_usd (pending + filled only)
# ---------------------------------------------------------------------------

def test_gross_open_notional_usd_sums_pending_and_filled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "BUY", 15.0, 0.55)
    tracker.mark_filled("ord-2", 0.57)
    assert abs(tracker.gross_open_notional_usd() - 35.0) < 1e-9


def test_gross_open_notional_usd_excludes_cancelled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_cancelled("ord-1")
    assert tracker.gross_open_notional_usd() == 0.0


def test_gross_open_notional_usd_excludes_settled(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_settled("ord-1", 20.0)
    assert tracker.gross_open_notional_usd() == 0.0


def test_gross_open_notional_usd_zero_when_no_orders(tracker):
    assert tracker.gross_open_notional_usd() == 0.0


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


# ---------------------------------------------------------------------------
# V2F05: record_order -- is_paper and executable_price columns
# ---------------------------------------------------------------------------

def test_record_order_writes_is_paper_false_by_default(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    row = db._conn.execute(
        "SELECT is_paper FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["is_paper"] == 0


def test_record_order_writes_is_paper_true(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55,
                         is_paper=True)
    row = db._conn.execute(
        "SELECT is_paper FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["is_paper"] == 1


def test_record_order_writes_executable_price(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55,
                         executable_price=0.57)
    row = db._conn.execute(
        "SELECT executable_price FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert abs(row["executable_price"] - 0.57) < 1e-9


def test_record_order_executable_price_defaults_none(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    row = db._conn.execute(
        "SELECT executable_price FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert row["executable_price"] is None


# ---------------------------------------------------------------------------
# V2F05: mark_filled -- shares and cash flow
# ---------------------------------------------------------------------------

def test_mark_filled_buy_writes_positive_signed_shares(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.50)
    row = db._conn.execute(
        "SELECT filled_shares, signed_shares, cash_flow_usd FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    # filled_shares = size_usd / fill_price = 20 / 0.50 = 40
    assert abs(row["filled_shares"] - 40.0) < 1e-6
    assert abs(row["signed_shares"] - 40.0) < 1e-6   # BUY = positive
    assert abs(row["cash_flow_usd"] - (-20.0)) < 1e-6  # BUY = outflow


def test_mark_filled_sell_writes_negative_signed_shares(tracker, db):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "SELL", 20.0, 0.55)
    tracker.mark_filled("ord-1", 0.50)
    row = db._conn.execute(
        "SELECT filled_shares, signed_shares, cash_flow_usd FROM orders WHERE order_id='ord-1'"
    ).fetchone()
    assert abs(row["filled_shares"] - 40.0) < 1e-6
    assert abs(row["signed_shares"] - (-40.0)) < 1e-6  # SELL = negative
    assert abs(row["cash_flow_usd"] - 20.0) < 1e-6     # SELL = inflow


# ---------------------------------------------------------------------------
# V2F05: position_shares
# ---------------------------------------------------------------------------

def test_position_shares_net_long_after_buy(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.50)
    tracker.mark_filled("ord-1", 0.50)
    assert abs(tracker.position_shares("tok-yes") - 40.0) < 1e-6


def test_position_shares_reduces_after_sell(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.50)
    tracker.mark_filled("ord-1", 0.50)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "SELL", 10.0, 0.60)
    tracker.mark_filled("ord-2", 0.60)
    # bought 40 shares, sold 10/0.60 = 16.67 shares
    expected = 40.0 - (10.0 / 0.60)
    assert abs(tracker.position_shares("tok-yes") - expected) < 1e-6


def test_position_shares_zero_when_no_fills(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.50)
    # pending, not filled
    assert tracker.position_shares("tok-yes") == pytest.approx(0.0)


def test_position_shares_zero_for_unknown_token(tracker):
    assert tracker.position_shares("tok-unknown") == pytest.approx(0.0)


def test_position_shares_is_paper_filter(tracker, db):
    """is_paper=True returns only paper shares; is_paper=False returns only live."""
    tracker.record_order("p-ord", "tok-yes", "cid-1", "BUY", 20.0, 0.50, is_paper=True)
    tracker.record_order("l-ord", "tok-yes", "cid-1", "BUY", 40.0, 0.50, is_paper=False)
    tracker.mark_filled("p-ord", 0.50)
    tracker.mark_filled("l-ord", 0.50)
    paper_shares = tracker.position_shares("tok-yes", is_paper=True)
    live_shares  = tracker.position_shares("tok-yes", is_paper=False)
    all_shares   = tracker.position_shares("tok-yes")
    assert paper_shares == pytest.approx(20.0 / 0.50)
    assert live_shares  == pytest.approx(40.0 / 0.50)
    assert all_shares   == pytest.approx(paper_shares + live_shares)

# ---------------------------------------------------------------------------
# V2F05: position_size_usd
# ---------------------------------------------------------------------------

def test_position_size_usd_after_buy(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.50)
    tracker.mark_filled("ord-1", 0.50)
    assert abs(tracker.position_size_usd("tok-yes") - 20.0) < 1e-9


def test_position_size_usd_reduces_after_sell(tracker):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 30.0, 0.50)
    tracker.mark_filled("ord-1", 0.50)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "SELL", 10.0, 0.60)
    tracker.mark_filled("ord-2", 0.60)
    assert abs(tracker.position_size_usd("tok-yes") - 20.0) < 1e-9


def test_position_size_usd_zero_for_unknown_token(tracker):
    assert tracker.position_size_usd("tok-unknown") == pytest.approx(0.0)
