"""Tests for core/pnl.py -- PnLCalculator V2F05."""
from __future__ import annotations

import pytest

from db.schema import DbClient
from core.pnl import PnLCalculator
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
    client.upsert_event(
        condition_id="cid-2",
        question="High temp B",
        event_slug="high-temp-nyc-b",
        event_end_date="2026-07-05",
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
        condition_id="cid-2",
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


@pytest.fixture()
def calc(db):
    return PnLCalculator(db)


# ---------------------------------------------------------------------------
# Semantics contract
#
# realized_pnl: cash flows from CLOSED and SETTLED rows only
#               + settled_payout for settled rows with positive signed_shares
#
# unrealized_pnl: SUM(mark * net_signed_shares + net_cash_flow_usd)
#                 for 'filled' status rows only, grouped per token_id
#
# total_pnl = realized_pnl + unrealized_pnl
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# realized_pnl
# ---------------------------------------------------------------------------

def test_realized_pnl_zero_when_no_orders(calc):
    assert calc.realized_pnl() == pytest.approx(0.0)


def test_realized_pnl_zero_for_open_fill(tracker, calc):
    """A filled-but-open BUY has no realized PnL yet."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    assert calc.realized_pnl() == pytest.approx(0.0)


def test_realized_pnl_closed_order_contributes_cash_flow(tracker, calc):
    """A closed BUY contributes its (negative) cash flow to realized."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    tracker.mark_closed("ord-1")
    assert calc.realized_pnl() == pytest.approx(-20.0)


def test_realized_pnl_settled_includes_payout(tracker, calc):
    """A settled BUY: cash_flow (-20) + settled_payout (50) = +30."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    tracker.mark_settled("ord-1", settled_payout=50.0)
    assert calc.realized_pnl() == pytest.approx(30.0)


def test_realized_pnl_settled_sell_no_extra_payout(tracker, calc):
    """A settled SELL with negative signed_shares does not receive extra payout."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "SELL", 30.0, 0.60)
    tracker.mark_filled("ord-1", 0.60)
    tracker.mark_settled("ord-1", settled_payout=0.0)
    # cash_flow = +30, settled_payout = 0 (signed_shares = -50, not > 0)
    assert calc.realized_pnl() == pytest.approx(30.0)


def test_realized_pnl_filter_by_condition(tracker, calc):
    # cid-1: BUY closed (-20)
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    tracker.mark_closed("ord-1")
    # cid-2: BUY settled (-10 + 25 = +15)
    tracker.record_order("ord-2", "tok-yes-2", "cid-2", "BUY", 10.0, 0.40)
    tracker.mark_filled("ord-2", 0.40)
    tracker.mark_settled("ord-2", settled_payout=25.0)

    assert calc.realized_pnl(condition_id="cid-1") == pytest.approx(-20.0)
    assert calc.realized_pnl(condition_id="cid-2") == pytest.approx(15.0)


def test_realized_pnl_filter_is_paper(tracker, calc):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40,
                         is_paper=True)
    tracker.mark_filled("ord-1", 0.40)
    tracker.mark_closed("ord-1")
    tracker.record_order("ord-2", "tok-yes", "cid-1", "BUY", 10.0, 0.40,
                         is_paper=False)
    tracker.mark_filled("ord-2", 0.40)
    tracker.mark_closed("ord-2")
    assert calc.realized_pnl(is_paper=True) == pytest.approx(-20.0)
    assert calc.realized_pnl(is_paper=False) == pytest.approx(-10.0)
    assert calc.realized_pnl(is_paper=None) == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# unrealized_pnl
# ---------------------------------------------------------------------------

def test_unrealized_pnl_zero_when_no_positions(calc):
    assert calc.unrealized_pnl({}) == pytest.approx(0.0)


def test_unrealized_pnl_long_at_profit(tracker, calc):
    """50 shares bought at 0.40; mark at 0.60 -> unrealized = 50*0.60 + (-20) = 10."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    pnl = calc.unrealized_pnl({"tok-yes": 0.60})
    assert pnl == pytest.approx(50.0 * 0.60 + (-20.0))   # = 10


def test_unrealized_pnl_token_not_in_mark_prices_excluded(tracker, calc):
    """Tokens without a mark price are excluded from unrealized PnL."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    assert calc.unrealized_pnl({}) == pytest.approx(0.0)


def test_unrealized_pnl_closed_not_included(tracker, calc):
    """Closed orders are NOT in unrealized PnL (they're realized)."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    tracker.mark_closed("ord-1")
    # No open positions -> unrealized = 0
    assert calc.unrealized_pnl({"tok-yes": 0.90}) == pytest.approx(0.0)


def test_unrealized_pnl_filter_is_paper(tracker, calc):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40,
                         is_paper=True)
    tracker.mark_filled("ord-1", 0.40)
    tracker.record_order("ord-2", "tok-yes", "cid-1", "BUY", 20.0, 0.40,
                         is_paper=False)
    tracker.mark_filled("ord-2", 0.40)
    # at mark 0.60: each = 50*0.60 + (-20) = 10
    assert calc.unrealized_pnl({"tok-yes": 0.60}, is_paper=True) == pytest.approx(10.0)
    assert calc.unrealized_pnl({"tok-yes": 0.60}, is_paper=False) == pytest.approx(10.0)
    # None = both: 100 shares, cash=-40; 100*0.60 + (-40) = 20
    assert calc.unrealized_pnl({"tok-yes": 0.60}, is_paper=None) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# total_pnl and pnl_by_event
# ---------------------------------------------------------------------------

def test_total_pnl_equals_realized_plus_unrealized(tracker, calc):
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    realized   = calc.realized_pnl()
    unrealized = calc.unrealized_pnl({"tok-yes": 0.60})
    assert calc.total_pnl({"tok-yes": 0.60}) == pytest.approx(realized + unrealized)


def test_total_pnl_open_buy_at_profit(tracker, calc):
    """Open BUY at 0.40, mark at 0.60 = total PnL +10."""
    tracker.record_order("ord-1", "tok-yes", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    assert calc.total_pnl({"tok-yes": 0.60}) == pytest.approx(10.0)


def test_pnl_by_event_groups_by_condition(tracker, calc):
    tracker.record_order("ord-1", "tok-yes",   "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("ord-1", 0.40)
    tracker.record_order("ord-2", "tok-yes-2", "cid-2", "BUY", 10.0, 0.40)
    tracker.mark_filled("ord-2", 0.40)

    by_event = calc.pnl_by_event({"tok-yes": 0.60, "tok-yes-2": 0.50})
    assert "cid-1" in by_event
    assert "cid-2" in by_event
    # cid-1: open fill: 50*0.60 + (-20) = 10
    assert by_event["cid-1"] == pytest.approx(10.0)
    # cid-2: open fill: 25*0.50 + (-10) = 2.5
    assert by_event["cid-2"] == pytest.approx(2.5)
