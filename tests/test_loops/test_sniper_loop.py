"""Tests for SniperLoop._handle_observation order-recording path."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from db.schema import DbClient
from core.positions import PositionTracker
from core.risk.fixed_fraction import FixedFraction
from loops.sniper_loop import SniperLoop, _parse_metar_temp_f


# ---------------------------------------------------------------------------
# Unit tests: METAR temperature parser
# ---------------------------------------------------------------------------

def test_parse_t_group_positive():
    metar = "KLGA 251951Z T02000139"
    assert abs(_parse_metar_temp_f(metar) - (20.0 * 9 / 5 + 32)) < 0.1


def test_parse_t_group_negative():
    metar = "KLGA T11050139"
    assert abs(_parse_metar_temp_f(metar) - (-10.5 * 9 / 5 + 32)) < 0.1


def test_parse_slash_positive():
    assert abs(_parse_metar_temp_f("KLGA 20/14") - (20 * 9 / 5 + 32)) < 0.1


def test_parse_slash_minus():
    assert abs(_parse_metar_temp_f("KLGA M03/M07") - (-3 * 9 / 5 + 32)) < 0.1


def test_parse_returns_none_when_no_temp():
    assert _parse_metar_temp_f("KLGA 251951Z 09007KT 10SM CAVOK") is None


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
    # YES token for 60-70F bucket
    client.upsert_market(
        token_id="tok-60-70-yes",
        condition_id="cid-1",
        market_slug="high-60-70f",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        outcome="YES",
    )
    # NO token for 60-70F bucket
    client.upsert_market(
        token_id="tok-60-70-no",
        condition_id="cid-1",
        market_slug="high-60-70f-no",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        outcome="NO",
    )
    # YES token for 70-80F bucket
    client.upsert_market(
        token_id="tok-70-80-yes",
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


def _make_loop(db, tracker):
    mgr      = MagicMock()
    mgr.queue = MagicMock()
    book_fetcher = MagicMock()
    book_fetcher.get_book.return_value = {"best_ask": 0.80}
    exchange = MagicMock()
    exchange.place_order.return_value = "ord-sniper-1"
    risk = FixedFraction(position_usd=20.0, max_total_usd=1000.0)
    return SniperLoop(
        sniper_mgr=mgr,
        orderbook=book_fetcher,
        risk=risk,
        exchange=exchange,
        positions=tracker,
        db=db,
    )


# ---------------------------------------------------------------------------
# _handle_observation: order recording
# ---------------------------------------------------------------------------

def test_handle_observation_calls_record_order(db, tracker):
    """A matching METAR should call record_order on PositionTracker."""
    loop = _make_loop(db, tracker)
    # 65F sits in the 60-70F bucket
    metar = "KLGA 251951Z 09007KT T01833 18C"  # fallback: slash field
    metar = "KLGA 251951Z 20/14"  # 20C = 68F -> in 60-70F

    asyncio.run(loop._handle_observation(metar, "NOAA", None))

    # At least one pending order recorded
    rows = db.query("SELECT order_id, status FROM orders WHERE status='pending'")
    assert len(rows) >= 1


def test_handle_observation_no_temp_skips_orders(db, tracker):
    """METAR with no temperature field should not produce any orders."""
    loop  = _make_loop(db, tracker)
    metar = "KLGA 251951Z 09007KT 10SM CAVOK"

    asyncio.run(loop._handle_observation(metar, "NOAA", None))

    rows = db.query("SELECT order_id FROM orders")
    assert len(rows) == 0


def test_handle_observation_uses_record_order_not_open(db, tracker):
    """SniperLoop must not hold a reference to the removed open() shim."""
    assert not hasattr(PositionTracker, "open"), (
        "PositionTracker.open() shim must not exist; SniperLoop must use record_order()"
    )
