"""Tests for SniperLoop._handle_observation order-recording path."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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
        event_end_date=datetime.now(timezone.utc).date().isoformat(),
        station_id="KLGA",
        status="active",
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


def _make_loop(db, tracker, settings=None, station_id="KLGA"):
    mgr      = MagicMock()
    mgr.queue = MagicMock()
    mgr.start_all = AsyncMock()
    mgr.stop_all = AsyncMock()
    book_fetcher = MagicMock()
    book_fetcher.get_book.return_value = {
        "bids": [{"price": 0.78, "size": 1000.0}],
        "asks": [{"price": 0.80, "size": 1000.0}],
        "best_bid": 0.78,
        "best_ask": 0.80,
        "mid_price": 0.79,
    }
    exchange = MagicMock()
    exchange.place_order.return_value = "ord-sniper-1"
    risk = FixedFraction(position_usd=20.0, max_total_usd=1000.0)
    if settings is None:
        settings = MagicMock()
        settings.trading_mode   = "paper"
        settings.quote_size_usd = 20.0
    return SniperLoop(
        sniper_mgr=mgr,
        orderbook=book_fetcher,
        risk=risk,
        exchange=exchange,
        positions=tracker,
        db=db,
        settings=settings,
        station_id=station_id,
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

    # At least one filled order recorded (paper mode fills immediately)
    rows = db.query("SELECT order_id, status FROM orders WHERE status='filled'")
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


def test_run_starts_manager_and_consumer(db, tracker):
    """run() owns both the source polling task and the queue consumer."""
    loop = _make_loop(db, tracker)
    loop._consume_queue = AsyncMock(side_effect=asyncio.CancelledError)

    try:
        asyncio.run(loop.run())
    except asyncio.CancelledError:
        pass

    loop._consume_queue.assert_awaited_once()
    loop.sniper_mgr.start_all.assert_awaited_once()
    loop.sniper_mgr.stop_all.assert_awaited_once()


def test_consume_queue_handles_observation(db, tracker):
    """_consume_queue() reads queue items and forwards them to _handle_observation()."""
    loop = _make_loop(db, tracker)
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait(("KLGA 251951Z 20/14", "NOAA", None))
    loop.sniper_mgr.queue = q
    loop._handle_observation = AsyncMock(side_effect=asyncio.CancelledError)

    try:
        asyncio.run(loop._consume_queue())
    except asyncio.CancelledError:
        pass

    loop._handle_observation.assert_awaited_once_with(
        "KLGA 251951Z 20/14", "NOAA", None
    )


def test_handle_observation_skips_invalid_bucket(db, tracker):
    """Invalid bucket bounds should not generate orders."""
    db.upsert_market(
        token_id="tok-invalid",
        condition_id="cid-1",
        market_slug="bad-bucket",
        bucket_lo_f=90.0,
        bucket_hi_f=80.0,
        outcome="YES",
    )
    db.execute("DELETE FROM markets WHERE token_id != ?", ("tok-invalid",))
    loop = _make_loop(db, tracker)

    asyncio.run(loop._handle_observation("KLGA 251951Z 20/14", "NOAA", None))

    rows = db.query("SELECT order_id FROM orders")
    assert len(rows) == 0

# ---------------------------------------------------------------------------
# V2V11 -- SniperLoop alignment with V2 contracts
# ---------------------------------------------------------------------------

class TestV2V11SniperQuotePricing:
    """SniperLoop prices via compute_quote instead of best_ask/mid fallback."""

    def test_multi_level_vwap_not_best_ask(self, db, tracker):
        """Two ask levels ensure VWAP differs from best_ask.

        $20 BUY: first 5 shares @ 0.80 = $4, remaining $16 @ 0.90
        VWAP = 20 / (5 + 17.778) = 20 / 22.778 = 0.8780...
        This is not equal to best_ask=0.80.
        """
        loop = _make_loop(db, tracker)
        loop.orderbook.get_book.return_value = {
            "bids": [{"price": 0.78, "size": 1000.0}],
            "asks": [
                {"price": 0.80, "size": 5.0},
                {"price": 0.90, "size": 1000.0},
            ],
        }
        metar = "KLGA 251951Z 20/14"  # 68F -> 60-70F bucket

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query("SELECT price FROM orders WHERE status='filled'")
        assert rows
        # VWAP should be between 0.80 and 0.90, not equal to best_ask
        price = rows[0]["price"]
        assert 0.85 < price < 0.90
        assert price != pytest.approx(0.80)


class TestV2V11SniperOrderRecording:
    """SniperLoop records orders with is_paper and executable_price."""

    def test_record_order_passes_is_paper_and_executable_price(self, db, tracker):
        """Verifies record_order receives the V2 keyword arguments."""
        loop = _make_loop(db, tracker)
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query(
            "SELECT is_paper, executable_price FROM orders WHERE status='filled'"
        )
        assert rows
        assert rows[0]["is_paper"] == 1   # paper mode
        assert rows[0]["executable_price"] == pytest.approx(0.80)

    def test_paper_mode_calls_mark_filled(self, db, tracker):
        """Paper mode marks orders filled immediately after recording."""
        loop = _make_loop(db, tracker)
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query(
            "SELECT status, fill_price FROM orders WHERE status='filled'"
        )
        assert rows
        assert rows[0]["status"] == "filled"
        assert rows[0]["fill_price"] == pytest.approx(0.80)


class TestV2V11SniperEventQuery:
    """SniperLoop always uses get_tradeable_events."""

    def test_get_tradeable_events_called_with_today_date(self, db, tracker):
        """SniperLoop calls get_tradeable_events; no more get_active_events fallback."""
        db_mock = MagicMock(wraps=db)
        db_mock.get_tradeable_events.return_value = db.get_tradeable_events(
            datetime.now(timezone.utc).date().isoformat()
        )
        loop = _make_loop(db_mock, tracker)
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        db_mock.get_tradeable_events.assert_called_once()
        # get_active_events must not be called (no fallback)
        get_active = getattr(db_mock, "get_active_events", None)
        if get_active is not None:
            get_active.assert_not_called()

    def test_missing_get_tradeable_events_raises(self, db, tracker):
        """A DbClient without get_tradeable_errors raises AttributeError (no
        silent fallback to get_active_events)."""
        db_mock = MagicMock()
        del db_mock.get_tradeable_events  # remove the method
        # Must also provide the other methods used by _handle_observation
        db_mock.get_markets_for_event.return_value = []
        loop = _make_loop(db_mock, tracker)
        metar = "KLGA 251951Z 20/14"

        with pytest.raises(AttributeError):
            asyncio.run(loop._handle_observation(metar, "NOAA", None))

    def test_compute_size_receives_positions(self, db, tracker):
        """compute_size() is called with positions=self.positions."""
        risk_mock = MagicMock()
        risk_mock.approve_signal.return_value = True
        risk_mock.compute_size.return_value = 20.0
        loop = _make_loop(db, tracker)
        loop.risk = risk_mock
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        # compute_size must have been called with positions= keyword
        _, kwargs = risk_mock.compute_size.call_args
        assert "positions" in kwargs
        assert kwargs["positions"] is tracker


class TestV2V11SniperDepthGate:
    """SniperLoop skips when L2 depth is insufficient."""

    def test_trial_quote_incomplete_fill_skips(self, db, tracker):
        """When trial quote at quote_size_usd has insufficient depth, skip."""
        loop = _make_loop(db, tracker)
        loop.orderbook.get_book.return_value = {
            "bids": [{"price": 0.78, "size": 0.1}],
            "asks": [{"price": 0.80, "size": 0.1}],
        }
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query("SELECT order_id FROM orders")
        assert len(rows) == 0

    def test_requote_incomplete_fill_skips(self, db, tracker):
        """When re-quote at actual size has insufficient depth, skip.

        Trial passes at quote_size_usd=20 but actual risk-computed size
        is larger and fails the re-quote depth check.
        """
        risk_mock = MagicMock()
        risk_mock.approve_signal.return_value = True
        risk_mock.compute_size.return_value = 500.0  # far exceeds depth
        loop = _make_loop(db, tracker)
        loop.risk = risk_mock
        # 100 shares @ 0.80 = $80 max fillable, far below 500
        loop.orderbook.get_book.return_value = {
            "bids": [{"price": 0.78, "size": 100.0}],
            "asks": [{"price": 0.80, "size": 100.0}],
        }
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query("SELECT order_id FROM orders")
        assert len(rows) == 0

    def test_compute_quote_exception_skips(self, db, tracker):
        """When compute_quote raises, skip the token (no fallback)."""
        loop = _make_loop(db, tracker)
        loop.orderbook.get_book.side_effect = RuntimeError("network down")
        metar = "KLGA 251951Z 20/14"

        asyncio.run(loop._handle_observation(metar, "NOAA", None))

        rows = db.query("SELECT order_id FROM orders")
        assert len(rows) == 0

def test_handle_observation_orderbook_none_skips_without_order(db, tracker):
    """Missing orderbook must skip entirely -- no 0.95 fallback."""
    loop = _make_loop(db, tracker)
    loop.orderbook = None

    asyncio.run(loop._handle_observation("KLGA 251951Z 20/14", "NOAA", None))

    rows = db.query("SELECT order_id FROM orders")
    assert len(rows) == 0


def test_handle_observation_empty_order_id_not_recorded(db, tracker):
    """Empty exchange order IDs should not be recorded."""
    loop = _make_loop(db, tracker)
    loop.exchange.place_order.return_value = ""

    asyncio.run(loop._handle_observation("KLGA 251951Z 20/14", "NOAA", None))

    rows = db.query("SELECT order_id FROM orders")
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# V2V11 fix -- SniperLoop filters events by station_id
# ---------------------------------------------------------------------------

def test_handle_observation_filters_events_by_station(db, tracker):
    """Only events matching SniperLoop's station_id are traded."""
    # Add an event for a DIFFERENT station that should be ignored
    db.upsert_station("KJFK", 40.64, -73.78, "America/New_York")
    db.upsert_event(
        condition_id="cid-jfk",
        question="High temp JFK",
        event_slug="high-temp-jfk",
        event_end_date=datetime.now(timezone.utc).date().isoformat(),
        station_id="KJFK",
        status="active",
    )
    db.upsert_market(
        token_id="tok-jfk-yes",
        condition_id="cid-jfk",
        market_slug="high-60-70f-jfk",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        outcome="YES",
    )
    db.upsert_market(
        token_id="tok-jfk-no",
        condition_id="cid-jfk",
        market_slug="high-60-70f-no-jfk",
        bucket_lo_f=60.0,
        bucket_hi_f=70.0,
        outcome="NO",
    )
    loop = _make_loop(db, tracker, station_id="KLGA")
    metar = "KLGA 251951Z 20/14"  # 68F -> 60-70F bucket

    asyncio.run(loop._handle_observation(metar, "NOAA", None))

    # Only orders for KLGA event (cid-1), not KJFK (cid-jfk)
    rows = db.query(
        "SELECT condition_id FROM orders WHERE status='filled'"
    )
    condition_ids = {r["condition_id"] for r in rows}
    assert "cid-1" in condition_ids
    assert "cid-jfk" not in condition_ids
