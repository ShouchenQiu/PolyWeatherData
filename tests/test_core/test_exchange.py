"""Tests for core/exchange/ -- PaperExchange, ClobExchange, ExitSettler."""
from __future__ import annotations

import asyncio

import pytest

from core.exchange.paper import PaperExchange
from core.exchange.settlement import ExitSettler


# -----------------------------------------------------------------------
# PaperExchange.place_order
# -----------------------------------------------------------------------

def test_paper_place_order_returns_string():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert isinstance(oid, str)
    assert len(oid) > 0


def test_paper_place_order_status_filled():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    order = exch.get_order(oid)
    assert order is not None
    assert order["status"] == "filled"


def test_paper_place_order_uses_submitted_price_when_no_fetcher():
    exch = PaperExchange(book_fetcher=None)
    exch.place_order("tok1", "BUY", 20.0, 0.55)
    pos = exch.get_positions()
    assert len(pos) == 1
    assert abs(pos[0]["avg_price"] - 0.55) < 1e-9


def test_paper_place_order_uses_mid_price_from_fetcher():
    class FakeFetcher:
        def get_book(self, token_id):
            return {"mid_price": 0.48}

    exch = PaperExchange(book_fetcher=FakeFetcher())
    exch.place_order("tok1", "BUY", 20.0, 0.55)
    pos = exch.get_positions()
    assert abs(pos[0]["avg_price"] - 0.48) < 1e-9


def test_paper_get_positions_after_buy():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 20.0, 0.50)
    positions = exch.get_positions()
    assert len(positions) == 1
    assert positions[0]["token_id"] == "tok1"
    assert positions[0]["size_usd"] == 20.0


def test_paper_sell_reduces_position():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY",  30.0, 0.50)
    exch.place_order("tok1", "SELL", 10.0, 0.55)
    pos = exch.get_positions()
    assert abs(pos[0]["size_usd"] - 20.0) < 1e-9


def test_paper_sell_closes_full_position():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY",  20.0, 0.50)
    exch.place_order("tok1", "SELL", 20.0, 0.55)
    pos = exch.get_positions()
    assert pos == []


def test_paper_cancel_order_pending_returns_true():
    exch = PaperExchange()
    # We must trick the order into pending status
    exch._orders["test-id"] = {"status": "pending", "token_id": "tok1",
                                "size_usd": 10.0, "price": 0.5}
    assert exch.cancel_order("test-id") is True
    assert exch._orders["test-id"]["status"] == "cancelled"


def test_paper_cancel_filled_order_returns_false():
    exch = PaperExchange()
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert exch.cancel_order(oid) is False


def test_paper_multiple_tokens_independent():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 10.0, 0.40)
    exch.place_order("tok2", "BUY", 20.0, 0.60)
    pos = {p["token_id"]: p for p in exch.get_positions()}
    assert abs(pos["tok1"]["size_usd"] - 10.0) < 1e-9
    assert abs(pos["tok2"]["size_usd"] - 20.0) < 1e-9


# -----------------------------------------------------------------------
# ExitSettler
# -----------------------------------------------------------------------

def test_settler_fallback_sells_open_positions():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 25.0, 0.50)

    settler = ExitSettler(relayer_client=None)
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    # After settlement the position should be closed
    pos = exch.get_positions()
    assert pos == []


def test_settler_only_closes_relevant_tokens():
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 25.0, 0.50)
    exch.place_order("tok2", "BUY", 25.0, 0.50)

    settler = ExitSettler(relayer_client=None)
    settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    # tok2 should still be open
    pos = {p["token_id"]: p for p in exch.get_positions()}
    assert "tok2" in pos
    assert "tok1" not in pos


def test_settler_relayer_success_returns_relayer():
    class FakeRelayer:
        def settle(self, condition_id):
            return True

    exch    = PaperExchange()
    settler = ExitSettler(relayer_client=FakeRelayer())
    result  = settler.settle(
        event = {"condition_id": "c1", "token_ids": []},
        exchange = exch,
    )
    assert result == "relayer"


def test_settler_relayer_failure_falls_back():
    class BrokenRelayer:
        def settle(self, condition_id):
            raise RuntimeError("network error")

    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 10.0, 0.50)
    settler = ExitSettler(relayer_client=BrokenRelayer())
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    assert exch.get_positions() == []


# -----------------------------------------------------------------------
# OrderbookFetcher
# -----------------------------------------------------------------------

def test_orderbook_fetcher_get_book_delegates_to_client():
    from data.clob.orderbook import OrderbookFetcher

    class FakeSummary:
        bids = [{"price": "0.52", "size": "100"}]
        asks = [{"price": "0.58", "size": "100"}]

    class FakeClient:
        def get_order_book(self, token_id):
            return FakeSummary()

    fetcher = OrderbookFetcher(FakeClient())
    book    = fetcher.get_book("tok1")
    assert book["mid_price"] == pytest.approx(0.55)


def test_orderbook_fetcher_raises_on_client_error():
    from data.clob.orderbook import OrderbookFetcher

    class BrokenClient:
        def get_order_book(self, token_id):
            raise RuntimeError("CLOB down")

    fetcher = OrderbookFetcher(BrokenClient())
    with pytest.raises(RuntimeError):
        fetcher.get_book("tok1")


# -----------------------------------------------------------------------
# PaperExchange -- order_failure alert on get_book() failure
# -----------------------------------------------------------------------

def test_paper_exchange_emits_book_fetch_failure_on_fetcher_error():
    """When get_book() raises, PaperExchange emits a book_fetch_failure alert."""
    from core.alerts import AlertBus, AlertEvent

    class BrokenFetcher:
        def get_book(self, token_id):
            raise RuntimeError("CLOB timeout")

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = PaperExchange(book_fetcher=BrokenFetcher(), alert_bus=bus)

    async def _run():
        exch.place_order("tok1", "BUY", 20.0, 0.50)
        # give the task a chance to run
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert len(emitted) == 1
    assert emitted[0].alert_type == "book_fetch_failure"


def test_paper_exchange_no_alert_on_success():
    """Successful get_book() must NOT emit an alert."""
    from core.alerts import AlertBus, AlertEvent

    class GoodFetcher:
        def get_book(self, token_id):
            return {"mid_price": 0.50}

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = PaperExchange(book_fetcher=GoodFetcher(), alert_bus=bus)

    async def _run():
        exch.place_order("tok1", "BUY", 20.0, 0.50)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert emitted == []


def test_paper_exchange_no_alert_when_no_fetcher():
    """No fetcher -> no alert bus invoked, falls back to submitted price."""
    from core.alerts import AlertBus, AlertEvent

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = PaperExchange(book_fetcher=None, alert_bus=bus)

    async def _run():
        exch.place_order("tok1", "BUY", 20.0, 0.50)
        await asyncio.sleep(0)

    asyncio.run(_run())
    assert emitted == []


# -----------------------------------------------------------------------
# ClobExchange -- order_failure alert
# -----------------------------------------------------------------------

def test_clob_exchange_emits_order_failure_on_place_exception():
    """place_order() raising an exception must emit an order_failure alert."""
    from core.alerts import AlertBus, AlertEvent
    from core.exchange.clob import ClobExchange

    class BrokenClient:
        def create_order(self, order_args):
            raise RuntimeError("auth failure")

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = ClobExchange(clob_client=BrokenClient(), alert_bus=bus)

    async def _run():
        with pytest.raises(RuntimeError):
            exch.place_order("tok1", "BUY", 20.0, 0.50)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert len(emitted) == 1
    assert emitted[0].alert_type == "order_failure"


def test_clob_exchange_emits_order_failure_on_rejected_response():
    """A response without an orderID must emit an order_failure alert."""
    from core.alerts import AlertBus, AlertEvent
    from core.exchange.clob import ClobExchange

    class RejectingClient:
        def create_order(self, order_args):
            return object()
        def post_order(self, signed):
            return {"errorMsg": "insufficient funds", "orderID": ""}

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = ClobExchange(clob_client=RejectingClient(), alert_bus=bus)

    async def _run():
        exch.place_order("tok1", "BUY", 20.0, 0.50)
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert len(emitted) == 1
    assert emitted[0].alert_type == "order_failure"


# -----------------------------------------------------------------------
# bot.py -- auth wiring (V07)
# -----------------------------------------------------------------------

def test_bot_paper_mode_no_api_creds(tmp_path, monkeypatch):
    """Paper mode must not pass ApiCreds to ClobClient constructor."""
    from unittest.mock import MagicMock, patch
    from config.settings import Settings

    settings = Settings(
        data_store_dir=tmp_path / "data",
        trading_mode="paper",
    )

    constructed_kwargs: list[dict] = []

    class FakeClobClient:
        def __init__(self, *args, **kwargs):
            constructed_kwargs.append(kwargs)

    with patch("core.bot.ClobClient", FakeClobClient):
        from core.bot import BotManager
        bot = BotManager(settings, db_path=str(tmp_path / "bot.db"))

    # Paper mode: should not have passed creds= kwarg
    assert all("creds" not in kw for kw in constructed_kwargs)


def test_bot_live_mode_passes_api_creds(tmp_path, monkeypatch):
    """Live mode must pass creds= to ClobClient constructor."""
    from unittest.mock import patch
    from config.settings import Settings
    import core.bot

    monkeypatch.setenv("POLYMARKET_API_KEY", "test-key")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "test-secret")
    monkeypatch.setenv("POLYMARKET_PASSPHRASE", "test-pass")

    settings = Settings(
        data_store_dir=tmp_path / "data",
        trading_mode="live",
    )

    constructed_kwargs: list[dict] = []

    class FakeClobClient:
        def __init__(self, *args, **kwargs):
            constructed_kwargs.append(kwargs)

    class FakeApiCreds:
        def __init__(self, **kwargs):
            pass

    with patch("core.bot.ClobClient", FakeClobClient), \
         patch("core.bot.ApiCreds", FakeApiCreds):
        bot = core.bot.BotManager(settings, db_path=str(tmp_path / "bot.db"))

    live_calls = [kw for kw in constructed_kwargs if "creds" in kw]
    assert len(live_calls) >= 1
