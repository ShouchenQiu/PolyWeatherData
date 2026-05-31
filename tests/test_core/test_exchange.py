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
        def place_limit_order(self, **kwargs):
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
    """A response without an order_id must emit an order_failure alert."""
    from types import SimpleNamespace
    from core.alerts import AlertBus, AlertEvent
    from core.exchange.clob import ClobExchange

    class RejectingClient:
        def place_limit_order(self, **kwargs):
            return SimpleNamespace(order_id=None, error_msg="insufficient funds")

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
# ClobExchange -- V2F02: SecureClient interface
# -----------------------------------------------------------------------

def test_clob_exchange_place_order_calls_place_limit_order():
    """place_order() must delegate to SecureClient.place_limit_order()."""
    from types import SimpleNamespace
    from core.exchange.clob import ClobExchange

    placed: list[dict] = []

    class FakeSecureClient:
        def place_limit_order(self, *, token_id, price, size, side):
            placed.append({"token_id": token_id, "price": price,
                           "size": size, "side": side})
            return SimpleNamespace(order_id="ord-001", error_msg=None)

    exch = ClobExchange(clob_client=FakeSecureClient())
    oid  = exch.place_order("tok1", "BUY", 50.0, 0.55)

    assert oid == "ord-001"
    assert len(placed) == 1
    assert placed[0]["token_id"] == "tok1"
    assert placed[0]["price"] == pytest.approx(0.55)
    assert placed[0]["size"] == pytest.approx(50.0)
    assert placed[0]["side"] == "BUY"


def test_clob_exchange_cancel_order_calls_cancel_orders():
    """cancel_order() must delegate to SecureClient.cancel_orders()."""
    from core.exchange.clob import ClobExchange

    cancelled: list[str] = []

    class FakeSecureClient:
        def cancel_orders(self, *, order_ids):
            cancelled.extend(order_ids)
            return True

    exch   = ClobExchange(clob_client=FakeSecureClient())
    result = exch.cancel_order("ord-999")

    assert result is True
    assert "ord-999" in cancelled


def test_clob_exchange_module_has_no_py_clob_client_import():
    """No py_clob_client import must remain in core/exchange/clob.py."""
    import inspect
    import core.exchange.clob as mod
    src = inspect.getsource(mod)
    assert "py_clob_client" not in src
