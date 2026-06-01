"""Tests for core/exchange/ -- PaperExchange, ClobExchange, ExitSettler."""
from __future__ import annotations

import asyncio

import pytest

from core.exchange.paper import PaperExchange
from core.exchange.settlement import ExitSettler


def _l2_book(ask: float, ask_size: float = 1000.0,
             bid: float | None = None, bid_size: float = 1000.0) -> dict:
    if bid is None:
        bid = max(0.01, ask - 0.02)
    return {
        "bids": [{"price": bid, "size": bid_size}],
        "asks": [{"price": ask, "size": ask_size}],
    }


class _FakeFetcher:
    def __init__(self, book: dict) -> None:
        self._book = book

    def get_book(self, token_id: str) -> dict:
        return self._book


# -----------------------------------------------------------------------
# PaperExchange -- no fetcher
# -----------------------------------------------------------------------

def test_paper_no_fetcher_returns_empty_string():
    """Without a book_fetcher, place_order always returns empty string."""
    exch = PaperExchange(book_fetcher=None)
    oid  = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert oid == ""


def test_paper_get_positions_always_empty():
    """PaperExchange.get_positions() returns [] (state is in PositionTracker)."""
    exch = PaperExchange()
    exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert exch.get_positions() == []


def test_paper_cancel_always_returns_false():
    """cancel_order returns False; no in-flight tracking."""
    exch = PaperExchange()
    assert exch.cancel_order("any-id") is False


# -----------------------------------------------------------------------
# PaperExchange -- BUY fills at asks VWAP
# -----------------------------------------------------------------------

def test_paper_buy_returns_order_id_on_sufficient_depth():
    """BUY with sufficient ask depth returns a non-empty order_id."""
    fetcher = _FakeFetcher(_l2_book(ask=0.50, ask_size=1000.0))
    exch    = PaperExchange(book_fetcher=fetcher)
    oid     = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert isinstance(oid, str) and len(oid) > 0


def test_paper_buy_returns_empty_string_on_insufficient_depth():
    """BUY size_usd exceeds available ask depth -> empty string."""
    # Only 5 shares at ask=0.50 -> $2.50 available; requesting $20
    fetcher = _FakeFetcher(_l2_book(ask=0.50, ask_size=5.0))
    exch    = PaperExchange(book_fetcher=fetcher)
    oid     = exch.place_order("tok1", "BUY", 20.0, 0.50)
    assert oid == ""


# -----------------------------------------------------------------------
# PaperExchange -- SELL fills at bids VWAP
# -----------------------------------------------------------------------

def test_paper_sell_returns_order_id_on_sufficient_depth():
    """SELL with sufficient bid depth returns a non-empty order_id."""
    fetcher = _FakeFetcher(_l2_book(ask=0.52, ask_size=1000.0,
                                    bid=0.48, bid_size=1000.0))
    exch    = PaperExchange(book_fetcher=fetcher)
    oid     = exch.place_order("tok1", "SELL", 20.0, 0.48)
    assert isinstance(oid, str) and len(oid) > 0


def test_paper_sell_returns_empty_string_on_insufficient_depth():
    """SELL size_usd exceeds available bid depth -> empty string."""
    # Only 5 shares at bid=0.48 -> $2.40 available; requesting $20
    fetcher = _FakeFetcher(_l2_book(ask=0.52, ask_size=1000.0,
                                    bid=0.48, bid_size=5.0))
    exch    = PaperExchange(book_fetcher=fetcher)
    oid     = exch.place_order("tok1", "SELL", 20.0, 0.48)
    assert oid == ""


# -----------------------------------------------------------------------
# PaperExchange -- multiple orders return distinct ids
# -----------------------------------------------------------------------

def test_paper_two_orders_return_distinct_ids():
    """Each successful place_order returns a unique order_id."""
    fetcher = _FakeFetcher(_l2_book(ask=0.50))
    exch    = PaperExchange(book_fetcher=fetcher)
    oid1    = exch.place_order("tok1", "BUY", 10.0, 0.50)
    oid2    = exch.place_order("tok1", "BUY", 10.0, 0.50)
    assert oid1 and oid2
    assert oid1 != oid2


# -----------------------------------------------------------------------
# ExitSettler
# -----------------------------------------------------------------------

class _FakeExchange:
    """Minimal exchange stub for ExitSettler tests."""

    def __init__(self, positions: list[dict]) -> None:
        self._positions = list(positions)
        self.sell_calls: list[tuple] = []

    def get_positions(self) -> list[dict]:
        return [p for p in self._positions if p.get("size_usd", 0) > 0]

    def place_order(self, token_id, side, size_usd, price) -> str:
        self.sell_calls.append((token_id, side, size_usd, price))
        # Simulate closing the position
        for p in self._positions:
            if p["token_id"] == token_id:
                p["size_usd"] = 0.0
        return "oid-settler"

    def cancel_order(self, order_id: str) -> bool:
        return False


def test_settler_fallback_sells_open_positions():
    exch = _FakeExchange([{"token_id": "tok1", "size_usd": 25.0}])
    settler = ExitSettler(relayer_client=None)
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    assert len(exch.sell_calls) == 1
    assert exch.sell_calls[0][0] == "tok1"
    assert exch.sell_calls[0][1] == "SELL"


def test_settler_only_closes_relevant_tokens():
    exch = _FakeExchange([
        {"token_id": "tok1", "size_usd": 25.0},
        {"token_id": "tok2", "size_usd": 25.0},
    ])
    settler = ExitSettler(relayer_client=None)
    settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    sold_tokens = [c[0] for c in exch.sell_calls]
    assert "tok1" in sold_tokens
    assert "tok2" not in sold_tokens


def test_settler_relayer_success_returns_relayer():
    class FakeRelayer:
        def settle(self, condition_id):
            return True

    exch    = _FakeExchange([])
    settler = ExitSettler(relayer_client=FakeRelayer())
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": []},
        exchange = exch,
    )
    assert result == "relayer"


def test_settler_relayer_failure_falls_back():
    class BrokenRelayer:
        def settle(self, condition_id):
            raise RuntimeError("network error")

    exch = _FakeExchange([{"token_id": "tok1", "size_usd": 10.0}])
    settler = ExitSettler(relayer_client=BrokenRelayer())
    result  = settler.settle(
        event    = {"condition_id": "c1", "token_ids": ["tok1"]},
        exchange = exch,
    )
    assert result == "fallback"
    assert len(exch.sell_calls) == 1


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
    """Successful get_book() with sufficient L2 depth must NOT emit an alert."""
    from core.alerts import AlertBus, AlertEvent

    emitted: list[AlertEvent] = []

    async def _handler(event: AlertEvent) -> None:
        emitted.append(event)

    bus  = AlertBus()
    bus.subscribe(_handler)
    exch = PaperExchange(book_fetcher=_FakeFetcher(_l2_book(ask=0.50)), alert_bus=bus)

    async def _run():
        oid = exch.place_order("tok1", "BUY", 20.0, 0.50)
        assert oid != ""   # depth was sufficient
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
