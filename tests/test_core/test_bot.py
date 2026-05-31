"""BotManager construction smoke test."""
from __future__ import annotations

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.alerts import AlertBus
from core.bot import BotManager
from core.probability_store import ProbabilityStore
from config.settings import Settings
from loops.forecast_loop import ForecastLoop
from loops.trading_loop import TradingLoop


def test_bot_manager_constructs(tmp_path):
    """BotManager._build() runs without error and wires the ADR-0004 objects."""
    settings = Settings(data_store_dir=tmp_path / "data")
    bot = BotManager(settings, db_path=str(tmp_path / "bot.db"))

    assert isinstance(bot.prob_store, ProbabilityStore)
    assert isinstance(bot.alert_bus, AlertBus)
    assert isinstance(bot.forecast_loop, ForecastLoop)
    assert isinstance(bot.trading_loop, TradingLoop)


def test_bot_manager_forecast_loop_shares_prob_store(tmp_path):
    """ForecastLoop and TradingLoop share the same ProbabilityStore instance."""
    settings = Settings(data_store_dir=tmp_path / "data")
    bot = BotManager(settings, db_path=str(tmp_path / "bot.db"))

    assert bot.forecast_loop.prob_store is bot.prob_store
    assert bot.trading_loop._prob_store is bot.prob_store


def test_bot_wires_discovery_loop_with_db_lifecycle_interval_and_alert_bus(tmp_path):
    """V2V01: DiscoveryLoop receives db, lifecycle, interval, and EventDiscovery gets alert_bus."""
    from events.discovery import EventDiscovery
    from events.lifecycle import EventLifecycle
    from loops.discovery_loop import DiscoveryLoop

    settings = Settings(
        data_store_dir=tmp_path / "data",
        discovery_loop_interval_s=120,
    )
    bot = BotManager(settings, db_path=str(tmp_path / "bot.db"))

    # EventDiscovery must receive the shared alert_bus
    assert bot.discovery._alert_bus is bot.alert_bus

    # DiscoveryLoop must receive db, lifecycle, and the configured interval
    loop = bot.discovery_loop
    assert isinstance(loop, DiscoveryLoop)
    assert loop._db is bot.db
    assert isinstance(loop._lifecycle, EventLifecycle)
    assert loop._interval_seconds == 120.0


class _RunStub:
    def __init__(self, name: str, calls: list[str], cancel: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.cancel = cancel

    async def run(self) -> None:
        self.calls.append(self.name)
        await asyncio.sleep(0)
        if self.cancel:
            raise asyncio.CancelledError
        await asyncio.Event().wait()


def _make_run_only_bot(enable_sniper: bool, calls: list[str]) -> BotManager:
    bot = BotManager.__new__(BotManager)
    bot.settings = SimpleNamespace(enable_sniper=enable_sniper)
    bot.forecast_loop = _RunStub("forecast", calls, cancel=True)
    bot.trading_loop = _RunStub("trading", calls)
    bot.discovery_loop = _RunStub("discovery", calls)
    bot.settlement_loop = _RunStub("settlement", calls)
    bot.sniper_loop = _RunStub("sniper", calls)
    bot.snipers = SimpleNamespace(start_all=AsyncMock())
    return bot


def test_bot_manager_run_skips_sniper_when_disabled():
    """enable_sniper=False does not start sniper source polling or consumer."""
    calls: list[str] = []
    bot = _make_run_only_bot(False, calls)

    try:
        asyncio.run(bot.run())
    except asyncio.CancelledError:
        pass

    assert "sniper" not in calls
    bot.snipers.start_all.assert_not_awaited()


def test_bot_manager_run_starts_single_sniper_entry_when_enabled():
    """enable_sniper=True starts SniperLoop only; manager is internal to it."""
    calls: list[str] = []
    bot = _make_run_only_bot(True, calls)

    try:
        asyncio.run(bot.run())
    except asyncio.CancelledError:
        pass

    assert "sniper" in calls
    bot.snipers.start_all.assert_not_awaited()


# ---------------------------------------------------------------------------
# V2F02 AC: paper mode never constructs SecureClient; live mode uses it
# ---------------------------------------------------------------------------

def test_bot_paper_mode_uses_paper_exchange(tmp_path):
    """Paper mode wires PaperExchange and does not require SDK credentials."""
    from core.exchange.paper import PaperExchange
    settings = Settings(data_store_dir=tmp_path / "data")
    bot = BotManager(settings, db_path=str(tmp_path / "bot.db"))
    assert isinstance(bot.exchange, PaperExchange)


def test_bot_live_mode_raises_without_sdk(tmp_path, monkeypatch):
    """Live mode raises ImportError when polymarket-client is not installed."""
    import core.bot as bot_mod
    monkeypatch.setattr(bot_mod, "SecureClient", None)
    settings = Settings(data_store_dir=tmp_path / "data")
    settings.trading_mode = "live"
    with pytest.raises(ImportError, match="polymarket-client"):
        BotManager(settings, db_path=str(tmp_path / "bot.db"))


def test_bot_live_mode_calls_secure_client_create(tmp_path, monkeypatch):
    """Live mode calls SecureClient.create(private_key=..., wallet=...)."""
    import core.bot as bot_mod
    from types import SimpleNamespace

    created_with: list[dict] = []

    class FakeSecure:
        @classmethod
        def create(cls, *, private_key, wallet):
            created_with.append({"private_key": private_key, "wallet": wallet})
            return cls()
        def place_limit_order(self, **kwargs):
            return SimpleNamespace(order_id="", error_msg=None)
        def cancel_orders(self, **kwargs):
            return True
        def get_positions(self):
            return []

    class FakePublic:
        def get_order_book(self, *, token_id):
            return SimpleNamespace(bids=[], asks=[])

    monkeypatch.setattr(bot_mod, "SecureClient", FakeSecure)
    monkeypatch.setattr(bot_mod, "PublicClient", FakePublic)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xprivkey")
    monkeypatch.setenv("POLYMARKET_WALLET_ADDRESS", "0xwallet")

    settings = Settings(data_store_dir=tmp_path / "data")
    settings.trading_mode = "live"

    BotManager(settings, db_path=str(tmp_path / "bot.db"))

    assert len(created_with) == 1
    assert created_with[0]["private_key"] == "0xprivkey"
    assert created_with[0]["wallet"] == "0xwallet"


def test_bot_module_has_no_py_clob_client_import():
    """No py_clob_client import must remain in core/bot.py."""
    import inspect
    import core.bot as mod
    src = inspect.getsource(mod)
    assert "py_clob_client" not in src
