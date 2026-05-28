"""BotManager construction smoke test."""
from __future__ import annotations

import asyncio
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
