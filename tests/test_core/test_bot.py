"""BotManager construction smoke test."""
from __future__ import annotations

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
