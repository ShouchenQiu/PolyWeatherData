"""BotManager: top-level coordinator that owns and starts all loops."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


class BotManager:
    """Initialise every component from Settings and run all loops concurrently.

    All loop objects are set as instance attributes so unit tests can
    inject mocks by replacing them after construction.

    Parameters
    ----------
    settings:
        Application-wide settings.
    db_path:
        Override the SQLite path.  Defaults to settings.data_store_dir/bot.db.
    """

    def __init__(
        self,
        settings: "Settings",
        db_path: str | None = None,
    ) -> None:
        self.settings = settings
        self._db_path = db_path
        self._build(settings, db_path)

    # ------------------------------------------------------------------
    # Component wiring
    # ------------------------------------------------------------------

    def _build(self, settings: "Settings", db_path: str | None) -> None:
        """Lazily import and construct every component."""
        from pathlib import Path

        from config.stations import STATIONS
        from core.exchange.paper import PaperExchange
        from core.exchange.settlement import Settler
        from core.positions import PositionTracker
        from core.risk.fixed_fraction import FixedFraction
        from core.strategy.edge import EdgeStrategy
        from data.manager import DataManager
        from data.storage.parquet_store import ParquetStore
        from db.schema import DbClient
        from events.discovery import EventDiscovery
        from events.lifecycle import EventLifecycle
        from loops.discovery_loop import DiscoveryLoop
        from loops.forecast_loop import ForecastLoop
        from loops.settlement_loop import SettlementLoop
        from model.mc import McSampler
        from model.pipeline import ModelPipeline
        from snipers.awc import AwcSniper
        from snipers.manager import SniperManager
        from snipers.noaa import NoaaSniper

        # DB
        path   = db_path or str(settings.data_store_dir / "bot.db")
        self.db = DbClient(path)

        # Data layer
        forecast_store = ParquetStore(settings.data_store_dir / "hrrr")
        obs_store      = ParquetStore(settings.data_store_dir / "obs")
        self.data_mgr  = DataManager(
            forecast_sources = [],
            obs_sources      = [],
            forecast_store   = forecast_store,
            obs_store        = obs_store,
        )

        # Model
        self.mc         = McSampler(settings)
        self.pipeline   = ModelPipeline(tft=None, mc=self.mc,
                                        data_mgr=self.data_mgr)

        # Station (default KLGA)
        self.station    = next(iter(STATIONS.values()))

        # Exchange + risk + positions
        self.exchange   = PaperExchange()
        self.risk       = FixedFraction.from_settings(settings)
        self.positions  = PositionTracker(self.db)

        # Strategy
        self.strategy   = EdgeStrategy(min_edge=settings.min_edge)

        # Orderbook (no real CLOB client in default paper mode)
        self.orderbook  = None

        # Events
        self.discovery  = EventDiscovery(self.db, settings)
        self.lifecycle  = EventLifecycle()

        # Settler
        self.settler    = Settler()

        # Snipers
        first_station   = self.station.station_id
        self.snipers    = SniperManager([
            NoaaSniper(first_station),
            AwcSniper(first_station),
        ])

        # Loops
        self.forecast_loop   = ForecastLoop(
            pipeline  = self.pipeline,
            orderbook = self.orderbook,
            strategy  = self.strategy,
            risk      = self.risk,
            exchange  = self.exchange,
            positions = self.positions,
            db        = self.db,
            station   = self.station,
        )
        self.discovery_loop  = DiscoveryLoop(self.discovery)
        self.settlement_loop = SettlementLoop(
            settler   = self.settler,
            exchange  = self.exchange,
            lifecycle = self.lifecycle,
            db        = self.db,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Launch all loops concurrently.  Runs until cancelled."""
        from loops.sniper_loop import SniperLoop

        sniper_loop = SniperLoop(
            sniper_mgr = self.snipers,
            pipeline   = self.pipeline,
            orderbook  = self.orderbook,
            strategy   = self.strategy,
            risk       = self.risk,
            exchange   = self.exchange,
            positions  = self.positions,
            db         = self.db,
            station    = self.station,
        )

        async def _forecast_hourly():
            """Wrapper: call tick() every hour."""
            import asyncio
            import pandas as pd

            while True:
                await self.forecast_loop.tick()
                await asyncio.sleep(3_600)

        await asyncio.gather(
            _forecast_hourly(),
            sniper_loop.run(),
            self.snipers.start_all(),
            self.discovery_loop.run(),
            self.settlement_loop.run(),
        )
