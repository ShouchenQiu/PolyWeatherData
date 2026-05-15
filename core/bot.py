"""BotManager: top-level coordinator that owns and starts all loops."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

try:
    from py_clob_client import ClobClient  # type: ignore[import-untyped]
    from py_clob_client.clob_types import ApiCreds  # type: ignore[import-untyped]
except ImportError:
    ClobClient = None  # type: ignore[assignment,misc]
    ApiCreds   = None  # type: ignore[assignment,misc]

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
        from config.stations import STATIONS
        from core.alerts import AlertBus
        from core.exchange.clob import ClobExchange
        from core.exchange.paper import OrderbookFetcher, PaperExchange
        from core.exchange.settlement import Settler
        from core.positions import PositionTracker
        from core.probability_store import ProbabilityStore
        from core.risk.fixed_fraction import FixedFraction
        from core.strategy.edge import EdgeStrategy
        from data.forecast.hrrr_nomads import HrrrNomadsSource
        from data.forecast.hrrr_s3 import HrrrS3Source
        from data.manager import DataManager
        from data.obs.iem import IemSource
        from data.storage.parquet_store import ParquetStore
        from db.schema import DbClient
        from events.discovery import EventDiscovery
        from events.lifecycle import EventLifecycle
        from loops.discovery_loop import DiscoveryLoop
        from loops.forecast_loop import ForecastLoop
        from loops.settlement_loop import SettlementLoop
        from loops.trading_loop import TradingLoop
        from model.mc import McSampler
        from model.pipeline import ModelPipeline
        from snipers.awc import AwcSniper
        from snipers.sniper_manager import SniperManager
        from snipers.noaa import NoaaSniper

        # DB
        path   = db_path or str(settings.data_store_dir / "bot.db")
        self.db = DbClient(path)

        # Data layer
        forecast_store = ParquetStore(settings.data_store_dir / "hrrr")
        obs_store      = ParquetStore(settings.data_store_dir / "obs")
        self.data_mgr  = DataManager(
            forecast_sources = [HrrrS3Source(), HrrrNomadsSource()],
            obs_sources      = [IemSource()],
            forecast_store   = forecast_store,
            obs_store        = obs_store,
        )

        # Model
        self.mc         = McSampler(settings)
        self.pipeline   = ModelPipeline(tft=None, mc=self.mc,
                                        data_mgr=self.data_mgr)

        # Station (default KLGA)
        self.station    = next(iter(STATIONS.values()))

        # Shared ADR-0004 objects
        self.prob_store = ProbabilityStore()
        self.alert_bus  = AlertBus()

        # Risk + positions
        self.risk      = FixedFraction.from_settings(settings)
        self.positions = PositionTracker(self.db)

        # Strategy
        self.strategy = EdgeStrategy(min_edge=settings.min_edge)

        # Exchange wiring [ADR-0008]
        if settings.trading_mode == "live":
            if ClobClient is None:
                raise ImportError(
                    "py_clob_client is required for live trading mode."
                )
            creds = ApiCreds(
                api_key        = settings.polymarket_api_key,
                api_secret     = settings.polymarket_api_secret,
                api_passphrase = settings.polymarket_passphrase,
            )
            l2_client      = ClobClient(
                host     = settings.clob_api_url,
                chain_id = settings.chain_id,
                creds    = creds,
            )
            self.orderbook = None
            self.exchange  = ClobExchange(l2_client, alert_bus=self.alert_bus)
        else:
            if ClobClient is not None:
                l1_client      = ClobClient(
                    host     = settings.clob_api_url,
                    chain_id = settings.chain_id,
                )
                self.orderbook = OrderbookFetcher(l1_client)
            else:
                self.orderbook = None
            self.exchange = PaperExchange(
                book_fetcher = self.orderbook,
                alert_bus    = self.alert_bus,
            )

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

        # Loops (ADR-0004: ForecastLoop writes ProbabilityStore; TradingLoop reads it)
        self.forecast_loop = ForecastLoop(
            pipeline   = self.pipeline,
            prob_store = self.prob_store,
            db         = self.db,
            station    = self.station,
        )
        self.trading_loop  = TradingLoop(
            prob_store = self.prob_store,
            alert_bus  = self.alert_bus,
            settings   = settings,
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
            orderbook  = self.orderbook,
            risk       = self.risk,
            exchange   = self.exchange,
            positions  = self.positions,
            db         = self.db,
        )

        async def _forecast_hourly():
            """Wrapper: call tick() every hour."""
            while True:
                await self.forecast_loop.tick()
                await asyncio.sleep(3_600)

        async def _trading_cadence():
            """Wrapper: drive TradingLoop at its configured interval."""
            while True:
                # V11 will supply active condition_ids from the DB.
                # For now the loop idles between ticks without processing events.
                await asyncio.sleep(self.trading_loop.interval_s)

        await asyncio.gather(
            _forecast_hourly(),
            _trading_cadence(),
            sniper_loop.run(),
            self.snipers.start_all(),
            self.discovery_loop.run(),
            self.settlement_loop.run(),
        )
