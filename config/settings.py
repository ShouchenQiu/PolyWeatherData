# config/settings.py - Global Settings dataclass for PolyWeatherBot.
# All hardcoded constants live here.
# Polymarket API credentials are loaded from environment variables (.env).
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (no-op if already set or file missing)
load_dotenv(override=False)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # Paths
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_store_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data_store"
    )
    model_store_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "model" / "store"
    )

    # Trading
    trading_mode: str = field(
        default_factory=lambda: os.environ.get("TRADING_MODE", "paper")
    )
    min_edge: float = 0.04
    max_position_usd: float = 50.0
    max_total_usd: float = 500.0

    # HRRR source
    hrrr_s3_bucket: str = "noaa-hrrr-bdp-pds"
    hrrr_zarr_bucket: str = "hrrrzarr"
    hrrr_source_priority: list[str] = field(
        default_factory=lambda: ["s3", "nomads"]
    )

    # Monte-Carlo
    mc_n_samp: int = 10000
    mc_rng_seed: int = 42

    # Data-source API URLs
    iem_api_url: str = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    )
    noaa_tgftp_url_tpl: str = (
        "https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT"
    )
    awc_metar_url: str = "https://aviationweather.gov/api/data/metar"
    hrrr_nomads_base: str = (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
    )
    gamma_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "GAMMA_API_URL", "https://gamma-api.polymarket.com"
        )
    )
    clob_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "CLOB_API_URL", "https://clob.polymarket.com"
        )
    )

    # Scheduling / TradingLoop
    trading_loop_interval_s: int = 300
    staleness_threshold_min: int = 65
    staleness_edge_bump: float = 0.02

    # Feature flags
    enable_sniper: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_SNIPER", "").lower() in ("1", "true", "yes")
    )

    # Model training windows (ISO dates, inclusive)
    cold_start_hours: int = 72
    train_start: str = "2024-10-01"
    train_end: str = "2025-12-31"
    val_start: str = "2026-01-01"
    val_end: str = "2026-02-28"
    test_start: str = "2026-03-01"
    test_end: str = "2026-04-30"

    # Risk
    max_open_per_token: int = 1
    quote_size_usd: float = 50.0

    # Polymarket credentials (V2: private-key wallet pair; read-only properties)
    @property
    def polymarket_private_key(self) -> str:
        return os.environ.get("POLYMARKET_PRIVATE_KEY", "")

    @property
    def polymarket_wallet_address(self) -> str:
        return os.environ.get("POLYMARKET_WALLET_ADDRESS", "")
