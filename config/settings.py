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

    # HRRR grid
    hrrr_s3_bucket: str = "noaa-hrrr-bdp-pds"
    hrrr_zarr_bucket: str = "hrrrzarr"
    klga_grid_row: int = 698    # Zarr row  (S3-idx row: 676)
    klga_grid_col: int = 1556   # Zarr col  (S3-idx col: 1168)

    # Monte-Carlo
    mc_n_samp: int = 5000
    mc_rng_seed: int = 42
    mc_rho: float = 0.90
    mc_bias_corr_c: float = 1.5206

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

    # Scheduling
    forecast_interval_min: int = 60
    clob_window_min: int = 30
    cutoff_hour_utc: int = 21       # 16:00 EST
    min_hours_before_cutoff: int = 8

    # Polymarket credentials (read-only properties)
    @property
    def polymarket_api_key(self) -> str:
        return os.environ.get("POLYMARKET_API_KEY", "")

    @property
    def polymarket_api_secret(self) -> str:
        return os.environ.get("POLYMARKET_API_SECRET", "")

    @property
    def polymarket_passphrase(self) -> str:
        return os.environ.get("POLYMARKET_PASSPHRASE", "")

    @property
    def polymarket_funder(self) -> str:
        return os.environ.get("POLYMARKET_FUNDER", "")

    @property
    def polymarket_proxy_address(self) -> str:
        return os.environ.get("POLYMARKET_PROXY_ADDRESS", "")

    @property
    def chain_id(self) -> int:
        return int(os.environ.get("CHAIN_ID", "137"))
