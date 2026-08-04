# config/settings.py - Global Settings dataclass for PolyWeatherData.
# Polymarket API credentials are loaded from environment variables (.env).
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # Paths
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_store_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data_store"
    )
    obs_store_dir: Path | None = None
    clob_store_dir: Path | None = None

    # Data-source API URLs
    iem_api_url: str = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
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

    # Analysis split windows (ISO dates, inclusive)
    train_start: str = "2024-10-01"
    train_end: str = "2026-03-31"
    val_start: str = "2026-04-01"
    val_end: str = "2026-04-30"
    test_start: str = "2026-05-01"
    test_end: str = "2026-06-04"

    # Mispricing analysis
    clob_snapshot_interval_s: int = 300
    min_stratum_events: int = 30

    def __post_init__(self) -> None:
        if self.obs_store_dir is None:
            self.obs_store_dir = self.data_store_dir / "obs"
        if self.clob_store_dir is None:
            self.clob_store_dir = self.data_store_dir / "backtest"
