"""PolyWeatherData analysis coordinator.

No live trading loops, no model inference.  Loads events/markets from DB
and coordinates mispricing analysis.
"""
from __future__ import annotations

from config.settings import Settings
from db.schema import DbClient


class AnalysisRunner:
    """Load events and markets from the DB for mispricing analysis."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_path: str | None = None,
    ) -> None:
        self.settings = settings or Settings()
        path = db_path or str(self.settings.data_store_dir / "polyweather.db")
        self.db = DbClient(path)

    def load_events(self) -> list[dict]:
        return self.db.get_all_events()

    def load_markets_for_event(self, condition_id: str) -> list[dict]:
        return self.db.get_markets_for_event(condition_id)
