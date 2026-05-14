"""TradingLoop skeleton: staleness gate and AlertBus integration.

Order submission, MultiStrategy, and RiskPipeline wiring are deferred to V11.
This slice (F05) implements only the cadence configuration and the staleness
gate that halts new order submission when ProbabilityStore data is stale.

References: ADR-0004 (two-loop design), ADR-0006 (stale-data halt).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.alerts import AlertBus, AlertEvent
from core.probability_store import ProbabilityStore

if TYPE_CHECKING:
    from config.settings import Settings


class TradingLoop:
    """Reads ProbabilityStore freshness and emits data_stale alerts.

    Parameters
    ----------
    prob_store:
        Shared ProbabilityStore written by ForecastLoop.
    alert_bus:
        AlertBus for broadcasting operational alerts.
    settings:
        Application settings; provides cadence and staleness threshold.
    """

    def __init__(
        self,
        prob_store: ProbabilityStore,
        alert_bus: AlertBus,
        settings: "Settings",
    ) -> None:
        self._prob_store = prob_store
        self._alert_bus = alert_bus
        self._settings = settings

    @property
    def interval_s(self) -> int:
        """Tick cadence in seconds, sourced from settings.trading_loop_interval_s."""
        return self._settings.trading_loop_interval_s

    def _on_data_stale(self, station: str | None, age_min: float) -> None:
        """Emit a data_stale AlertEvent via the AlertBus."""
        if age_min == float("inf"):
            message = "Forecast data missing from ProbabilityStore"
        else:
            message = f"Forecast data stale: {age_min:.1f} min since last update"
        self._alert_bus.emit(
            AlertEvent(
                alert_type="data_stale",
                station=station,
                message=message,
                occurred_at=datetime.now(timezone.utc),
                context={"age_min": age_min},
            )
        )

    async def tick(
        self,
        condition_id: str,
        day_offset: int,
        station: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Apply the staleness gate for one (condition_id, day_offset) pair.

        Parameters
        ----------
        condition_id:
            Polymarket condition/event ID.
        day_offset:
            Day offset in {0, 1, 2}.
        station:
            Station ID associated with this event, forwarded in the alert.
        now:
            Reference time for age calculation; defaults to current UTC.

        Returns
        -------
        bool
            True when data is fresh and order submission may proceed.
            False when data is absent or stale; a data_stale alert is emitted.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        buckets = await self._prob_store.read(condition_id, day_offset)
        if not buckets:  # None or empty dict -> treat as missing data
            self._on_data_stale(station, float("inf"))
            return False

        sample = next(iter(buckets.values()))
        age_min = (now - sample.last_tick_utc).total_seconds() / 60.0
        if age_min > self._settings.staleness_threshold_min:
            self._on_data_stale(station, age_min)
            return False

        return True
