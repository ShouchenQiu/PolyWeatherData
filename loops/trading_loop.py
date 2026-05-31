"""TradingLoop: wired trading cycle over all active events.

Each tick:
  1. Fetches active events from the DB.
  2. For each event checks data freshness (ADR-0006); skips stale events.
  3. Reads ProbabilityStore for the event's day_offset.
  4. Fetches live CLOB books (if orderbook is wired).
  5. Generates signals via MultiStrategy.
  6. Filters signals through RiskPipeline; emits risk_rejected on rejection.
  7. Submits approved orders to the exchange.
  8. Records successfully submitted orders via PositionTracker.

References: ADR-0004 (two-loop design), ADR-0006 (stale-data halt).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from core.alerts import AlertBus, AlertEvent
from core.probability_store import ProbabilitySnapshot, ProbabilityStore
from core.risk.pipeline import RiskPipeline
from core.strategy.multi import MultiStrategy
from data.clob.orderbook import OrderbookReader

if TYPE_CHECKING:
    from config.settings import Settings
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
    from core.types import Signal
    from db.schema import DbClient


class TradingLoop:
    """Wired trading cycle: strategy -> risk -> exchange over all active events.

    Parameters
    ----------
    prob_store:
        Shared ProbabilityStore written by ForecastLoop.
    alert_bus:
        AlertBus for broadcasting operational alerts.
    settings:
        Application settings; provides cadence and staleness threshold.
    strategy:
        Exactly one MultiStrategy instance.
    risk:
        Exactly one RiskPipeline instance.
    exchange:
        Exchange adapter (live CLOB or paper).
    positions:
        PositionTracker for recording submitted orders.
    db:
        DbClient for fetching active events and markets.
    orderbook:
        Optional CLOB book fetcher; when None, books dict is empty and
        edge-based strategies will generate no signals.
    """

    def __init__(
        self,
        prob_store: ProbabilityStore,
        alert_bus: AlertBus,
        settings: "Settings",
        strategy: MultiStrategy,
        risk: RiskPipeline,
        exchange: "BaseExchange",
        positions: "PositionTracker",
        db: "DbClient",
        orderbook: OrderbookReader | None = None,
    ) -> None:
        if not isinstance(strategy, MultiStrategy):
            raise TypeError(
                f"strategy must be MultiStrategy, got {type(strategy).__name__}"
            )
        if not isinstance(risk, RiskPipeline):
            raise TypeError(
                f"risk must be RiskPipeline, got {type(risk).__name__}"
            )
        self._prob_store = prob_store
        self._alert_bus  = alert_bus
        self._settings   = settings
        self._strategy   = strategy
        self._risk       = risk
        self._exchange   = exchange
        self._positions  = positions
        self._db         = db
        self._orderbook  = orderbook

    @property
    def interval_s(self) -> int:
        """Tick cadence in seconds, sourced from settings.trading_loop_interval_s."""
        return self._settings.trading_loop_interval_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_data_stale(self, station: str | None, age_min: float) -> None:
        """Emit a data_stale AlertEvent via the AlertBus and persist to DB."""
        if age_min == float("inf"):
            message = "Forecast data missing from ProbabilityStore"
        else:
            message = f"Forecast data stale: {age_min:.1f} min since last update"
        occurred_at = datetime.now(timezone.utc)
        self._alert_bus.emit(
            AlertEvent(
                alert_type  = "data_stale",
                station     = station,
                message     = message,
                occurred_at = occurred_at,
                context     = {"age_min": age_min},
            )
        )
        self._db.set_data_stale(station, occurred_at)

    async def _is_fresh(
        self,
        condition_id: str,
        day_offset: int,
        station: str | None = None,
        now: datetime | None = None,
    ) -> ProbabilitySnapshot | None:
        """Return the ProbabilitySnapshot when data is fresh, or None when stale.

        Emits a data_stale alert and returns None when data is absent or
        older than settings.staleness_threshold_min.  Callers use the
        returned snapshot directly, avoiding a second ProbabilityStore.read().
        """
        if now is None:
            now = datetime.now(timezone.utc)

        snapshot = await self._prob_store.read(condition_id, day_offset)
        if snapshot is None:
            self._on_data_stale(station, float("inf"))
            return None

        if not snapshot.is_fresh(self._settings.staleness_threshold_min, now):
            self._on_data_stale(station, snapshot.age_minutes(now))
            return None

        return snapshot

    def _emit_risk_rejected(self, signal: "Signal", condition_id: str) -> None:
        """Emit a risk_rejected AlertEvent."""
        self._alert_bus.emit(
            AlertEvent(
                alert_type  = "risk_rejected",
                station     = None,
                message     = (
                    f"Risk rejected signal for token {signal.token_id} "
                    f"(edge={signal.edge:.4f})"
                ),
                occurred_at = datetime.now(timezone.utc),
                context     = {
                    "token_id":     signal.token_id,
                    "condition_id": condition_id,
                    "edge":         signal.edge,
                },
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def tick(self, now: datetime | None = None) -> list[str]:
        """Execute one trading cycle over all active events.

        Parameters
        ----------
        now:
            Reference time for staleness and day_offset calculation.
            Defaults to current UTC.

        Returns
        -------
        List of order IDs successfully submitted to the exchange.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        active_events = self._db.get_tradeable_events(now.date().isoformat())
        submitted: list[str] = []

        for event in active_events:
            condition_id = event["condition_id"]
            station_id   = event.get("station_id")

            try:
                day_offset = (
                    date.fromisoformat(event["event_end_date"]) - now.date()
                ).days
            except (KeyError, ValueError, TypeError):
                continue

            snapshot = await self._is_fresh(
                condition_id, day_offset, station=station_id, now=now
            )
            if snapshot is None:
                continue

            markets = self._db.get_markets_for_event(condition_id)
            if not markets:
                continue

            probs: dict[tuple[float, float], float] = dict(snapshot.buckets)
            books: dict[str, dict] = {}
            if self._orderbook is not None:
                for mkt in markets:
                    tid = mkt["token_id"]
                    try:
                        books[tid] = self._orderbook.get_book(tid)
                    except Exception:
                        logger.warning(
                            "TradingLoop: book fetch failed for token {}", tid
                        )

            signals = self._strategy.compute_signals(probs, books, markets)

            current_exposure = self._positions.total_exposure()
            open_positions   = self._positions.open_positions_by_token()

            for signal in signals:
                if not self._risk.approve_signal(
                    signal,
                    current_exposure=current_exposure,
                    open_positions=open_positions,
                ):
                    self._emit_risk_rejected(signal, condition_id)
                    continue

                size_usd = self._risk.compute_size(
                    signal,
                    current_exposure=current_exposure,
                    open_positions=open_positions,
                )
                try:
                    order_id = self._exchange.place_order(
                        token_id = signal.token_id,
                        side     = signal.side,
                        size_usd = size_usd,
                        price    = signal.clob_price,
                    )
                    if not order_id:
                        logger.error(
                            "TradingLoop: empty order_id for token {}; order not recorded",
                            signal.token_id,
                        )
                        continue
                    self._positions.record_order(
                        order_id     = order_id,
                        token_id     = signal.token_id,
                        condition_id = condition_id,
                        side         = signal.side,
                        size_usd     = size_usd,
                        price        = signal.clob_price,
                    )
                    submitted.append(order_id)
                    logger.info(
                        "TradingLoop: submitted order {} for token {}",
                        order_id, signal.token_id,
                    )
                except Exception:
                    logger.exception(
                        "TradingLoop: order submission failed for token {}",
                        signal.token_id,
                    )

        logger.info("TradingLoop tick done: {} order(s) submitted", len(submitted))
        return submitted

    async def run(self) -> None:
        """Run TradingLoop indefinitely, ticking every interval_s seconds."""
        while True:
            await self.tick()
            await asyncio.sleep(self.interval_s)
