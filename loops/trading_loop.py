"""TradingLoop: wired trading cycle over all active events.

Each tick:
  1. Fetches active events from the DB.
  2. For each event checks data freshness (ADR-0006); skips stale events.
  3. Reads ProbabilityStore for the event's day_offset.
  4. Fetches live CLOB books (if orderbook is wired).
  5. Generates DirectionalSignals via MultiStrategy / EdgeStrategy.
  6. Arbitrates each signal: bullish -> BUY YES; bearish -> SELL YES (if
     YES inventory) or BUY NO.
  7. Depth-checks and re-quotes with actual size via compute_quote.
  8. Filters through RiskPipeline; emits risk_rejected alert on rejection.
  9. Submits approved orders to the exchange.
 10. Records submitted orders via PositionTracker; links signal rows in DB.

References: ADR-0004 (two-loop design), ADR-0006 (stale-data halt),
            ADR-0011 (L2 VWAP pricing layer).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from core.alerts import AlertBus, AlertEvent
from core.pricing.quote import compute_quote
from core.probability_store import ProbabilitySnapshot, ProbabilityStore
from core.risk.pipeline import RiskPipeline
from core.strategy.multi import MultiStrategy
from core.types import OrderIntent
from data.clob.orderbook import OrderbookReader

if TYPE_CHECKING:
    from config.settings import Settings
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
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
        is_paper: bool = False,
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
        self._is_paper   = is_paper

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

        active_events  = self._db.get_tradeable_events(now.date().isoformat())
        submitted: list[str] = []
        total_signals  = 0

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
            total_signals += len(signals)

            current_exposure = self._positions.gross_open_notional_usd()
            local_yes_buys: set[str] = set()

            for signal in signals:
                tick_utc_str = now.isoformat()
                signal_id    = str(uuid.uuid4())

                # 1. Write pending signal row before arbitration
                self._db.record_signal(
                    signal_id       = signal_id,
                    yes_token_id    = signal.yes_token_id,
                    no_token_id     = signal.no_token_id,
                    direction       = signal.direction,
                    mc_prob         = signal.model_prob,
                    reference_price = signal.reference_price,
                    edge            = signal.edge,
                    tick_utc        = tick_utc_str,
                )

                # 2. Directional arbitration: resolve intent token and side
                if signal.direction == "bullish":
                    intent_token_id = signal.yes_token_id
                    intent_side     = "BUY"
                else:  # bearish
                    has_yes = (
                        self._positions.position_shares(signal.yes_token_id) > 0
                        or signal.yes_token_id in local_yes_buys
                    )
                    if has_yes:
                        intent_token_id = signal.yes_token_id
                        intent_side     = "SELL"
                    else:
                        intent_token_id = signal.no_token_id
                        intent_side     = "BUY"

                if not intent_token_id:
                    self._db.update_signal_decision(
                        signal_id        = signal_id,
                        final_token_id   = signal.yes_token_id,
                        final_side       = intent_side,
                        executable_price = 0.0,
                        approved         = 0,
                        reject_reason    = "no_token_id_unavailable",
                        computed_size    = None,
                    )
                    continue

                # 3. Initial depth check (skipped when no orderbook is wired)
                if self._orderbook is not None:
                    book = books.get(intent_token_id, {})
                    try:
                        init_quote = compute_quote(
                            book, intent_side, self._settings.quote_size_usd
                        )
                    except Exception:
                        self._db.update_signal_decision(
                            signal_id        = signal_id,
                            final_token_id   = intent_token_id,
                            final_side       = intent_side,
                            executable_price = 0.0,
                            approved         = 0,
                            reject_reason    = "depth_check_error",
                            computed_size    = None,
                        )
                        continue
                    if not init_quote.complete_fill:
                        self._db.update_signal_decision(
                            signal_id        = signal_id,
                            final_token_id   = intent_token_id,
                            final_side       = intent_side,
                            executable_price = 0.0,
                            approved         = 0,
                            reject_reason    = "insufficient_depth",
                            computed_size    = None,
                        )
                        continue
                    executable_price = init_quote.vwap_price
                else:
                    executable_price = signal.reference_price

                # 4. Create OrderIntent
                intent = OrderIntent(
                    token_id         = intent_token_id,
                    side             = intent_side,
                    size_usd         = None,
                    executable_price = executable_price,
                    edge             = signal.edge,
                    bucket_lo_f      = signal.bucket_lo_f,
                    bucket_hi_f      = signal.bucket_hi_f,
                    condition_id     = condition_id,
                )

                # 5. Risk approval
                if not self._risk.approve_signal(
                    intent, current_exposure=current_exposure
                ):
                    self._emit_risk_rejected(intent, condition_id)
                    self._db.update_signal_decision(
                        signal_id        = signal_id,
                        final_token_id   = intent_token_id,
                        final_side       = intent_side,
                        executable_price = executable_price,
                        approved         = 0,
                        reject_reason    = "risk_rejected",
                        computed_size    = None,
                    )
                    continue

                # 6. Compute size
                size_usd = self._risk.compute_size(
                    intent,
                    current_exposure = current_exposure,
                    positions        = self._positions,
                )
                if size_usd <= 0:
                    self._db.update_signal_decision(
                        signal_id        = signal_id,
                        final_token_id   = intent_token_id,
                        final_side       = intent_side,
                        executable_price = executable_price,
                        approved         = 0,
                        reject_reason    = "zero_size",
                        computed_size    = size_usd,
                    )
                    continue

                # 7. Re-quote with actual size and re-check edge (skipped when
                #    no orderbook is wired; EdgeStrategy already checked depth)
                if self._orderbook is not None:
                    book = books.get(intent_token_id, {})
                    try:
                        fresh_quote = compute_quote(book, intent_side, size_usd)
                    except Exception:
                        self._db.update_signal_decision(
                            signal_id        = signal_id,
                            final_token_id   = intent_token_id,
                            final_side       = intent_side,
                            executable_price = executable_price,
                            approved         = 0,
                            reject_reason    = "requote_error",
                            computed_size    = size_usd,
                        )
                        continue
                    if not fresh_quote.complete_fill:
                        self._db.update_signal_decision(
                            signal_id        = signal_id,
                            final_token_id   = intent_token_id,
                            final_side       = intent_side,
                            executable_price = executable_price,
                            approved         = 0,
                            reject_reason    = "requote_insufficient_depth",
                            computed_size    = size_usd,
                        )
                        continue
                    # Verify edge still clears min_edge at re-quoted price
                    if intent_side == "BUY" and intent_token_id == signal.yes_token_id:
                        requote_edge = signal.model_prob - fresh_quote.vwap_price
                    elif intent_side == "BUY":  # BUY NO
                        requote_edge = (
                            (1.0 - signal.model_prob) - fresh_quote.vwap_price
                        )
                    else:  # SELL YES
                        requote_edge = fresh_quote.vwap_price - signal.model_prob
                    if requote_edge < self._settings.min_edge:
                        self._db.update_signal_decision(
                            signal_id        = signal_id,
                            final_token_id   = intent_token_id,
                            final_side       = intent_side,
                            executable_price = fresh_quote.vwap_price,
                            approved         = 0,
                            reject_reason    = "stale_edge",
                            computed_size    = size_usd,
                        )
                        continue
                    executable_price        = fresh_quote.vwap_price
                    intent.executable_price = executable_price

                # 8. Record approved decision
                self._db.update_signal_decision(
                    signal_id        = signal_id,
                    final_token_id   = intent_token_id,
                    final_side       = intent_side,
                    executable_price = executable_price,
                    approved         = 1,
                    reject_reason    = None,
                    computed_size    = size_usd,
                )

                # 9. Place order and record
                try:
                    order_id = self._exchange.place_order(
                        token_id = intent_token_id,
                        side     = intent_side,
                        size_usd = size_usd,
                        price    = executable_price,
                    )
                    if not order_id:
                        logger.error(
                            "TradingLoop: empty order_id for token {}; "
                            "order not recorded",
                            intent_token_id,
                        )
                        continue
                    self._positions.record_order(
                        order_id         = order_id,
                        token_id         = intent_token_id,
                        condition_id     = condition_id,
                        side             = intent_side,
                        size_usd         = size_usd,
                        price            = executable_price,
                        is_paper         = self._is_paper,
                        executable_price = executable_price,
                    )
                    self._db.update_signal_order_id(
                        signal_id = signal_id,
                        order_id  = order_id,
                    )
                    submitted.append(order_id)

                    # 10. Update per-tick local state immediately
                    current_exposure += size_usd
                    if intent_side == "BUY" and intent_token_id == signal.yes_token_id:
                        local_yes_buys.add(intent_token_id)

                    logger.info(
                        "TradingLoop: submitted order {} for token {}",
                        order_id, intent_token_id,
                    )
                except Exception:
                    logger.exception(
                        "TradingLoop: order submission failed for token {}",
                        intent_token_id,
                    )

        logger.info(
            "TradingLoop tick done: events={} signals={} submitted={}",
            len(active_events), total_signals, len(submitted),
        )
        return submitted

    async def run(self) -> None:
        """Run TradingLoop indefinitely, ticking every interval_s seconds."""
        while True:
            await self.tick()
            await asyncio.sleep(self.interval_s)
