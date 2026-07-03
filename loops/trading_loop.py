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
import time
from typing import TYPE_CHECKING

from loguru import logger

from core.alerts import AlertBus, AlertEvent
from core.observability import ObservabilityRecorder
from core.pricing.quote import compute_quote
from core.probability_store import ProbabilitySnapshot, ProbabilityStore
from core.risk.pipeline import RiskPipeline
from core.strategy.multi import MultiStrategy
from core.trading.decision import (
    FeeSchedule,
    PortfolioState,
    TradingDecisionKernel,
)
from core.types import OrderIntent
from data.clob.orderbook import OrderbookReader

if TYPE_CHECKING:
    from config.settings import Settings
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
    from core.types import DirectionalSignal, OrderIntent
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
        decision_kernel: TradingDecisionKernel | None = None,
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
        self._decision_kernel = decision_kernel
        self._observability = ObservabilityRecorder(db)

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

    def _emit_risk_rejected(self, intent: "OrderIntent", condition_id: str) -> None:
        """Emit a risk_rejected AlertEvent for a rejected OrderIntent."""
        self._alert_bus.emit(
            AlertEvent(
                alert_type  = "risk_rejected",
                station     = None,
                message     = (
                    f"Risk rejected intent for token {intent.token_id} "
                    f"(edge={intent.edge:.4f})"
                ),
                occurred_at = datetime.now(timezone.utc),
                context     = {
                    "token_id":     intent.token_id,
                    "condition_id": condition_id,
                    "edge":         intent.edge,
                },
            )
        )

    def _reject_signal(
        self,
        signal_id: str,
        token_id:  str | None,
        side:      str | None,
        price:     float,
        size:      float | None,
        reason:    str,
    ) -> None:
        """Write a rejected decision row to the signals DB table."""
        self._db.update_signal_decision(
            signal_id        = signal_id,
            final_token_id   = token_id,
            final_side       = side,
            executable_price = price,
            approved         = 0,
            reject_reason    = reason,
            computed_size    = size,
        )

    def _arbitrate(
        self,
        signal:         "DirectionalSignal",
        local_yes_buys: set[str],
    ) -> tuple[str, str] | None:
        """Resolve (intent_token_id, intent_side) from a directional signal.

        Returns a ``(token_id, side)`` pair on success, or ``None`` when the
        token_id cannot be resolved (bearish signal with neither YES inventory
        nor a NO token_id).
        """
        if signal.direction == "bullish":
            return (signal.yes_token_id, "BUY")

        # bearish: SELL YES if YES inventory exists, else BUY NO
        has_yes = (
            self._positions.position_shares(signal.yes_token_id, is_paper=self._is_paper) > 0
            or signal.yes_token_id in local_yes_buys
        )
        if has_yes:
            return (signal.yes_token_id, "SELL")
        if signal.no_token_id:
            return (signal.no_token_id, "BUY")
        return None

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
        started = time.perf_counter()

        active_events  = self._db.get_tradeable_events(now.date().isoformat())
        submitted: list[str] = []
        total_signals  = 0
        mark_prices: dict[str, float] = {}

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
                        mid_price = self._observability.record_market_book(
                            token_id=tid,
                            condition_id=condition_id,
                            tick_utc=now,
                            book=books[tid],
                        )
                        if mid_price is not None:
                            mark_prices[tid] = mid_price
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

                if self._decision_kernel is not None:
                    token_ids = [str(market["token_id"]) for market in markets]
                    portfolio = PortfolioState(
                        cash_usd=(
                            self._settings.starting_cash_usd
                            + self._positions.net_cash_flow_usd(
                                is_paper=self._is_paper
                            )
                        ),
                        shares_by_token={
                            token_id: self._positions.position_shares(
                                token_id,
                                is_paper=self._is_paper,
                            )
                            for token_id in token_ids
                        },
                        notional_by_token={
                            token_id: self._positions.position_size_usd(
                                token_id,
                                is_paper=self._is_paper,
                            )
                            for token_id in token_ids
                        },
                        event_by_token={
                            token_id: condition_id for token_id in token_ids
                        },
                        gross_exposure_usd=current_exposure,
                        event_exposure_by_id={
                            condition_id: self._positions.exposure_by_condition(
                                condition_id
                            )
                        },
                    )
                    decision = self._decision_kernel.decide(
                        signal,
                        books,
                        portfolio,
                        fee_schedule=_fee_schedule_for_signal(
                            signal,
                            markets,
                        ),
                    )
                    if not decision.approved:
                        self._reject_signal(
                            signal_id,
                            decision.token_id,
                            decision.side,
                            decision.executable_price or 0.0,
                            decision.size_usd,
                            decision.reason or "kernel_rejected",
                        )
                        continue
                    intent_token_id = str(decision.token_id)
                    intent_side = str(decision.side)
                    size_usd = float(decision.size_usd)
                    executable_price = float(decision.executable_price)
                    self._db.update_signal_decision(
                        signal_id=signal_id,
                        final_token_id=intent_token_id,
                        final_side=intent_side,
                        executable_price=executable_price,
                        approved=1,
                        reject_reason=None,
                        computed_size=size_usd,
                    )
                    try:
                        order_id = self._exchange.place_order(
                            token_id=intent_token_id,
                            side=intent_side,
                            size_usd=size_usd,
                            price=executable_price,
                        )
                        if not order_id:
                            self._reject_signal(
                                signal_id,
                                intent_token_id,
                                intent_side,
                                executable_price,
                                size_usd,
                                "order_submission_failed",
                            )
                            continue
                        self._positions.record_order(
                            order_id=order_id,
                            token_id=intent_token_id,
                            condition_id=condition_id,
                            side=intent_side,
                            size_usd=size_usd,
                            price=executable_price,
                            is_paper=self._is_paper,
                            executable_price=executable_price,
                            fee_usd=decision.fee_usd,
                        )
                        if self._is_paper:
                            self._positions.mark_filled(
                                order_id,
                                executable_price,
                            )
                        self._db.update_signal_order_id(
                            signal_id=signal_id,
                            order_id=order_id,
                        )
                        submitted.append(order_id)
                        if intent_side == "BUY":
                            current_exposure += size_usd
                            if intent_token_id == signal.yes_token_id:
                                local_yes_buys.add(intent_token_id)
                        else:
                            current_exposure = max(
                                0.0,
                                current_exposure - size_usd,
                            )
                    except Exception:
                        logger.exception(
                            "TradingLoop: shared-kernel order submission failed"
                        )
                        self._reject_signal(
                            signal_id,
                            intent_token_id,
                            intent_side,
                            executable_price,
                            size_usd,
                            "order_submission_failed",
                        )
                    continue

                # 2. Directional arbitration: resolve intent token and side
                result = self._arbitrate(signal, local_yes_buys)
                if result is None:
                    self._reject_signal(signal_id, None, None, 0.0, None,
                                        "no_token_id_unavailable")
                    continue
                intent_token_id, intent_side = result

                # 3. Initial depth check -- orderbook is required for execution.
                #    Reject with orderbook_unavailable when not wired so that
                #    the execution layer contract does not depend on strategies
                #    never emitting signals without books.
                if self._orderbook is None:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        0.0, None, "orderbook_unavailable")
                    continue
                book = books.get(intent_token_id, {})
                try:
                    init_quote = compute_quote(
                        book, intent_side, self._settings.quote_size_usd
                    )
                except Exception:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        0.0, None, "depth_check_error")
                    continue
                if not init_quote.complete_fill:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        0.0, None, "insufficient_depth")
                    continue
                executable_price = init_quote.vwap_price

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
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        executable_price, None, "risk_rejected")
                    continue

                # 6. Compute size
                size_usd = self._risk.compute_size(
                    intent,
                    current_exposure = current_exposure,
                    positions        = self._positions,
                    is_paper         = self._is_paper,
                )
                if size_usd <= 0:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        executable_price, size_usd, "zero_size")
                    continue

                # 7. Re-quote with actual size and re-check edge.
                #    Fetch a fresh book to avoid stale cached prices.
                if self._orderbook is not None:
                    try:
                        fresh_book = self._orderbook.get_book(intent_token_id)
                    except Exception:
                        fresh_book = {}
                else:
                    fresh_book = books.get(intent_token_id, {})
                try:
                    fresh_quote = compute_quote(fresh_book, intent_side, size_usd)
                except Exception:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        executable_price, size_usd, "requote_error")
                    continue
                if not fresh_quote.complete_fill:
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        executable_price, size_usd,
                                        "requote_insufficient_depth")
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
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        fresh_quote.vwap_price, size_usd, "stale_edge")
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
                        self._reject_signal(signal_id, intent_token_id, intent_side,
                                            executable_price, size_usd,
                                            "order_submission_failed")
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
                    if self._is_paper:
                        self._positions.mark_filled(order_id, executable_price)
                    self._db.update_signal_order_id(
                        signal_id = signal_id,
                        order_id  = order_id,
                    )
                    submitted.append(order_id)

                    # 10. Update per-tick local state immediately.
                    #     BUY increases exposure and adds to local_yes_buys.
                    #     SELL reduces exposure and decrements inventory.
                    if intent_side == "BUY":
                        current_exposure += size_usd
                        if intent_token_id == signal.yes_token_id:
                            local_yes_buys.add(intent_token_id)
                    else:  # SELL
                        current_exposure = max(0.0, current_exposure - size_usd)
                        # Remove from local_yes_buys when net shares drop
                        # to or below zero after this SELL.
                        remaining = self._positions.position_shares(
                            intent_token_id, is_paper=self._is_paper,
                        ) - (size_usd / executable_price if executable_price > 0 else 0)
                        if remaining <= 0 and intent_token_id in local_yes_buys:
                            local_yes_buys.discard(intent_token_id)

                    logger.info(
                        "TradingLoop: submitted order {} for token {}",
                        order_id, intent_token_id,
                    )
                except Exception:
                    logger.exception(
                        "TradingLoop: order submission failed for token {}",
                        intent_token_id,
                    )
                    # NOTE: if place_order raised after actually submitting the
                    # order to the exchange, the order may exist on-exchange but
                    # not in the local DB.  Recovery requires an order-reconcile
                    # pass (future slice).
                    self._reject_signal(signal_id, intent_token_id, intent_side,
                                        executable_price, size_usd,
                                        "order_submission_failed")

        try:
            self._observability.record_trading_tick(
                tick_utc=now,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                events_processed=len(active_events),
                signals_processed=total_signals,
                orders_submitted=len(submitted),
                positions=self._positions,
                is_paper=self._is_paper,
                mark_prices=mark_prices,
            )
        except Exception:
            logger.exception("TradingLoop: failed to record observability snapshot")

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


def _fee_schedule_for_signal(
    signal: "DirectionalSignal",
    markets: list[dict],
) -> FeeSchedule:
    tokens = {signal.yes_token_id, signal.no_token_id}
    market = next(
        (
            row
            for row in markets
            if str(row.get("token_id")) in tokens
        ),
        {},
    )
    return FeeSchedule(
        enabled=bool(market.get("fees_enabled", False)),
        rate=market.get("fee_rate"),
        exponent=market.get("fee_exponent", 1.0),
    )
