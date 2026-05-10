"""ForecastLoop: hourly trading cycle driven by the model pipeline.

On each tick:
  1. Query active events from the DB.
  2. For each active event + station: run pipeline.run() -> probs.
  3. Fetch CLOB mid-prices for all relevant token_ids.
  4. strategy.compute_signals(probs, books, markets) -> signals.
  5. For each signal: risk.approve_signal() -> place order -> tracker.open().
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
    from core.risk.base import BaseRiskModel
    from core.strategy.base import BaseStrategy
    from db.schema import DbClient
    from model.pipeline import ModelPipeline


class ForecastLoop:
    """Run one model-pipeline trading cycle per active event.

    Parameters
    ----------
    pipeline:
        ModelPipeline that returns per-bucket probabilities.
    orderbook:
        Duck-typed fetcher with ``get_book(token_id) -> dict``.
    strategy:
        BaseStrategy implementation (e.g. EdgeStrategy).
    risk:
        BaseRiskModel implementation (e.g. FixedFraction).
    exchange:
        BaseExchange implementation (paper or live CLOB).
    positions:
        PositionTracker backed by the DB.
    db:
        DbClient for querying active events and markets.
    station:
        Station configuration for the pipeline.
    """

    def __init__(
        self,
        pipeline:  "ModelPipeline",
        orderbook: Any,
        strategy:  "BaseStrategy",
        risk:      "BaseRiskModel",
        exchange:  "BaseExchange",
        positions: "PositionTracker",
        db:        "DbClient",
        station:   Any,
    ) -> None:
        self.pipeline  = pipeline
        self.orderbook = orderbook
        self.strategy  = strategy
        self.risk      = risk
        self.exchange  = exchange
        self.positions = positions
        self.db        = db
        self.station   = station

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def tick(self, init_time: pd.Timestamp | None = None) -> list:
        """Execute one forecast-and-trade cycle.

        Parameters
        ----------
        init_time:
            Override the HRRR init time.  Defaults to the current UTC
            hour (floor to hour).

        Returns
        -------
        List of Order objects placed during this tick.
        """
        if init_time is None:
            init_time = pd.Timestamp.utcnow().floor("h")

        logger.info("ForecastLoop tick: init_time={}", init_time)
        active_events = self.db.get_active_events()
        logger.debug("ForecastLoop: {} active event(s)", len(active_events))
        placed_orders = []

        for event in active_events:
            condition_id = event["condition_id"]
            markets      = self.db.get_markets_for_event(condition_id)
            if not markets:
                continue

            buckets = [
                (m["bucket_lo_f"], m["bucket_hi_f"])
                for m in markets
                if m["outcome"] == "YES"
            ]
            if not buckets:
                continue

            # 1. Run pipeline
            try:
                result = self.pipeline.run(self.station, init_time, buckets)
            except Exception:
                logger.exception("Pipeline failed for event {}", condition_id)
                continue
            probs = result.get("probs", {})
            logger.debug(
                "Event {} p50_high_f={:.1f}",
                condition_id, result.get("p50_high_f", float("nan")),
            )

            # 2. Fetch orderbooks for each YES token
            books = {}
            for m in markets:
                if m["outcome"] != "YES":
                    continue
                try:
                    books[m["token_id"]] = self.orderbook.get_book(
                        m["token_id"]
                    )
                except Exception:
                    logger.warning("Could not fetch book for token {}", m["token_id"])

            # 3. Compute signals
            signals = self.strategy.compute_signals(probs, books, markets)
            logger.debug("Event {}: {} signal(s) generated", condition_id, len(signals))

            # 4. Risk-check and place orders
            exposure = self.positions.total_exposure()
            for sig in signals:
                if not self.risk.approve_signal(sig, exposure):
                    logger.debug(
                        "Signal rejected by risk: token={} edge={:.4f} exposure={:.2f}",
                        sig.token_id, sig.edge, exposure,
                    )
                    continue
                size = self.risk.compute_size(sig, exposure)
                if size <= 0:
                    continue
                order_id = self.exchange.place_order(
                    sig.token_id, sig.side, size, sig.clob_price
                )
                logger.info(
                    "Order placed: {} {} ${:.2f} @ {:.4f} edge={:.4f} order_id={}",
                    sig.side, sig.token_id, size, sig.clob_price, sig.edge, order_id,
                )
                from core.types import Order
                order = Order(
                    order_id = order_id,
                    token_id = sig.token_id,
                    side     = sig.side,
                    size_usd = size,
                    price    = sig.clob_price,
                )
                self.positions.open(sig, order)
                exposure += size
                placed_orders.append(order)

        logger.info("ForecastLoop tick done: {} order(s) placed", len(placed_orders))
        return placed_orders
