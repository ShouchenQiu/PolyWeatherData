"""SniperLoop: event-driven trading triggered by real-time METAR observations.

Consumes the SniperManager queue.  For each new METAR observation it
re-runs the pipeline with the updated observation as the latest encoder
point, then issues FOK (fill-or-kill) style orders via the exchange.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from core.exchange.base import BaseExchange
    from core.positions import PositionTracker
    from core.risk.base import BaseRiskModel
    from core.strategy.base import BaseStrategy
    from db.schema import DbClient
    from model.pipeline import ModelPipeline
    from snipers.manager import SniperManager


class SniperLoop:
    """Consume METAR observations and fire rapid pipeline+order cycles.

    Parameters
    ----------
    sniper_mgr:
        SniperManager whose queue yields (metar_text, source, timestamp).
    pipeline:
        ModelPipeline to run on each new observation.
    orderbook:
        Duck-typed fetcher with ``get_book(token_id) -> dict``.
    strategy:
        BaseStrategy implementation.
    risk:
        BaseRiskModel implementation.
    exchange:
        BaseExchange implementation.
    positions:
        PositionTracker backed by the DB.
    db:
        DbClient for querying active events and markets.
    station:
        Station configuration for the pipeline.
    """

    def __init__(
        self,
        sniper_mgr: "SniperManager",
        pipeline:   "ModelPipeline",
        orderbook:  Any,
        strategy:   "BaseStrategy",
        risk:       "BaseRiskModel",
        exchange:   "BaseExchange",
        positions:  "PositionTracker",
        db:         "DbClient",
        station:    Any,
    ) -> None:
        self.sniper_mgr = sniper_mgr
        self.pipeline   = pipeline
        self.orderbook  = orderbook
        self.strategy   = strategy
        self.risk       = risk
        self.exchange   = exchange
        self.positions  = positions
        self.db         = db
        self.station    = station

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume the sniper queue indefinitely.

        Each item dequeued triggers a rapid pipeline cycle.  The loop
        exits only when the task is cancelled.
        """
        import asyncio

        while True:
            metar_text, source, timestamp = await self.sniper_mgr.queue.get()
            try:
                await self._handle_observation(metar_text, source, timestamp)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_observation(
        self,
        metar_text: str,
        source:     str,
        timestamp:  Any,
    ) -> None:
        """Re-run pipeline with the new METAR and place signals."""
        import asyncio

        init_time    = pd.Timestamp.utcnow().floor("h")
        active_events = self.db.get_active_events()

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

            try:
                result = await asyncio.to_thread(
                    self.pipeline.run, self.station, init_time, buckets
                )
            except Exception:
                continue
            probs = result.get("probs", {})

            books = {}
            for m in markets:
                if m["outcome"] != "YES":
                    continue
                try:
                    books[m["token_id"]] = self.orderbook.get_book(
                        m["token_id"]
                    )
                except Exception:
                    pass

            signals  = self.strategy.compute_signals(probs, books, markets)
            exposure = self.positions.total_exposure()
            for sig in signals:
                if not self.risk.approve_signal(sig, exposure):
                    continue
                size = self.risk.compute_size(sig, exposure)
                if size <= 0:
                    continue
                order_id = self.exchange.place_order(
                    sig.token_id, sig.side, size, sig.clob_price
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
