"""ForecastLoop: hourly model-pipeline cycle that populates ProbabilityStore.

On each tick:
  1. Query active events from the DB.
  2. For each active event + station: run pipeline.run() -> probs.
  3. Wrap probs in BucketPayload and write to ProbabilityStore.

Trading decisions (orderbook, strategy, risk, exchange, PositionTracker)
are the sole responsibility of TradingLoop (ADR-0004).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from core.probability_store import BucketPayload, ProbabilityStore

if TYPE_CHECKING:
    from db.schema import DbClient
    from model.pipeline import ModelPipeline


class ForecastLoop:
    """Run one model-pipeline cycle per active event and write to ProbabilityStore.

    Parameters
    ----------
    pipeline:
        ModelPipeline that returns per-bucket probabilities.
    prob_store:
        Shared ProbabilityStore written here; read by TradingLoop.
    db:
        DbClient for querying active events and markets.
    station:
        Station configuration for the pipeline.
    """

    def __init__(
        self,
        pipeline:   "ModelPipeline",
        prob_store: ProbabilityStore,
        db:         "DbClient",
        station:    Any,
    ) -> None:
        self.pipeline   = pipeline
        self.prob_store = prob_store
        self.db         = db
        self.station    = station

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def tick(self, init_time: pd.Timestamp | None = None) -> list[str]:
        """Execute one forecast cycle.

        Parameters
        ----------
        init_time:
            Override the HRRR init time.  Defaults to the current UTC hour.

        Returns
        -------
        List of condition_ids successfully written to ProbabilityStore.
        """
        if init_time is None:
            init_time = pd.Timestamp.utcnow().floor("h")

        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )

        logger.info("ForecastLoop tick: init_time={}", init_utc)
        active_events = self.db.get_active_events()
        logger.debug("ForecastLoop: {} active event(s)", len(active_events))
        written: list[str] = []

        for event in active_events:
            condition_id = event["condition_id"]

            # day_offset: days from init_utc.date() to event_end_date, restricted to {0,1,2}
            try:
                delta = (
                    date.fromisoformat(event["event_end_date"]) - init_utc.date()
                ).days
            except (KeyError, ValueError, TypeError):
                logger.warning(
                    "ForecastLoop: event {} has invalid event_end_date, skipping",
                    condition_id,
                )
                continue
            if delta not in (0, 1, 2):
                logger.debug(
                    "ForecastLoop: event {} day_offset={} outside 0..2, skipping",
                    condition_id, delta,
                )
                continue
            day_offset = delta

            markets = self.db.get_markets_for_event(condition_id)
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
                result = self.pipeline.run(self.station, init_utc, buckets)
            except Exception:
                logger.exception("Pipeline failed for event {}", condition_id)
                continue
            probs = result.get("probs", {})
            logger.debug(
                "Event {} p50_high_f={:.1f}",
                condition_id, result.get("p50_high_f", float("nan")),
            )

            last_tick = datetime.now(timezone.utc)
            bucket_map = {
                (lo, hi): BucketPayload(
                    mc_prob=prob,
                    quantiles_by_fxx={},
                    last_tick_utc=last_tick,
                )
                for (lo, hi), prob in probs.items()
            }
            try:
                await self.prob_store.write(condition_id, day_offset, bucket_map)
            except Exception:
                logger.exception(
                    "ForecastLoop: prob_store.write failed for event {}", condition_id
                )
                continue

            logger.info(
                "ForecastLoop: wrote {} bucket(s) for event {}",
                len(bucket_map), condition_id,
            )
            written.append(condition_id)

        logger.info("ForecastLoop tick done: {} event(s) written", len(written))
        return written
