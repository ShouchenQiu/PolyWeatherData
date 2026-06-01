"""ForecastLoop: hourly model-pipeline cycle that populates ProbabilityStore.

On each tick:
  1. Query active events from the DB.
  2. For each active event + station: run pipeline.run() -> probs.
  3. Wrap probs in ProbabilitySnapshot and write to ProbabilityStore.

Trading decisions (orderbook, strategy, risk, exchange, PositionTracker)
are the sole responsibility of TradingLoop (ADR-0004).
"""
from __future__ import annotations

import asyncio

from datetime import date, datetime, timedelta, timezone
import time
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from core.observability import ObservabilityRecorder
from core.probability_store import ProbabilitySnapshot, ProbabilityStore
from config.stations import STATIONS
from model.tft import MAX_ENCODER_LENGTH

if TYPE_CHECKING:
    from data.manager import DataManager
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
    data_manager:
        Optional DataManager used to gate pipeline execution on encoder
        readiness (V2V04).  When provided, each per-event tick calls
        ``data_manager.inference_readiness()`` and skips the pipeline if
        the encoder window is not yet ready, logging the reason and
        ``first_eligible_init``.  When None the readiness gate is skipped
        (backward-compatible with earlier callers that do not inject
        DataManager).
    """

    def __init__(
        self,
        pipeline:      "ModelPipeline",
        prob_store:    ProbabilityStore,
        db:            "DbClient",
        station:       Any,
        data_manager:  "DataManager | None" = None,
    ) -> None:
        self.pipeline      = pipeline
        self.prob_store    = prob_store
        self.db            = db
        self.station       = station
        self.data_manager  = data_manager
        self.observability = ObservabilityRecorder(db)

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

        started = time.perf_counter()
        logger.info("ForecastLoop tick: init_time={}", init_utc)
        forecast_events = self.db.get_forecast_candidate_events(init_utc.date().isoformat())
        logger.debug("ForecastLoop: {} forecast event(s)", len(forecast_events))
        written: list[str] = []
        errors: list[str] = []

        for event in forecast_events:
            condition_id = event["condition_id"]

            # Derive station_config from event station_id
            station_id = event.get("station_id")
            station_config = STATIONS.get(station_id) if station_id else None
            if station_config is None:
                logger.warning(
                    "ForecastLoop: event {} has unknown station_id {!r}, skipping",
                    condition_id, station_id,
                )
                continue

            # day_offset: days from init_utc.date() to event_end_date
            try:
                day_offset = (
                    date.fromisoformat(event["event_end_date"]) - init_utc.date()
                ).days
            except (KeyError, ValueError, TypeError):
                logger.warning(
                    "ForecastLoop: event {} has invalid event_end_date, skipping",
                    condition_id,
                )
                continue

            markets = self.db.get_markets_for_event(condition_id)
            if not markets:
                continue

            # get_markets_for_event already applies the DB hard gate
            # (discovery_status='parsed' AND bucket_lo_f IS NOT NULL AND
            # bucket_hi_f IS NOT NULL), so null-bound rows never reach here.
            buckets: list[tuple[float, float]] = [
                (m["bucket_lo_f"], m["bucket_hi_f"])
                for m in markets
                if str(m["outcome"]).upper() == "YES"
            ]
            if not buckets:
                continue

            # --- V2V04: inference readiness gate ---
            if self.data_manager is not None:
                status = self.data_manager.inference_readiness(
                    station_config, init_utc, encoder_hours=MAX_ENCODER_LENGTH
                )
                if not status.ready:
                    logger.info(
                        "ForecastLoop: skipping event {} -- reason={} "
                        "first_eligible_init={}",
                        condition_id, status.reason, status.first_eligible_init,
                    )
                    continue

            try:
                result = self.pipeline.run(
                    station_config, init_utc, buckets,
                    target_date=event["event_end_date"],
                )
            except Exception as exc:
                logger.exception("Pipeline failed for event {}", condition_id)
                errors.append(f"{condition_id}: {exc}")
                try:
                    self.observability.record_forecast_failure(
                        event=event,
                        init_utc=init_utc,
                        day_offset=day_offset,
                        error=exc,
                    )
                except Exception:
                    logger.exception(
                        "ForecastLoop: failed to record failed snapshot for {}",
                        condition_id,
                    )
                continue
            probs    = result.get("probs", {})
            quantiles = result.get("quantiles", {})
            logger.debug(
                "Event {} p50_high_f={:.1f}",
                condition_id, result.get("p50_high_f", float("nan")),
            )

            if not probs:
                logger.warning(
                    "ForecastLoop: event {} returned empty prob map, skipping write",
                    condition_id,
                )
                continue

            last_tick = result.get("last_tick_utc") or datetime.now(timezone.utc)
            snapshot = ProbabilitySnapshot(
                buckets={
                    (lo, hi): prob
                    for (lo, hi), prob in probs.items()
                },
                quantiles_by_fxx=dict(quantiles),
                last_tick_utc=last_tick,
            )
            try:
                await self.prob_store.write(condition_id, day_offset, snapshot)
            except Exception as exc:
                logger.exception(
                    "ForecastLoop: prob_store.write failed for event {}", condition_id
                )
                errors.append(f"{condition_id}: {exc}")
                continue

            try:
                self.observability.record_forecast_success(
                    event=event,
                    init_utc=init_utc,
                    day_offset=day_offset,
                    markets=markets,
                    result=result,
                    snapshot=snapshot,
                )
            except Exception as exc:
                logger.exception(
                    "ForecastLoop: failed to record observability snapshot for {}",
                    condition_id,
                )
                errors.append(f"{condition_id}: observability write failed: {exc}")

            logger.info(
                "ForecastLoop: wrote {} bucket(s) for event {}",
                len(snapshot.buckets), condition_id,
            )
            written.append(condition_id)

        if errors and written:
            status = "partial"
        elif errors:
            status = "failed"
        else:
            status = "success"
        try:
            self.observability.record_forecast_tick(
                tick_utc=datetime.now(timezone.utc),
                status=status,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                events_processed=len(forecast_events),
                error="; ".join(errors) if errors else None,
            )
        except Exception:
            logger.exception("ForecastLoop: failed to record loop tick")

        logger.info("ForecastLoop tick done: {} event(s) written", len(written))
        return written

    async def run(self) -> None:
        """Run ForecastLoop indefinitely, aligned to wall-clock UTC hours.

        Each iteration: tick() first, then sleep until the start of the next
        full UTC hour so subsequent ticks land within +-5 s of the hour boundary.
        """
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("ForecastLoop.run: unhandled exception in tick, continuing")
            now       = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            sleep_s = (next_hour - now).total_seconds()
            logger.info(
                "ForecastLoop: next tick in {:.0f}s at {}", sleep_s, next_hour
            )
            await asyncio.sleep(sleep_s)
