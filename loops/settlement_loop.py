"""SettlementLoop: confirm temperature-bucket settlements for expired events."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from core.positions import PositionTracker
    from data.storage.parquet_store import ParquetStore
    from db.schema import DbClient
    from events.lifecycle import EventLifecycle


class SettlementLoop:
    """Confirm settlement for events in ``pending_settle`` status.

    For each pending event:
    1. Reads IEM observed max temperature from the ParquetStore.
    2. Identifies the winning bucket (bucket_lo_f <= max_temp < bucket_hi_f).
    3. Marks each filled/closed order settled with the correct payout.
    4. Calls EventLifecycle.set_settled() to finalise the event.

    Parameters
    ----------
    lifecycle:
        EventLifecycle for the set_settled() call.
    positions:
        PositionTracker used to write the per-order settled_payout.
    obs_store:
        ParquetStore holding IEM observed temperature data.
    db:
        DbClient for querying events, markets, and orders.
    interval_seconds:
        How often to run the settlement check.  Default 3600 (1 h).
    """

    def __init__(
        self,
        lifecycle:        "EventLifecycle",
        positions:        "PositionTracker",
        obs_store:        "ParquetStore",
        db:               "DbClient",
        interval_seconds: float = 3_600.0,
    ) -> None:
        self._lifecycle        = lifecycle
        self._positions        = positions
        self._obs_store        = obs_store
        self._db               = db
        self._interval_seconds = interval_seconds

    # ------------------------------------------------------------------
    # Settlement logic
    # ------------------------------------------------------------------

    def confirm_settlement(self, event: dict, station: str | None) -> None:
        """Settle all filled/closed orders for *event* and mark it settled.

        Reads the IEM max temperature for the event date, identifies the
        winning bucket, pays out orders accordingly, then calls set_settled().

        If *station* is None or the obs store has no data for the event date,
        the method logs a warning and returns without modifying the DB so the
        event stays in ``pending_settle`` for the next attempt.

        Parameters
        ----------
        event:
            Row dict from the events table (must contain ``condition_id`` and
            ``event_end_date``).
        station:
            IEM station identifier (e.g. ``"KLGA"``).
        """
        condition_id = event["condition_id"]
        end_date_str = event.get("event_end_date") or ""

        if not station:
            logger.warning(
                "SettlementLoop: no station for event {}; skipping", condition_id
            )
            return

        if not end_date_str:
            logger.warning(
                "SettlementLoop: no event_end_date for event {}; skipping",
                condition_id,
            )
            return

        try:
            day_start = pd.Timestamp(end_date_str, tz="UTC")
        except (ValueError, TypeError):
            logger.warning(
                "SettlementLoop: unparseable event_end_date '{}' for {}; skipping",
                end_date_str,
                condition_id,
            )
            return

        day_end = day_start + pd.Timedelta(days=1)
        obs_df  = self._obs_store.read(station, day_start, day_end)

        if obs_df.empty or "tmpf" not in obs_df.columns:
            logger.warning(
                "SettlementLoop: no tmpf obs for station={} date={}; skipping {}",
                station,
                end_date_str,
                condition_id,
            )
            return

        actual_max_temp: float = obs_df["tmpf"].max()

        # Build a token_id -> (bucket_lo_f, bucket_hi_f) map
        markets = self._db.get_markets_for_event(condition_id)
        token_bucket: dict[str, tuple[float, float]] = {}
        for mkt in markets:
            lo = mkt.get("bucket_lo_f")
            hi = mkt.get("bucket_hi_f")
            if lo is not None and hi is not None:
                token_bucket[mkt["token_id"]] = (float(lo), float(hi))

        # Settle each filled/closed order
        orders = self._db.get_orders_for_settlement(condition_id)
        for order in orders:
            order_id = order["order_id"]
            token_id = order["token_id"]
            size_usd = float(order.get("size_usd") or 0.0)

            bucket = token_bucket.get(token_id)
            if bucket is not None:
                lo, hi = bucket
                payout = size_usd if lo <= actual_max_temp < hi else 0.0
            else:
                logger.warning(
                    "SettlementLoop: token_id {} has no bucket; payout=0", token_id
                )
                payout = 0.0

            self._positions.mark_settled(order_id, payout)

        self._lifecycle.set_settled(condition_id, self._db)
        logger.info(
            "SettlementLoop: event {} settled (max_temp={:.1f}F)",
            condition_id,
            actual_max_temp,
        )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Periodically confirm settlement for all pending_settle events."""
        logger.info("SettlementLoop started (interval={}s)", self._interval_seconds)
        while True:
            try:
                events = self._db.get_pending_settlement_events()
                for event in events:
                    cid        = event["condition_id"]
                    station_id = event.get("station_id")
                    try:
                        await asyncio.to_thread(
                            self.confirm_settlement, event, station_id
                        )
                    except Exception:
                        logger.exception(
                            "SettlementLoop: confirm_settlement failed for {}", cid
                        )
            except Exception:
                logger.exception("SettlementLoop: unhandled error in run()")
            await asyncio.sleep(self._interval_seconds)
