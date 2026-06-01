"""Observability recorder for UI-facing runtime snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.pnl import PnLCalculator
from core.pricing.quote import compute_book_metrics

if TYPE_CHECKING:
    from core.positions import PositionTracker
    from core.probability_store import ProbabilitySnapshot
    from db.schema import DbClient


class ObservabilityRecorder:
    """Persist runtime observations without changing trading decisions.

    The trading source of truth remains ProbabilityStore and the existing
    Order/Signal tables; this module records UI-readable snapshots only.
    """

    def __init__(self, db: "DbClient") -> None:
        self._db = db

    def record_forecast_success(
        self,
        event: dict,
        init_utc: pd.Timestamp,
        day_offset: int,
        markets: list[dict],
        result: dict,
        snapshot: "ProbabilitySnapshot",
    ) -> int:
        snapshot_id = self._db.record_forecast_snapshot(
            condition_id=event["condition_id"],
            station_id=event.get("station_id"),
            init_time=init_utc.isoformat(),
            event_end_date=event.get("event_end_date"),
            day_offset=day_offset,
            tick_utc=_isoformat_utc(snapshot.last_tick_utc),
            status="success",
            p50_high_f=result.get("p50_high_f"),
        )
        token_by_bucket = {
            (m["bucket_lo_f"], m["bucket_hi_f"]): m["token_id"]
            for m in markets
            if str(m["outcome"]).upper() == "YES"
        }
        for (lo, hi), prob in snapshot.buckets.items():
            self._db.record_forecast_bucket_prob(
                snapshot_id=snapshot_id,
                token_id=token_by_bucket.get((lo, hi)),
                bucket_lo_f=lo,
                bucket_hi_f=hi,
                mc_prob=prob,
            )
        for fxx, q in snapshot.quantiles_by_fxx.items():
            self._db.record_forecast_quantile(
                snapshot_id=snapshot_id,
                fxx=int(fxx),
                p10=float(q["p10"]),
                p50=float(q["p50"]),
                p90=float(q["p90"]),
            )
        return snapshot_id

    def record_forecast_failure(
        self,
        event: dict,
        init_utc: pd.Timestamp,
        day_offset: int | None,
        error: Exception,
    ) -> int:
        return self._db.record_forecast_snapshot(
            condition_id=event["condition_id"],
            station_id=event.get("station_id"),
            init_time=init_utc.isoformat(),
            event_end_date=event.get("event_end_date"),
            day_offset=day_offset,
            tick_utc=datetime.now(timezone.utc).isoformat(),
            status="failed",
            error=str(error),
        )

    def record_forecast_tick(
        self,
        tick_utc: datetime,
        status: str,
        duration_ms: float,
        events_processed: int,
        error: str | None = None,
    ) -> int:
        return self._db.record_loop_tick(
            loop_name="ForecastLoop",
            tick_utc=tick_utc.isoformat(),
            status=status,
            duration_ms=duration_ms,
            events_processed=events_processed,
            error=error,
        )

    def record_market_book(
        self,
        token_id: str,
        condition_id: str,
        tick_utc: datetime,
        book: dict,
    ) -> float | None:
        metrics = compute_book_metrics(book)
        self._db.record_market_price_snapshot(
            token_id=token_id,
            condition_id=condition_id,
            tick_utc=tick_utc.isoformat(),
            **metrics.as_record(),
        )
        return metrics.mid_price

    def record_trading_tick(
        self,
        tick_utc: datetime,
        duration_ms: float,
        events_processed: int,
        signals_processed: int,
        orders_submitted: int,
        positions: "PositionTracker",
        is_paper: bool,
        mark_prices: dict[str, float],
    ) -> None:
        pnl = PnLCalculator(self._db)
        realized = pnl.realized_pnl(is_paper=is_paper)
        unrealized = pnl.unrealized_pnl(mark_prices, is_paper=is_paper)
        self._db.record_pnl_snapshot(
            tick_utc=tick_utc.isoformat(),
            is_paper=is_paper,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=realized + unrealized,
            gross_exposure=positions.gross_open_notional_usd(),
        )
        self._db.record_loop_tick(
            loop_name="TradingLoop",
            tick_utc=tick_utc.isoformat(),
            status="success",
            duration_ms=duration_ms,
            events_processed=events_processed,
            signals_processed=signals_processed,
            orders_submitted=orders_submitted,
        )


def _isoformat_utc(value: Any) -> str:
    """Return an ISO string for datetime-like values returned by the pipeline."""
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)
