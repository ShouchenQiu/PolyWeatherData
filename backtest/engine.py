"""BacktestEngine: replay historical pipeline runs against CLOB snapshot data.

Ported from Herbie-main/_sniper_backtest.py.

For each init_time in the requested date range:
  1. Run pipeline.run() to get per-bucket MC probabilities.
  2. Look up the contemporaneous CLOB snapshot (+-30 min window).
  3. Compute edge = mc_prob - clob_price.
  4. Record raw PnL and metadata for each (init, bucket) pair.

Key constants
-------------
HALFSPREAD_DRAG : float = -1.27 pts
    Average half-spread cost applied in real PnL calculations.
CLOB_WIN_MIN : int = 30
    Half-width of the CLOB price look-up window in minutes.
MIN_HOURS : int = 8
    Minimum target-day EST hours to include a partial-coverage init.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from config.stations import StationConfig
    from core.strategy.base import BaseStrategy
    from data.manager import DataManager
    from data.storage.parquet_store import ParquetStore
    from model.pipeline import ModelPipeline

# Matching _sniper_analysis.py: average half-spread cost in probability pts
HALFSPREAD_DRAG: float = -1.27 / 100.0   # -0.0127 as a fraction of $1

_MIN_HOURS         = 8   # minimum target-day local hours
_CUTOFF_LOCAL_HOUR = 16  # no new signals at or after 16:00 local time
_CLOB_WIN_MIN      = 30  # CLOB look-up half-window in minutes


def _bucket_outcome(actual_f: float, lo: float, hi: float) -> float:
    """Return 1.0 if actual temperature falls in the bucket, 0.0 otherwise.

    NaN actual -> NaN outcome.
    """
    if np.isnan(actual_f):
        return float("nan")
    if hi == float("inf"):
        return 1.0 if actual_f >= lo else 0.0
    elif lo == float("-inf"):
        return 1.0 if actual_f <= hi else 0.0
    else:
        return 1.0 if (actual_f >= lo and actual_f <= hi) else 0.0


class BacktestEngine:
    """Replay historical model runs against CLOB price snapshots.

    Parameters
    ----------
    pipeline:
        A ModelPipeline instance.  Its ``data_mgr`` must be able to serve
        historical HRRR + obs data for the requested date range.
    strategy:
        BaseStrategy implementation.  Used only to inspect which buckets
        are meaningful; PnL is computed directly from edge, not from
        order placement.
    clob_df:
        Historical CLOB snapshot DataFrame with columns:
          timestamp_utc  (UTC-aware datetime)
          event_end_date (UTC-aware datetime)
          market_slug    (str)
          bucket_lo_f    (float)
          bucket_hi_f    (float)
          mid_price_yes  (float in 0..1)
          estimated_spread (float)
          is_filled      (int: 0=real, 1=synthetic fill)
    actual_highs:
        Mapping of ``date_str -> actual_daily_high_f`` (e.g. from IEM).
        Keys are ``"YYYY-MM-DD"`` strings.
    min_edge:
        Minimum absolute edge required to record a signal row.  Set to
        0.0 to record all (init, bucket) pairs regardless of edge.
    cold_start_hours:
        Number of hours from the start of the date range to skip.
        The encoder window requires ``cold_start_hours`` hours of
        historical data before each init time; inits that fall inside
        this window cannot produce a valid encoder frame and are skipped
        (default 72, matching ``MAX_ENCODER_LENGTH`` [ADR-0001]).
        Ignored when ``data_manager`` is provided.
    data_manager:
        Optional DataManager.  When provided, the time-based
        ``cold_start_hours`` gate is replaced by
        ``data_manager.is_inference_ready()`` for each candidate init
        (V2V04).  Backward-compatible: when None the old
        ``cold_start_hours`` gate remains active.
    """

    def __init__(
        self,
        pipeline:          "ModelPipeline",
        strategy:          "BaseStrategy",
        clob_df:           pd.DataFrame,
        actual_highs:      dict[str, float],
        min_edge:          float = 0.0,
        cold_start_hours:  int   = 72,
        data_manager:      "DataManager | None" = None,
    ) -> None:
        self.pipeline          = pipeline
        self.strategy          = strategy
        self.actual_highs      = actual_highs
        self.min_edge          = min_edge
        self._cold_start_hours = cold_start_hours
        self._data_manager     = data_manager
        self._readiness_skips: list[dict] = []
        self._clob_index       = self._build_clob_index(clob_df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date:   str,
        station:    "StationConfig",
        init_hours: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run the backtest over the requested date range.

        Parameters
        ----------
        start_date / end_date:
            Inclusive date strings ``"YYYY-MM-DD"`` (UTC calendar dates).
        station:
            Station configuration for the pipeline.
        init_hours:
            UTC hours to use as init times (default 0, 6, 12, 18).

        Returns
        -------
        DataFrame with one row per (init_time, bucket) signal, columns:
          target_date, init_utc, init_local, n_hours, coverage_type,
          lead_hours_to_4pm, bucket_lo_f, bucket_hi_f, market_slug,
          mc_prob, clob_price, estimated_spread, halfspread,
          clob_quality, edge, actual_max_f, actual_outcome,
          pnl_if_yes, pnl_if_no
        """
        if init_hours is None:
            init_hours = [0, 6, 12, 18]

        dates = pd.date_range(start_date, end_date, freq="D", tz="UTC")
        cold_start_cutoff = pd.Timestamp(start_date, tz="UTC") + pd.Timedelta(
            hours=self._cold_start_hours
        )
        records: list[dict] = []
        self._readiness_skips = []
        total_inits = len(dates) * len(init_hours)
        logger.info(
            "Backtest starting: {} to {} ({} inits, cold-start cutoff {})",
            start_date, end_date, total_inits, cold_start_cutoff,
        )

        for day in dates:
            for hour in init_hours:
                init_utc = day + pd.Timedelta(hours=hour)
                if self._data_manager is not None:
                    from model.tft import MAX_ENCODER_LENGTH
                    status = self._data_manager.inference_readiness(
                        station, init_utc, MAX_ENCODER_LENGTH
                    )
                    if not status.ready:
                        logger.debug(
                            "Backtest: readiness skip init={} reason={}",
                            init_utc, status.reason,
                        )
                        self._readiness_skips.append({
                            "init_utc":           init_utc,
                            "reason":             status.reason,
                            "first_eligible_init": status.first_eligible_init,
                        })
                        continue
                elif init_utc < cold_start_cutoff:
                    logger.debug("cold-start skip: {}", init_utc)
                    continue
                self._process_init(init_utc, station, records)

        df = pd.DataFrame(records)
        logger.info(
            "Backtest done: {} rows, {} unique inits",
            len(df), df["init_utc"].nunique() if not df.empty else 0,
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clob_index(
        self, df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Pre-index CLOB rows by event_end_date string for fast look-up."""
        idx: dict[str, pd.DataFrame] = {}
        if df.empty:
            return idx
        col = "event_end_date"
        for ev_date, grp in df.groupby(df[col].dt.date):
            idx[str(ev_date)] = grp.sort_values("timestamp_utc").reset_index(
                drop=True
            )
        return idx

    def _clob_window(
        self, event_str: str, init_utc: pd.Timestamp
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Return aggregated CLOB prices in the +-CLOB_WIN_MIN window.

        Prefers ``is_filled=0`` rows; falls back to all rows.
        Returns (aggregated_df, quality) where quality is 'real' or 'filled',
        or (None, None) if no data.
        """
        if event_str not in self._clob_index:
            return None, None
        grp = self._clob_index[event_str]
        lo  = init_utc - pd.Timedelta(minutes=_CLOB_WIN_MIN)
        hi  = init_utc + pd.Timedelta(minutes=_CLOB_WIN_MIN)
        win = grp[
            (grp["timestamp_utc"] >= lo) & (grp["timestamp_utc"] <= hi)
        ]
        if len(win) == 0:
            return None, None
        real = win[win["is_filled"] == 0]
        src  = real if len(real) > 0 else win
        qual = "real" if len(real) > 0 else "filled"
        agg  = src.groupby(
            ["market_slug", "bucket_lo_f", "bucket_hi_f"], as_index=False
        ).agg(
            clob_price        =("mid_price_yes",     "median"),
            estimated_spread  =("estimated_spread",   "median"),
        )
        return agg, qual

    def _process_init(
        self,
        init_utc: pd.Timestamp,
        station:  "StationConfig",
        records:  list[dict],
    ) -> None:
        """Run pipeline for one init time and append rows to records."""
        from collections import Counter

        import datetime as _dt
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(station.timezone)
        init_local = init_utc.to_pydatetime().astimezone(tz)

        # Local dates covered by this init's 48 forecast hours
        fxx_dates = [
            (init_utc + pd.Timedelta(hours=h)).to_pydatetime().astimezone(tz).date()
            for h in range(1, 49)
        ]
        date_count: Counter = Counter(fxx_dates)

        for target_date, n_hours in date_count.items():
            target_str = str(target_date)

            if target_str not in self.actual_highs:
                continue
            if n_hours < _MIN_HOURS:
                continue

            # Cutoff: 16:00 local time on target day
            cutoff_utc = pd.Timestamp(
                _dt.datetime(
                    target_date.year, target_date.month, target_date.day,
                    _CUTOFF_LOCAL_HOUR, tzinfo=tz,
                )
            ).tz_convert("UTC")
            if init_utc >= cutoff_utc:
                continue

            coverage_type = "full" if n_hours == 24 else "partial"
            lead_hours    = (
                (cutoff_utc - init_utc).total_seconds() / 3600.0
            )

            # fxx mask for this target date
            mask = [d == target_date for d in fxx_dates]
            fxx_in_target = [i + 1 for i, m in enumerate(mask) if m]

            # Collect bucket list from CLOB window
            agg, qual = self._clob_window(target_str, init_utc)
            if agg is None:
                continue

            buckets = [
                (float(r["bucket_lo_f"]), float(r["bucket_hi_f"]))
                for _, r in agg.iterrows()
            ]
            if not buckets:
                continue

            # Run pipeline
            try:
                result = self.pipeline.run(
                    station, init_utc, buckets, target_date=target_str
                )
            except Exception:
                continue

            probs      = result.get("probs", {})
            actual_f   = self.actual_highs[target_str]

            for _, br in agg.iterrows():
                lo = float(br["bucket_lo_f"])
                hi = float(br["bucket_hi_f"])
                cp = float(br["clob_price"])
                sp = float(br["estimated_spread"])

                mc_p = probs.get((lo, hi), float("nan"))
                if np.isnan(mc_p):
                    continue

                edge = mc_p - cp
                if abs(edge) < self.min_edge:
                    continue

                out = _bucket_outcome(actual_f, lo, hi)

                records.append({
                    "target_date":        target_str,
                    "init_utc":           str(init_utc)[:16],
                    "init_local":         str(init_local)[:16],
                    "n_hours":            n_hours,
                    "coverage_type":      coverage_type,
                    "lead_hours_to_4pm":  round(lead_hours, 2),
                    "bucket_lo_f":        lo,
                    "bucket_hi_f":        hi,
                    "market_slug":        str(br["market_slug"]),
                    "mc_prob":            round(mc_p, 5),
                    "clob_price":         round(cp, 5),
                    "estimated_spread":   round(sp, 5),
                    "halfspread":         round(sp / 2.0, 5),
                    "clob_quality":       qual,
                    "edge":               round(edge, 5),
                    "actual_max_f":       actual_f,
                    "actual_outcome":     out,
                    "pnl_if_yes":         round(float(out) - cp, 5)
                                          if not np.isnan(out) else float("nan"),
                    "pnl_if_no":          round(cp - float(out), 5)
                                          if not np.isnan(out) else float("nan"),
                })
