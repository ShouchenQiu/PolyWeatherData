"""End-to-end model pipeline.

Coordinates data fetching, TFT inference, and Monte Carlo sampling to
produce per-bucket probabilities for a given station and init time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from model.features import REQUIRED_DECODER_SOURCE_COLUMNS, REQUIRED_ENCODER_SOURCE_COLUMNS

if TYPE_CHECKING:
    from config.stations import StationConfig
    from model.mc import McSampler
    from model.store.registry import ModelRegistry
    from model.tft import TftModel


class ModelPipeline:
    """Orchestrate TFT inference and MC sampling.

    Parameters
    ----------
    tft:
        Loaded TftModel instance.
    mc:
        McSampler instance.
    data_mgr:
        Duck-typed data manager that must implement:
          - fetch_hrrr(init_time, station) -> pd.DataFrame
          - fetch_latest_iem_tick(station, now_utc) -> pd.DataFrame
          - build_encoder_obs_window(station, init_time) -> pd.DataFrame
    registry:
        Optional ModelRegistry instance.  When provided, per-station
        max_bias is loaded from the registry and injected into the
        McSampler before each run.  When None the McSampler's existing
        _max_bias value (default 0.0) is used unchanged.
    """

    _N_ENC: int = 72  # encoder lookback window in hours

    def __init__(
        self,
        tft: "TftModel",
        mc: "McSampler",
        data_mgr: Any,
        registry: "ModelRegistry | None" = None,
    ) -> None:
        if tft is None:
            raise TypeError(
                "tft must not be None; provide a TftModel instance"
            )
        self.tft      = tft
        self.mc       = mc
        self.data_mgr = data_mgr
        self.registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        station: "StationConfig",
        init_time: pd.Timestamp,
        buckets: list[tuple[float, float]],
        target_date: "str | None" = None,
    ) -> dict:
        """Run the full pipeline for one init cycle.

        Parameters
        ----------
        station:
            Station configuration (StationConfig with .timezone field).
        init_time:
            HRRR cycle init time (UTC).
        buckets:
            List of (lo_f, hi_f) temperature buckets in degF.
        target_date:
            Date string "YYYY-MM-DD" for the target local calendar day used
            to select the daily-high fxx window (ADR-0010).  When None, all
            48 decoder hours are used (backward-compatible behaviour).

        Returns
        -------
        {
            "station":      str,
            "init_time":    pd.Timestamp,
            "probs":        {(lo_f, hi_f): float},
            "p50_high_f":   float,
            "quantiles":    {fxx: {p10, p50, p90}},
            "last_tick_utc": pd.Timestamp (UTC),
        }
        """
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )

        # 1. Fetch and cache the current HRRR cycle (fxx 0..48, writes to store)
        logger.debug("Pipeline: fetching HRRR init={} station={}",
                     init_utc, station.station_id)
        hrrr_df = self.data_mgr.fetch_hrrr(init_utc, station)

        # 2. Append the H-1 :51 obs tick (narrow window; backfill done separately)
        logger.debug("Pipeline: fetching latest IEM tick for {}", station.station_id)
        self.data_mgr.fetch_latest_iem_tick(station, init_utc)

        # 3. Build encoder window from store (merged HRRR f01 + obs)
        encoder_df = self.data_mgr.build_encoder_obs_window(
            station, init_utc, n_hours=self._N_ENC
        )

        # 4. Decoder: fxx 1..48 from the HRRR fetch
        #    Completeness gate: require non-empty rows and TMP_2m present.
        if hrrr_df.empty or "fxx" not in hrrr_df.columns:
            raise RuntimeError(
                f"HRRR fetch returned no usable data for station="
                f"{station.station_id} init={init_utc}"
            )
        decoder_raw = hrrr_df[hrrr_df["fxx"].between(1, 48)].copy()
        if decoder_raw.empty:
            raise RuntimeError(
                f"HRRR fetch returned no fxx 1..48 rows for station="
                f"{station.station_id} init={init_utc}"
            )
        if "TMP_2m" not in decoder_raw.columns or decoder_raw["TMP_2m"].isna().all():
            raise RuntimeError(
                f"HRRR decoder rows missing TMP_2m for station="
                f"{station.station_id} init={init_utc}"
            )
        decoder_df = decoder_raw.sort_values("fxx").reset_index(drop=True)

        # 4a. Add model-specific features (time_idx, calendar, local_idx, etc.)
        encoder_df = self._add_model_features(
            encoder_df, start_idx=0, init_utc=init_utc,
            station_id=station.station_id,
        )
        decoder_df = self._add_model_features(
            decoder_df, start_idx=self._N_ENC, init_utc=init_utc,
            station_id=station.station_id,
        )

        # 4b. Pre-predict source column validation (before superset fill).
        missing_enc = set(REQUIRED_ENCODER_SOURCE_COLUMNS) - set(encoder_df.columns)
        if missing_enc:
            raise ValueError(
                f"encoder_df is missing required source columns: {sorted(missing_enc)}"
            )
        missing_dec = set(REQUIRED_DECODER_SOURCE_COLUMNS) - set(decoder_df.columns)
        if missing_dec:
            raise ValueError(
                f"decoder_df is missing required source columns: {sorted(missing_dec)}"
            )

        # 4c. Superset column alignment: fill any column absent from one frame with 0.0
        all_cols = set(encoder_df.columns) | set(decoder_df.columns)
        for col in all_cols - set(encoder_df.columns):
            encoder_df[col] = 0.0
        for col in all_cols - set(decoder_df.columns):
            decoder_df[col] = 0.0

        # 4d. Length guard before TFT call.
        if len(encoder_df) != self._N_ENC:
            raise ValueError(
                f"encoder_df must have exactly {self._N_ENC} rows, "
                f"got {len(encoder_df)}"
            )
        if len(decoder_df) != 48:
            raise ValueError(
                f"decoder_df must have exactly 48 rows, got {len(decoder_df)}"
            )

        # 5. TFT inference -> {fxx: {p10, p50, p90}}
        quantiles = self.tft.predict(encoder_df, decoder_df)

        # 5a. Quantile sanity: all finite, ordered p10 <= p50 <= p90.
        for fxx in range(1, 49):
            q = quantiles[fxx]
            p10, p50, p90 = q["p10"], q["p50"], q["p90"]
            if not (np.isfinite(p10) and np.isfinite(p50) and np.isfinite(p90)):
                raise ValueError(
                    f"Non-finite quantile values for fxx={fxx}: "
                    f"p10={p10}, p50={p50}, p90={p90}"
                )
            if not (p10 <= p50 <= p90):
                raise ValueError(
                    f"Quantiles not ordered for fxx={fxx}: "
                    f"p10={p10}, p50={p50}, p90={p90}"
                )

        # 6. Select target fxx for daily-high computation (ADR-0010).
        #    When target_date is provided: use local-calendar-day window and
        #    exclude is_extrapolated rows.  When None: use all 48 fxx.
        if target_date is not None:
            target_fxx = self._target_day_fxx(hrrr_df, str(target_date), station)
            if not target_fxx:
                raise ValueError(
                    f"No fxx rows fall within target_date={target_date} "
                    f"for station={station.station_id} init={init_utc}"
                )
            if "is_extrapolated" in hrrr_df.columns:
                non_extrap = set(
                    hrrr_df.loc[~hrrr_df["is_extrapolated"].astype(bool), "fxx"]
                    .astype(int).tolist()
                )
                target_fxx = [f for f in target_fxx if f in non_extrap]
            if not target_fxx:
                raise ValueError(
                    f"All target fxx rows are extrapolated for "
                    f"station={station.station_id} init={init_utc}"
                )
        else:
            target_fxx = list(range(1, 49))

        # 6a. Build mu_vec / sigma_vec from TFT quantiles over target fxx.
        mu_vec = np.array([quantiles[fxx]["p50"] for fxx in target_fxx], dtype=float)
        sigma_vec = np.array(
            [(quantiles[fxx]["p90"] - quantiles[fxx]["p10"]) / (2 * 1.282)
             for fxx in target_fxx],
            dtype=float,
        )
        if np.any(sigma_vec <= 0):
            bad = [f for f in target_fxx
                   if (quantiles[f]["p90"] - quantiles[f]["p10"]) / (2 * 1.282) <= 0]
            raise ValueError(
                f"Sigma <= 0 for fxx {bad}: quantile spread (p90-p10) must be positive"
            )

        # 7. Load per-station max_bias from registry (if available) then
        #    run MC -> (n_samp,) daily high in degF
        max_bias = (
            self.registry.load_max_bias(station.station_id)
            if self.registry is not None
            else 0.0
        )
        daily_highs = self.mc.sample_daily_high(mu_vec, sigma_vec, max_bias=max_bias)

        # 8. Convert to bucket probabilities + extract sample quantiles
        raw       = self.mc.bucket_probs(daily_highs, buckets)
        probs     = {k: v for k, v in raw.items() if isinstance(k, tuple)}
        p50_high_f = raw["p50_f"]

        logger.info(
            "Pipeline result: station={} init={} p50_high_f={:.1f}F buckets={}",
            station.station_id, init_utc, p50_high_f,
            {f"{lo:.0f}-{hi:.0f}": f"{p:.3f}" for (lo, hi), p in probs.items()},
        )
        return {
            "station":       station.station_id,
            "init_time":     init_time,
            "probs":         probs,
            "p50_high_f":    p50_high_f,
            "quantiles":     quantiles,
            "last_tick_utc": datetime.now(timezone.utc),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_model_features(
        self,
        df: pd.DataFrame,
        start_idx: int,
        init_utc: pd.Timestamp,
        station_id: str,
    ) -> pd.DataFrame:
        """Add model-specific index and calendar columns required by TFT.

        Parameters
        ----------
        df:
            Encoder or decoder DataFrame (already cleaned by Cleaner).
        start_idx:
            First time_idx value for this DataFrame's rows.
            Use 0 for encoder (rows 0..71), N_ENC for decoder (rows 72..119).
        init_utc:
            Model initialisation timestamp (UTC).
        station_id:
            Station identifier used as the TFT group key.

        Columns added
        -------------
        time_idx    : monotonically increasing integer from start_idx
        local_idx   : alias for time_idx (matches training convention)
        month       : calendar month (1-12)
        day_of_year : day of year (1-365)
        hour_of_day : hour of day (0-23)
        sin_hour, cos_hour, sin_doy, cos_doy : cyclical calendar features
        init_time_str : string representation of init_utc
        Node_ID     : station_id (TFT group identifier)
        """
        df = df.copy()
        n = len(df)
        idx = list(range(start_idx, start_idx + n))
        df["time_idx"]  = idx
        df["local_idx"] = idx

        if "valid_time" not in df.columns:
            raise ValueError(
                f"DataFrame passed to _add_model_features must have a 'valid_time' "
                f"column (columns present: {list(df.columns)})"
            )
        vt  = pd.to_datetime(df["valid_time"], utc=True)
        hour = vt.dt.hour.values
        doy  = vt.dt.day_of_year.values
        df["month"]       = vt.dt.month.values
        df["day_of_year"] = doy
        df["hour_of_day"] = hour
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        df["sin_doy"]  = np.sin(2 * np.pi * doy  / 365).astype("float32")
        df["cos_doy"]  = np.cos(2 * np.pi * doy  / 365).astype("float32")

        df["init_time_str"] = str(init_utc)
        df["Node_ID"]       = station_id
        return df

    def _target_day_fxx(
        self,
        hrrr_df: pd.DataFrame,
        target_date: str,
        station: "StationConfig",
    ) -> list[int]:
        """Return fxx integers whose valid_time falls in the target local calendar day.

        Parameters
        ----------
        hrrr_df:
            Full HRRR DataFrame (fxx 0..48) from fetch_hrrr.
        target_date:
            Date string "YYYY-MM-DD" representing the target local calendar day.
        station:
            Station configuration providing the IANA timezone string.

        Returns
        -------
        Sorted list of fxx integers in range 1..48 whose valid_time falls in
        [target_date 00:00:00 local, target_date 24:00:00 local).
        """
        tz = ZoneInfo(station.timezone)
        naive_start = pd.Timestamp(target_date + " 00:00:00")
        utc_start = naive_start.tz_localize(tz).tz_convert("UTC")
        utc_end   = (naive_start + pd.Timedelta(hours=24)).tz_localize(tz).tz_convert("UTC")

        df = hrrr_df[hrrr_df["fxx"].between(1, 48)].copy()
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        mask = (df["valid_time"] >= utc_start) & (df["valid_time"] < utc_end)
        return sorted(df.loc[mask, "fxx"].astype(int).tolist())
