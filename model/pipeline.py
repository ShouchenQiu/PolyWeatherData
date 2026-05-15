"""End-to-end model pipeline.

Coordinates data fetching, TFT inference, and Monte Carlo sampling to
produce per-bucket probabilities for a given station and init time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

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
    ) -> dict:
        """Run the full pipeline for one init cycle.

        Parameters
        ----------
        station:
            Station configuration.
        init_time:
            HRRR cycle init time (UTC).
        buckets:
            List of (lo_f, hi_f) temperature buckets in degF.

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
        if not hrrr_df.empty:
            decoder_df = hrrr_df[hrrr_df["fxx"].between(1, 48)].copy()
        else:
            decoder_df = pd.DataFrame([
                {
                    "fxx":        fxx,
                    "init_time":  init_utc,
                    "valid_time": init_utc + pd.Timedelta(hours=fxx),
                }
                for fxx in range(1, 49)
            ])

        # 4a. Add model-specific features (time_idx, calendar, local_idx, etc.)
        encoder_df = self._add_model_features(
            encoder_df, start_idx=0, init_utc=init_utc,
            station_id=station.station_id,
        )
        decoder_df = self._add_model_features(
            decoder_df, start_idx=self._N_ENC, init_utc=init_utc,
            station_id=station.station_id,
        )

        # 4b. Superset column alignment: fill any column absent from one frame with 0.0
        all_cols = set(encoder_df.columns) | set(decoder_df.columns)
        for col in all_cols - set(encoder_df.columns):
            encoder_df[col] = 0.0
        for col in all_cols - set(decoder_df.columns):
            decoder_df[col] = 0.0

        # 5. TFT inference -> {fxx: {p10, p50, p90}}
        quantiles = self.tft.predict(encoder_df, decoder_df)

        # 6. Build mu_vec / sigma_vec from TFT quantiles over decoder hours
        mu_vec    = np.array([quantiles[fxx]["p50"] for fxx in range(1, 49)],
                             dtype=float)
        sigma_vec = np.array(
            [(quantiles[fxx]["p90"] - quantiles[fxx]["p10"]) / (2 * 1.282)
             for fxx in range(1, 49)],
            dtype=float,
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

        if "valid_time" in df.columns:
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
        else:
            for col in ("month", "day_of_year", "hour_of_day",
                        "sin_hour", "cos_hour", "sin_doy", "cos_doy"):
                df[col] = 0.0

        df["init_time_str"] = str(init_utc)
        df["Node_ID"]       = station_id
        return df
