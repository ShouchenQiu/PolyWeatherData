"""End-to-end model pipeline.

Coordinates data fetching, TFT inference, and Monte Carlo sampling to
produce per-bucket probabilities for a given station and init time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings as _Settings

if TYPE_CHECKING:
    from config.stations import StationConfig
    from model.mc import McSampler
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
          - fetch_iem(station, start, end) -> pd.DataFrame
          - build_encoder_obs_window(station, init_time) -> pd.DataFrame
    """

    _N_ENC: int = 72  # encoder lookback window in hours

    def __init__(
        self,
        tft: "TftModel",
        mc: "McSampler",
        data_mgr: Any,
    ) -> None:
        self.tft      = tft
        self.mc       = mc
        self.data_mgr = data_mgr

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
            "station":    str,
            "init_time":  pd.Timestamp,
            "probs":      {(lo_f, hi_f): float},
            "p50_high_f": float,
        }
        """
        settings = _Settings()
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )

        # 1. Fetch and cache the current HRRR cycle (fxx 0..48, writes to store)
        logger.debug("Pipeline: fetching HRRR init={} station={}",
                     init_utc, station.station_id)
        hrrr_df = self.data_mgr.fetch_hrrr(init_utc, station)

        # 2. Populate obs store for encoder window
        obs_end   = init_utc - pd.Timedelta(hours=1)
        obs_start = init_utc - pd.Timedelta(hours=self._N_ENC + settings.cold_start_hours)
        logger.debug("Pipeline: fetching IEM obs {}..{}", obs_start, obs_end)
        self.data_mgr.fetch_iem(station, obs_start, obs_end)

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

        # 7. Monte Carlo -> (n_samp,) daily high in degF
        daily_highs = self.mc.sample_daily_high(mu_vec, sigma_vec)

        # 8. Convert to bucket probabilities
        probs = self.mc.bucket_probs(daily_highs, buckets)

        p50_high_f = float(np.median(daily_highs))

        logger.info(
            "Pipeline result: station={} init={} p50_high_f={:.1f}F buckets={}",
            station.station_id, init_utc, p50_high_f,
            {f"{lo:.0f}-{hi:.0f}": f"{p:.3f}" for (lo, hi), p in probs.items()},
        )
        return {
            "station":         station.station_id,
            "init_time":       init_time,
            "probs":           probs,
            "p50_high_f":      p50_high_f,
            "quantiles":       quantiles,
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

