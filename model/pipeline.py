"""End-to-end model pipeline.

Coordinates data fetching, TFT inference, and Monte Carlo sampling to
produce per-bucket probabilities for a given station and init time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config.settings import Settings
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
          - fetch_hrrr(init_time) -> pd.DataFrame  (fxx 1..48, HRRR vars)
          - fetch_iem(start, end) -> pd.DataFrame   (obs vars + valid_time)
    """

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
            Station configuration (used for IEM fetch).
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
        # 1. Fetch 48-hour HRRR forecast (decoder rows fxx 1..48)
        hrrr_df = self.data_mgr.fetch_hrrr(init_time)

        # 2. Fetch 24-hour observations for the encoder window
        #    encoder covers fxx -23..0  =>  valid_time = init_time-23h .. init_time
        enc_start = init_time - pd.Timedelta(hours=23)
        enc_end   = init_time
        iem_df    = self.data_mgr.fetch_iem(enc_start, enc_end)

        # 3. Assemble encoder_df and decoder_df
        encoder_df, decoder_df = self._assemble_frames(
            hrrr_df, iem_df, init_time
        )

        # 4. TFT inference -> {fxx: {p10, p50, p90}}
        quantiles = self.tft.predict(encoder_df, decoder_df)

        # 5. Build mu_vec / sigma_vec from TFT quantiles over daytime hours
        #    Use all decoder hours [1..48] for daily high estimation
        mu_vec    = np.array([quantiles[fxx]["p50"] for fxx in range(1, 49)],
                             dtype=float)
        sigma_vec = np.array(
            [(quantiles[fxx]["p90"] - quantiles[fxx]["p10"]) / (2 * 1.282)
             for fxx in range(1, 49)],
            dtype=float,
        )

        # 6. Monte Carlo -> (n_samp,) daily high in degF
        daily_highs = self.mc.sample_daily_high(mu_vec, sigma_vec)

        # 7. Convert to bucket probabilities
        probs = self.mc.bucket_probs(daily_highs, buckets)

        # Median daily high from MC samples (degF)
        p50_high_f = float(np.median(daily_highs))

        return {
            "station":    station.station_id,
            "init_time":  init_time,
            "probs":      probs,
            "p50_high_f": p50_high_f,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_frames(
        hrrr_df: pd.DataFrame,
        iem_df: pd.DataFrame,
        init_time: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build 24-row encoder and 48-row decoder DataFrames.

        The encoder covers fxx -23..0 (hourly valid_time = init_time - 23h
        through init_time).  Obs values from iem_df are merged by valid_time.
        The decoder covers fxx 1..48 from hrrr_df.
        """
        # --- Encoder rows (fxx -23..0) ---------------------------------
        enc_rows: list[dict] = []
        for offset in range(-23, 1):
            vt = init_time + pd.Timedelta(hours=offset)
            row: dict = {
                "fxx":        offset,
                "init_time":  init_time,
                "valid_time": vt,
            }
            # Merge obs by valid_time (nearest minute tolerance)
            if not iem_df.empty:
                diffs = (iem_df["valid_time"] - vt).abs()
                best  = int(diffs.idxmin())
                if diffs.iloc[best] <= pd.Timedelta(minutes=30):
                    row.update(iem_df.iloc[best].to_dict())
            enc_rows.append(row)
        encoder_df = pd.DataFrame(enc_rows)

        # --- Decoder rows (fxx 1..48) ----------------------------------
        if not hrrr_df.empty:
            decoder_df = hrrr_df[hrrr_df["fxx"].between(1, 48)].copy()
        else:
            dec_rows = [
                {
                    "fxx":        fxx,
                    "init_time":  init_time,
                    "valid_time": init_time + pd.Timedelta(hours=fxx),
                }
                for fxx in range(1, 49)
            ]
            decoder_df = pd.DataFrame(dec_rows)

        return encoder_df, decoder_df
