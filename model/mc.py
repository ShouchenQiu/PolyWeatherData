"""Monte Carlo sampler for daily high temperature estimation.

Implements empirical-covariance sampling [ADR-0002] that converts per-hour
TFT quantile outputs into a distribution over the daily high temperature,
then maps that distribution onto Polymarket bucket probabilities.

The AR(1) covariance and bias_corr_c correction present in earlier versions
have been removed per ADR-0002.  Temporal correlation is now captured via a
48x48 empirical covariance matrix estimated from training residuals using
fit_cov().  Until fit_cov() is called the sampler falls back to a diagonal
covariance built from the per-hour sigma_vec supplied by the caller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings


class McSampler:
    """Empirical-covariance Monte Carlo sampler.

    Parameters
    ----------
    n_samp:
        Number of Monte Carlo draws per evaluation (default 10000).
    rng_seed:
        Seed for the random number generator (default 42).
    """

    def __init__(
        self,
        n_samp: int = 10000,
        rng_seed: int = 42,
    ) -> None:
        self.n_samp    = n_samp
        self.rng_seed  = rng_seed
        self._rng      = np.random.default_rng(rng_seed)
        self._cov: np.ndarray | None = None
        self._max_bias: float = 0.0

    @classmethod
    def from_settings(cls, settings: "Settings") -> "McSampler":
        return cls(
            n_samp   = settings.mc_n_samp,
            rng_seed = settings.mc_rng_seed,
        )

    def set_max_bias(self, max_bias: float) -> None:
        """Set the per-station max-temperature bias correction.

        Loaded from ModelRegistry.load_max_bias() at the pipeline assembly
        point before each station's MC run.  Defaults to 0.0 (no correction)
        until calibrated.
        """
        self._max_bias = float(max_bias)

    # ------------------------------------------------------------------
    # Covariance fitting
    # ------------------------------------------------------------------

    def fit_cov(self, residuals: np.ndarray) -> None:
        """Estimate empirical covariance from historical forecast residuals.

        Parameters
        ----------
        residuals:
            (n_obs, n_hours) array of forecast error residuals from training.
            Rows are individual cycles; columns are decoder hours.
        """
        self._cov = np.cov(residuals.T)  # (n_hours, n_hours)

    # ------------------------------------------------------------------
    # Core sampler
    # ------------------------------------------------------------------

    def _mc_sample(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
    ) -> np.ndarray:
        """Draw correlated temperature samples.

        Uses the empirical covariance stored by fit_cov() when available.
        Falls back to a diagonal covariance built from sigma_vec otherwise.

        Parameters
        ----------
        mu_vec:
            (n_hours,) mean temperature in degC for each forecast hour.
        sigma_vec:
            (n_hours,) per-hour standard deviation in degC (already clipped).

        Returns
        -------
        (n_samp, n_hours) array of temperature samples in degC.
        """
        n = len(mu_vec)
        if self._cov is not None and self._cov.shape == (n, n):
            Sigma = self._cov.copy()
        else:
            Sigma = np.diag(sigma_vec ** 2)

        Sigma += 1e-8 * np.eye(n)
        L = np.linalg.cholesky(Sigma)

        z       = self._rng.standard_normal((n, self.n_samp))
        samples = mu_vec[:, None] + L @ z  # (n, n_samp)
        return samples.T                   # (n_samp, n)

    # ------------------------------------------------------------------
    # Daily high estimation
    # ------------------------------------------------------------------

    def sample_daily_high(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
    ) -> np.ndarray:
        """Sample daily high temperatures (degF) with max_bias correction.

        Parameters
        ----------
        mu_vec:
            (n_hours,) post-processed TFT p50 in degC.
        sigma_vec:
            (n_hours,) per-hour sigma in degC.

        Returns
        -------
        (n_samp,) daily high temperature samples in degF, bias-corrected.
        """
        sigma_safe  = np.clip(sigma_vec, 0.01, None)
        samples_c   = self._mc_sample(mu_vec, sigma_safe)  # (n_samp, n_hours)
        daily_high_c = samples_c.max(axis=1)               # (n_samp,)
        daily_high_f = daily_high_c * 1.8 + 32.0          # -> degF
        return daily_high_f - self._max_bias

    # ------------------------------------------------------------------
    # Bucket probability
    # ------------------------------------------------------------------

    def bucket_probs(
        self,
        daily_highs: np.ndarray,
        buckets: list[tuple[float, float]],
    ) -> dict:
        """Compute per-bucket probabilities and overall sample quantiles.

        Parameters
        ----------
        daily_highs:
            (n_samp,) daily high temperature samples in degF.
        buckets:
            List of (lo_f, hi_f) tuples in degF.
            Use -inf / +inf for open-ended buckets.

        Returns
        -------
        dict with:
          - (lo_f, hi_f) tuple keys -> float probability (sums to ~1.0)
          - "p10_f" -> 10th percentile of daily_highs
          - "p50_f" -> 50th percentile of daily_highs
          - "p90_f" -> 90th percentile of daily_highs
        """
        probs: dict = {}
        n = len(daily_highs)
        for lo, hi in buckets:
            mask = (daily_highs >= lo) & (daily_highs < hi)
            probs[(lo, hi)] = float(mask.sum()) / n
        probs["p10_f"] = float(np.percentile(daily_highs, 10))
        probs["p50_f"] = float(np.percentile(daily_highs, 50))
        probs["p90_f"] = float(np.percentile(daily_highs, 90))
        return probs

