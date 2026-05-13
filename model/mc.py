"""Monte Carlo sampler for daily high temperature estimation.

Implements an AR(1) correlated sampling scheme that converts per-hour
TFT quantile outputs into a distribution over the daily high temperature,
then maps that distribution onto Polymarket bucket probabilities.

Reference: Herbie-main/mc_helpers.py  (ar1_mc_sample)
           Herbie-main/mc_stage2_clob.py  (bias correction, C->F, bucket_probs)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings


class McSampler:
    """AR(1) Monte Carlo sampler.

    Parameters
    ----------
    n_samp:
        Number of Monte Carlo draws per evaluation (default 5000).
    rng_seed:
        Seed for the random number generator (default 42).
    rho:
        AR(1) correlation coefficient across forecast hours (default 0.90).
    bias_corr_c:
        Celsius bias correction subtracted from mu before sampling.
        Derived from training residuals (default 1.5206 C = 2.737 F / 1.8).
    """

    def __init__(
        self,
        n_samp: int = 5000,
        rng_seed: int = 42,
        rho: float = 0.90,
        bias_corr_c: float = 1.5206,
    ) -> None:
        self.n_samp      = n_samp
        self.rng_seed    = rng_seed
        self.rho         = rho
        self.bias_corr_c = bias_corr_c
        self._rng        = np.random.default_rng(rng_seed)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "McSampler":
        return cls(
            n_samp   = settings.mc_n_samp,
            rng_seed = settings.mc_rng_seed,
        )

    # ------------------------------------------------------------------
    # Core sampler
    # ------------------------------------------------------------------

    def ar1_mc_sample(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
    ) -> np.ndarray:
        """Draw correlated temperature samples using an AR(1) covariance.

        Parameters
        ----------
        mu_vec:
            (n_hours,) mean temperature in degC for each forecast hour.
        sigma_vec:
            (n_hours,) per-hour standard deviation in degC.
            Typically computed as (p90 - p10) / (2 * 1.282).

        Returns
        -------
        (n_samp, n_hours) array of correlated temperature samples in degC.
        """
        n = len(mu_vec)
        # Build AR(1) correlation matrix  R[i,j] = rho^|i-j|
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        R = self.rho ** np.abs(ii - jj).astype(float)

        # Scale to covariance: Sigma = D @ R @ D
        D     = np.diag(sigma_vec)
        Sigma = D @ R @ D

        # Cholesky decomposition (add small jitter for numerical stability)
        L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))

        # Draw standard normal and transform
        z       = self._rng.standard_normal((n, self.n_samp))
        samples = mu_vec[:, None] + L @ z   # (n, n_samp)
        return samples.T                     # (n_samp, n)

    # ------------------------------------------------------------------
    # Daily high estimation
    # ------------------------------------------------------------------

    def sample_daily_high(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
    ) -> np.ndarray:
        """Sample daily high temperatures (Fahrenheit).

        Applies bias correction to mu_vec, draws AR(1) correlated samples,
        takes the per-sample maximum across hours, and converts to Fahrenheit.

        Parameters
        ----------
        mu_vec:
            (n_hours,) post-processed TFT p50 in degC.
        sigma_vec:
            (n_hours,) per-hour sigma in degC.

        Returns
        -------
        (n_samp,) daily high temperature samples in degF.
        """
        mu_corr = mu_vec - self.bias_corr_c
        sigma_safe = np.clip(sigma_vec, 0.01, None)
        samples_c = self.ar1_mc_sample(mu_corr, sigma_safe)  # (n_samp, n_hours)
        daily_high_c = samples_c.max(axis=1)                  # (n_samp,)
        return daily_high_c * 1.8 + 32.0                      # -> degF

    # ------------------------------------------------------------------
    # Bucket probability
    # ------------------------------------------------------------------

    def bucket_probs(
        self,
        daily_highs: np.ndarray,
        buckets: list[tuple[float, float]],
    ) -> dict[tuple[float, float], float]:
        """Compute probability for each temperature bucket.

        Parameters
        ----------
        daily_highs:
            (n_samp,) daily high temperature samples in degF.
        buckets:
            List of (lo_f, hi_f) tuples in degF.
            Use -inf / +inf for open-ended buckets.

        Returns
        -------
        {(lo_f, hi_f): probability}  probabilities sum to ~1.0 over all
        buckets that collectively cover the full range.
        """
        probs: dict[tuple[float, float], float] = {}
        n = len(daily_highs)
        for lo, hi in buckets:
            mask = (daily_highs >= lo) & (daily_highs < hi)
            probs[(lo, hi)] = float(mask.sum()) / n
        return probs
