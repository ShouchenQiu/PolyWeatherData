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

    @classmethod
    def from_settings(cls, settings: "Settings") -> "McSampler":
        return cls(
            n_samp   = settings.mc_n_samp,
            rng_seed = settings.mc_rng_seed,
        )

    # ------------------------------------------------------------------
    # Covariance fitting
    # ------------------------------------------------------------------

    def set_cov(self, cov_matrix: np.ndarray) -> None:
        """Load a pre-computed covariance matrix for production use.

        Parameters
        ----------
        cov_matrix:
            (n_hours, n_hours) covariance matrix loaded from disk (e.g. via
            ModelRegistry.load_covariance()).  Must be square.
        """
        if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(
                f"set_cov requires a square matrix, got shape {cov_matrix.shape}"
            )
        self._cov = cov_matrix

    def fit_cov(self, residuals: np.ndarray) -> None:
        """Estimate empirical covariance from historical forecast residuals.

        Parameters
        ----------
        residuals:
            (n_obs, n_hours) array of forecast error residuals from training.
            Rows are individual cycles; columns are decoder hours.
        """
        if residuals.size == 0 or residuals.shape[0] == 0:
            raise ValueError(
                "fit_cov requires a non-empty residuals array"
            )
        if residuals.ndim < 2 or residuals.shape[0] < residuals.shape[1]:
            raise ValueError(
                f"fit_cov requires at least as many rows as columns; "
                f"got shape {residuals.shape}"
            )
        self._cov = np.cov(residuals.T)  # (n_hours, n_hours)

    # ------------------------------------------------------------------
    # Core sampler
    # ------------------------------------------------------------------

    def _mc_sample(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
        hour_indices: "list[int] | None" = None,
    ) -> np.ndarray:
        """Draw correlated temperature samples.

        Uses the empirical covariance stored by fit_cov() / set_cov() when
        available.  When ``hour_indices`` is provided and the stored covariance
        is larger than (n, n), the appropriate submatrix is extracted so that
        partial-day target windows still benefit from empirical correlation
        (ADR-0002).  Falls back to a diagonal covariance when no stored
        covariance covers the requested indices.

        Parameters
        ----------
        mu_vec:
            (n_hours,) mean temperature in degC for each forecast hour.
        sigma_vec:
            (n_hours,) per-hour standard deviation in degC.  Must be positive.
        hour_indices:
            0-based positions into the full 48-hour covariance matrix.  When
            provided, the (n, n) submatrix is extracted instead of requiring
            the stored covariance to be exactly (n, n).

        Returns
        -------
        (n_samp, n_hours) array of temperature samples in degC.
        """
        if len(mu_vec) == 0:
            raise ValueError("_mc_sample: mu_vec and sigma_vec must not be empty")
        if len(mu_vec) != len(sigma_vec):
            raise ValueError(
                f"_mc_sample: mu_vec length {len(mu_vec)} != "
                f"sigma_vec length {len(sigma_vec)}"
            )
        if not np.all(np.isfinite(mu_vec)):
            raise ValueError("_mc_sample: mu_vec contains non-finite values")
        if not np.all(np.isfinite(sigma_vec)):
            raise ValueError("_mc_sample: sigma_vec contains non-finite values")
        if not np.all(sigma_vec > 0):
            raise ValueError("_mc_sample: sigma_vec must contain strictly positive values")
        n = len(mu_vec)

        if self._cov is not None:
            if hour_indices is not None:
                idx = list(hour_indices)
                Sigma = self._cov[np.ix_(idx, idx)].copy()
            elif self._cov.shape == (n, n):
                Sigma = self._cov.copy()
            else:
                Sigma = np.diag(sigma_vec ** 2)
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
        max_bias: float = 0.0,
        hour_indices: "list[int] | None" = None,
    ) -> np.ndarray:
        """Sample daily high temperatures (degF) with max_bias correction.

        Parameters
        ----------
        mu_vec:
            (n_hours,) post-processed TFT p50 in degC.
        sigma_vec:
            (n_hours,) per-hour sigma in degC.
        max_bias:
            Per-station bias correction (degF) subtracted from the daily
            high samples.  Load from ModelRegistry.load_max_bias() before
            each call.  Defaults to 0.0 (no correction).
        hour_indices:
            0-based positions into the full 48-hour covariance matrix.  Pass
            this when mu_vec/sigma_vec are derived from a target-date fxx
            subset so the empirical covariance submatrix is used instead of
            diagonal fallback (ADR-0002).

        Returns
        -------
        (n_samp,) daily high temperature samples in degF, bias-corrected.
        """
        sigma_safe   = np.clip(sigma_vec, 0.01, None)
        samples_c    = self._mc_sample(mu_vec, sigma_safe,
                                       hour_indices=hour_indices)
        daily_high_c = samples_c.max(axis=1)       # (n_samp,)
        daily_high_f = daily_high_c * 1.8 + 32.0  # -> degF
        return daily_high_f - float(max_bias)

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
        if len(daily_highs) == 0:
            raise ValueError("bucket_probs: daily_highs sample array must not be empty")
        # Check for malformed and overlapping buckets
        for lo, hi in buckets:
            if lo >= hi:
                raise ValueError(
                    f"bucket_probs: malformed bucket ({lo}, {hi}): lo must be < hi"
                )
        sorted_b = sorted(buckets, key=lambda t: t[0])
        for i in range(len(sorted_b) - 1):
            _, hi_a = sorted_b[i]
            lo_b, _ = sorted_b[i + 1]
            if hi_a > lo_b:
                raise ValueError(
                    f"bucket_probs: overlapping buckets detected "
                    f"(hi={hi_a} > next lo={lo_b})"
                )
        probs: dict = {}
        n = len(daily_highs)
        for lo, hi in buckets:
            mask = (daily_highs >= lo) & (daily_highs < hi)
            probs[(lo, hi)] = float(mask.sum()) / n
        probs["p10_f"] = float(np.percentile(daily_highs, 10))
        probs["p50_f"] = float(np.percentile(daily_highs, 50))
        probs["p90_f"] = float(np.percentile(daily_highs, 90))
        return probs

