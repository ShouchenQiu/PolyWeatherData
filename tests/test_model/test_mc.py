"""Tests for model/mc.py -- McSampler (empirical-covariance, ADR-0002)."""
import math

import numpy as np
import pytest

from model.mc import McSampler


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sampler():
    return McSampler(n_samp=5000, rng_seed=42)


@pytest.fixture
def small_sampler():
    """Smaller sampler for shape-only checks (faster)."""
    return McSampler(n_samp=200, rng_seed=0)


# -----------------------------------------------------------------------
# Constructor / defaults
# -----------------------------------------------------------------------

def test_default_n_samp():
    s = McSampler()
    assert s.n_samp == 10000


def test_ar1_and_bias_corr_absent():
    """AR(1) and bias_corr_c must not exist on the instance."""
    s = McSampler()
    assert not hasattr(s, "rho")
    assert not hasattr(s, "bias_corr_c")
    assert not hasattr(s, "ar1_mc_sample")


# -----------------------------------------------------------------------
# fit_cov
# -----------------------------------------------------------------------

def test_fit_cov_stores_covariance():
    s = McSampler(n_samp=100, rng_seed=0)
    rng = np.random.default_rng(1)
    residuals = rng.standard_normal((200, 48))
    s.fit_cov(residuals)
    assert s._cov is not None
    assert s._cov.shape == (48, 48)


def test_fit_cov_known_values():
    """fit_cov on a 2-hour residual set matches np.cov directly."""
    s = McSampler(n_samp=50, rng_seed=0)
    residuals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    s.fit_cov(residuals)
    expected = np.cov(residuals.T)
    np.testing.assert_allclose(s._cov, expected)


# -----------------------------------------------------------------------
# _mc_sample -- shape and statistical properties
# -----------------------------------------------------------------------

def test_mc_sample_shape_no_cov(small_sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float)
    out   = small_sampler._mc_sample(mu, sigma)
    assert out.shape == (200, 48)


def test_mc_sample_shape_with_cov():
    s   = McSampler(n_samp=300, rng_seed=5)
    rng = np.random.default_rng(9)
    s.fit_cov(rng.standard_normal((100, 12)))
    mu    = np.zeros(12)
    sigma = np.ones(12)
    out   = s._mc_sample(mu, sigma)
    assert out.shape == (300, 12)


def test_mc_sample_mean_approx(sampler):
    """Sample mean close to mu_vec with 5000 draws (diagonal fallback)."""
    n     = 24
    mu    = np.arange(n, dtype=float)
    sigma = np.ones(n, dtype=float) * 2.0
    out   = sampler._mc_sample(mu, sigma)
    max_err = np.abs(out.mean(axis=0) - mu).max()
    assert max_err < 0.5, f"max mean error {max_err:.3f}"


def test_mc_sample_no_nan(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float) * 1.5
    out   = sampler._mc_sample(mu, sigma)
    assert not np.any(np.isnan(out))


def test_mc_sample_rng_reproducible():
    s1 = McSampler(n_samp=100, rng_seed=7)
    s2 = McSampler(n_samp=100, rng_seed=7)
    mu    = np.array([10.0, 12.0, 14.0])
    sigma = np.array([1.0,  1.0,  1.0])
    np.testing.assert_array_equal(
        s1._mc_sample(mu, sigma),
        s2._mc_sample(mu, sigma),
    )


# -----------------------------------------------------------------------
# sample_daily_high
# -----------------------------------------------------------------------

def test_sample_daily_high_shape(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float)
    out   = sampler.sample_daily_high(mu, sigma)
    assert out.shape == (5000,)


def test_sample_daily_high_fahrenheit_range(sampler):
    """For mu_vec = 25 C (77 F) daily high should cluster near 77 F."""
    mu    = np.full(48, 25.0, dtype=float)
    sigma = np.ones(48, dtype=float) * 1.0
    out   = sampler.sample_daily_high(mu, sigma)
    assert 60.0 < float(np.median(out)) < 85.0, f"Unexpected median {np.median(out):.1f} F"


def test_sample_daily_high_no_nan(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float)
    out   = sampler.sample_daily_high(mu, sigma)
    assert not np.any(np.isnan(out))


def test_sample_daily_high_no_bias_by_default():
    """With no bias_corr_c and tight sigma, median daily high ~ max(mu)*1.8+32."""
    s   = McSampler(n_samp=5000, rng_seed=42)
    mu  = np.full(12, 20.0)    # 20 C -> 68 F
    sig = np.full(12, 0.001)   # very tight
    out = s.sample_daily_high(mu, sig)
    assert 67.0 < float(np.median(out)) < 69.0


def test_sample_daily_high_max_bias_shifts_output():
    """Setting _max_bias shifts the distribution by exactly that amount."""
    s1 = McSampler(n_samp=2000, rng_seed=3)
    s2 = McSampler(n_samp=2000, rng_seed=3)
    s2._max_bias = 2.0
    mu    = np.full(48, 20.0)
    sigma = np.ones(48) * 1.0
    out1 = s1.sample_daily_high(mu, sigma)
    out2 = s2.sample_daily_high(mu, sigma)
    np.testing.assert_allclose(out1 - 2.0, out2)


# -----------------------------------------------------------------------
# bucket_probs
# -----------------------------------------------------------------------

def test_bucket_probs_sum_to_one(sampler):
    """Per-bucket probs over exhaustive buckets must sum to ~1."""
    mu    = np.full(48, 25.0)
    sigma = np.ones(48)
    daily = sampler.sample_daily_high(mu, sigma)
    buckets = [
        (float("-inf"), 50.0),
        (50.0, 60.0),
        (60.0, 70.0),
        (70.0, 80.0),
        (80.0, 90.0),
        (90.0, float("inf")),
    ]
    raw   = sampler.bucket_probs(daily, buckets)
    probs = {k: v for k, v in raw.items() if isinstance(k, tuple)}
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-9, f"probs sum = {total}"


def test_bucket_probs_returns_quantiles(sampler):
    """bucket_probs must include p10_f, p50_f, p90_f keys."""
    daily   = np.full(100, 70.0, dtype=float)
    buckets = [(60.0, 80.0)]
    raw     = sampler.bucket_probs(daily, buckets)
    assert "p10_f" in raw
    assert "p50_f" in raw
    assert "p90_f" in raw


def test_bucket_probs_quantile_values():
    """Quantile values match np.percentile on known data."""
    s = McSampler(n_samp=100, rng_seed=0)
    daily = np.arange(100, dtype=float)  # 0..99 F
    raw   = s.bucket_probs(daily, [(0.0, float("inf"))])
    assert math.isclose(raw["p10_f"], 9.9,  abs_tol=0.01)
    assert math.isclose(raw["p50_f"], 49.5, abs_tol=0.01)
    assert math.isclose(raw["p90_f"], 89.1, abs_tol=0.01)


def test_bucket_probs_keys_match_buckets(sampler):
    daily   = np.full(100, 70.0)
    buckets = [(60.0, 70.0), (70.0, 80.0)]
    raw     = sampler.bucket_probs(daily, buckets)
    tuple_keys = {k for k in raw if isinstance(k, tuple)}
    assert tuple_keys == {(60.0, 70.0), (70.0, 80.0)}


def test_bucket_probs_all_in_one_bucket():
    s       = McSampler(n_samp=100, rng_seed=0)
    daily   = np.full(100, 75.0)
    buckets = [(float("-inf"), 70.0), (70.0, 80.0), (80.0, float("inf"))]
    raw     = s.bucket_probs(daily, buckets)
    assert raw[(70.0, 80.0)] == 1.0
    assert raw[(float("-inf"), 70.0)] == 0.0
    assert raw[(80.0, float("inf"))] == 0.0


def test_bucket_probs_max_bias_shifts_probabilities():
    """max_bias correction moves samples; bucket assignments shift accordingly."""
    s_no_bias   = McSampler(n_samp=5000, rng_seed=1)
    s_with_bias = McSampler(n_samp=5000, rng_seed=1)
    s_with_bias._max_bias = 5.0

    mu    = np.full(48, 20.0)   # ~68 F
    sigma = np.ones(48) * 1.0
    buckets = [(float("-inf"), 70.0), (70.0, float("inf"))]

    raw_no   = s_no_bias.bucket_probs(s_no_bias.sample_daily_high(mu, sigma), buckets)
    raw_with = s_with_bias.bucket_probs(s_with_bias.sample_daily_high(mu, sigma), buckets)

    prob_no   = raw_no[(float("-inf"), 70.0)]
    prob_with = raw_with[(float("-inf"), 70.0)]
    assert prob_with > prob_no, "max_bias should shift more samples below 70 F"

