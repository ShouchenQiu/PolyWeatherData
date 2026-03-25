"""Tests for model/mc.py -- McSampler."""
import math

import numpy as np
import pytest

from model.mc import McSampler


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sampler():
    return McSampler(n_samp=5000, rng_seed=42, rho=0.90, bias_corr_c=1.5206)


@pytest.fixture
def small_sampler():
    """Smaller sampler for shape-only checks (faster)."""
    return McSampler(n_samp=200, rng_seed=0, rho=0.90, bias_corr_c=0.0)


# -----------------------------------------------------------------------
# ar1_mc_sample -- output shape
# -----------------------------------------------------------------------

def test_ar1_sample_shape(sampler):
    n_hours = 48
    mu      = np.zeros(n_hours, dtype=float)
    sigma   = np.ones(n_hours, dtype=float)
    out = sampler.ar1_mc_sample(mu, sigma)
    assert out.shape == (5000, 48)


def test_ar1_sample_shape_small(small_sampler):
    mu    = np.zeros(6, dtype=float)
    sigma = np.ones(6, dtype=float)
    out = small_sampler.ar1_mc_sample(mu, sigma)
    assert out.shape == (200, 6)


# -----------------------------------------------------------------------
# ar1_mc_sample -- statistical properties
# -----------------------------------------------------------------------

def test_ar1_sample_mean_approx(sampler):
    """Sample mean should be close to mu_vec with 5000 draws."""
    n      = 24
    mu     = np.arange(n, dtype=float)      # 0..23
    sigma  = np.ones(n, dtype=float) * 2.0
    out    = sampler.ar1_mc_sample(mu, sigma)
    mean   = out.mean(axis=0)
    max_err = np.abs(mean - mu).max()
    assert max_err < 0.5, f"max mean error {max_err:.3f} >= 0.5"


def test_ar1_sample_std_approx(sampler):
    """Sample std should be within 15% of requested sigma."""
    n      = 12
    mu     = np.zeros(n, dtype=float)
    sigma  = np.full(n, 3.0, dtype=float)
    out    = sampler.ar1_mc_sample(mu, sigma)
    std    = out.std(axis=0)
    rel_err = np.abs(std - sigma) / sigma
    assert rel_err.max() < 0.15, f"max rel std error {rel_err.max():.3f}"


def test_ar1_sample_no_nan(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float) * 1.5
    out   = sampler.ar1_mc_sample(mu, sigma)
    assert not np.any(np.isnan(out))


def test_ar1_sample_rng_reproducible():
    """Two samplers with same seed produce identical results."""
    s1 = McSampler(n_samp=100, rng_seed=7, rho=0.9, bias_corr_c=0.0)
    s2 = McSampler(n_samp=100, rng_seed=7, rho=0.9, bias_corr_c=0.0)
    mu    = np.array([10.0, 12.0, 14.0])
    sigma = np.array([1.0,  1.0,  1.0])
    out1 = s1.ar1_mc_sample(mu, sigma)
    out2 = s2.ar1_mc_sample(mu, sigma)
    np.testing.assert_array_equal(out1, out2)


# -----------------------------------------------------------------------
# sample_daily_high
# -----------------------------------------------------------------------

def test_sample_daily_high_shape(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float)
    out   = sampler.sample_daily_high(mu, sigma)
    assert out.shape == (5000,)


def test_sample_daily_high_fahrenheit_range(sampler):
    """For mu_vec = 25 C (77 F) the daily high should cluster near 77 F."""
    mu    = np.full(48, 25.0, dtype=float)
    sigma = np.ones(48, dtype=float) * 1.0
    out   = sampler.sample_daily_high(mu, sigma)
    median_f = float(np.median(out))
    # bias_corr_c = 1.5206 shifts mu down, so median < 77
    assert 60.0 < median_f < 80.0, f"Unexpected median {median_f:.1f} F"


def test_sample_daily_high_no_nan(sampler):
    mu    = np.zeros(48, dtype=float)
    sigma = np.ones(48, dtype=float)
    out   = sampler.sample_daily_high(mu, sigma)
    assert not np.any(np.isnan(out))


def test_sample_daily_high_bias_correction_effect():
    """With bias_corr_c=0, median daily high ~ max(mu)*1.8+32."""
    s    = McSampler(n_samp=5000, rng_seed=42, rho=0.0, bias_corr_c=0.0)
    mu   = np.full(12, 20.0)   # 20 C -> 68 F
    sig  = np.full(12, 0.001)  # very tight
    out  = s.sample_daily_high(mu, sig)
    # With near-zero sigma, max(mu) ~ 20 C -> 68 F
    assert 67.0 < float(np.median(out)) < 69.0


# -----------------------------------------------------------------------
# bucket_probs
# -----------------------------------------------------------------------

def test_bucket_probs_sum_to_one(sampler):
    """Probs over exhaustive buckets should sum to ~1."""
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
    probs = sampler.bucket_probs(daily, buckets)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-9, f"probs sum = {total}"


def test_bucket_probs_keys_match_buckets(sampler):
    daily   = np.full(100, 70.0)
    buckets = [(60.0, 70.0), (70.0, 80.0)]
    probs   = sampler.bucket_probs(daily, buckets)
    assert set(probs.keys()) == {(60.0, 70.0), (70.0, 80.0)}


def test_bucket_probs_all_in_one_bucket():
    """If all samples fall in one bucket its prob should be 1."""
    s       = McSampler(n_samp=100, rng_seed=0)
    daily   = np.full(100, 75.0)   # all exactly 75 F
    buckets = [(float("-inf"), 70.0), (70.0, 80.0), (80.0, float("inf"))]
    probs   = s.bucket_probs(daily, buckets)
    assert probs[(70.0, 80.0)] == 1.0
    assert probs[(float("-inf"), 70.0)] == 0.0
    assert probs[(80.0, float("inf"))] == 0.0


def test_bucket_probs_open_low_end():
    s       = McSampler(n_samp=50, rng_seed=1)
    daily   = np.array([50.0] * 25 + [90.0] * 25, dtype=float)
    buckets = [(float("-inf"), 70.0), (70.0, float("inf"))]
    probs   = s.bucket_probs(daily, buckets)
    assert math.isclose(probs[(float("-inf"), 70.0)], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[(70.0, float("inf"))],  0.5, abs_tol=1e-9)
