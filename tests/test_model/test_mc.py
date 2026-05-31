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
    """max_bias parameter shifts the distribution by exactly that amount."""
    s1 = McSampler(n_samp=2000, rng_seed=3)
    s2 = McSampler(n_samp=2000, rng_seed=3)
    mu    = np.full(48, 20.0)
    sigma = np.ones(48) * 1.0
    out1 = s1.sample_daily_high(mu, sigma, max_bias=0.0)
    out2 = s2.sample_daily_high(mu, sigma, max_bias=2.0)
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
    s = McSampler(n_samp=5000, rng_seed=1)

    mu    = np.full(48, 20.0)   # ~68 F
    sigma = np.ones(48) * 1.0
    buckets = [(float("-inf"), 70.0), (70.0, float("inf"))]

    raw_no   = s.bucket_probs(s.sample_daily_high(mu, sigma, max_bias=0.0), buckets)
    raw_with = s.bucket_probs(s.sample_daily_high(mu, sigma, max_bias=5.0), buckets)

    prob_no   = raw_no[(float("-inf"), 70.0)]
    prob_with = raw_with[(float("-inf"), 70.0)]
    assert prob_with > prob_no, "max_bias should shift more samples below 70 F"


# -----------------------------------------------------------------------
# V2V06 -- set_cov, fit_cov validation, _mc_sample validation, bucket_probs guards
# -----------------------------------------------------------------------

def test_set_cov_stores_matrix():
    """set_cov() stores the matrix so subsequent _mc_sample uses it."""
    s = McSampler(n_samp=200, rng_seed=0)
    cov = np.eye(6) * 4.0
    s.set_cov(cov)
    assert s._cov is not None
    np.testing.assert_array_equal(s._cov, cov)


def test_set_cov_rejects_non_square():
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="square"):
        s.set_cov(np.ones((4, 5)))


def test_fit_cov_raises_on_empty_residuals():
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="empty"):
        s.fit_cov(np.empty((0, 48)))


def test_fit_cov_raises_when_rows_lt_cols():
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="rows"):
        s.fit_cov(np.ones((3, 48)))


def test_mc_sample_raises_on_empty_mu():
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="empty"):
        s._mc_sample(np.array([]), np.array([]))


def test_mc_sample_raises_on_non_finite_mu():
    s = McSampler(n_samp=100, rng_seed=0)
    mu = np.array([1.0, float("nan"), 3.0])
    sigma = np.ones(3)
    with pytest.raises(ValueError, match="finite"):
        s._mc_sample(mu, sigma)


def test_mc_sample_raises_on_non_finite_sigma():
    s = McSampler(n_samp=100, rng_seed=0)
    mu = np.ones(4)
    sigma = np.array([1.0, 1.0, float("inf"), 1.0])
    with pytest.raises(ValueError, match="finite"):
        s._mc_sample(mu, sigma)


def test_bucket_probs_raises_on_empty_samples():
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="empty"):
        s.bucket_probs(np.array([]), [(60.0, 80.0)])


def test_bucket_probs_raises_on_overlapping_buckets():
    s = McSampler(n_samp=100, rng_seed=0)
    daily = np.full(100, 70.0)
    with pytest.raises(ValueError, match="overlap"):
        s.bucket_probs(daily, [(60.0, 75.0), (70.0, 85.0)])


# -----------------------------------------------------------------------
# V2V06 post-review fixes
# -----------------------------------------------------------------------

# -- Fix 4: McSampler validation gaps --

def test_mc_sample_raises_on_mismatched_lengths():
    """_mc_sample raises ValueError when mu_vec and sigma_vec have different lengths."""
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="length"):
        s._mc_sample(np.ones(5), np.ones(6))


def test_mc_sample_raises_on_non_positive_sigma():
    """_mc_sample raises ValueError when any sigma value is <= 0."""
    s = McSampler(n_samp=100, rng_seed=0)
    with pytest.raises(ValueError, match="positive"):
        s._mc_sample(np.ones(4), np.array([1.0, 1.0, 0.0, 1.0]))


def test_bucket_probs_raises_on_malformed_bucket():
    """bucket_probs raises ValueError when lo >= hi."""
    s = McSampler(n_samp=100, rng_seed=0)
    daily = np.full(100, 70.0)
    with pytest.raises(ValueError, match="malformed"):
        s.bucket_probs(daily, [(80.0, 70.0)])


def test_bucket_probs_raises_on_equal_bounds():
    """bucket_probs raises ValueError when lo == hi (zero-width bucket)."""
    s = McSampler(n_samp=100, rng_seed=0)
    daily = np.full(100, 70.0)
    with pytest.raises(ValueError, match="malformed"):
        s.bucket_probs(daily, [(70.0, 70.0)])


# -- Fix 2: Covariance submatrix selection --

def test_mc_sample_uses_submatrix_when_hour_indices_provided():
    """When hour_indices provided and _cov is full, the submatrix is extracted."""
    s = McSampler(n_samp=200, rng_seed=0)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((48, 48))
    s.set_cov(A @ A.T + np.eye(48) * 2.0)

    indices = list(range(12, 28))  # 16 hours (0-indexed, fxx 13..28)
    mu = np.zeros(16)
    sigma = np.ones(16)
    out = s._mc_sample(mu, sigma, hour_indices=indices)
    assert out.shape == (200, 16)


def test_mc_sample_submatrix_differs_from_diagonal():
    """Samples drawn from submatrix differ from diagonal-only samples (statistical)."""
    rng_np = np.random.default_rng(0)
    A = rng_np.standard_normal((48, 48))
    full_cov = A @ A.T + np.eye(48) * 10.0

    s_sub  = McSampler(n_samp=2000, rng_seed=7)
    s_diag = McSampler(n_samp=2000, rng_seed=7)
    s_sub.set_cov(full_cov)

    indices = list(range(0, 16))
    mu = np.zeros(16)
    sigma = np.sqrt(np.diag(full_cov[:16, :16]))

    out_sub  = s_sub._mc_sample(mu, sigma, hour_indices=indices)
    out_diag = s_diag._mc_sample(mu, sigma)  # no _cov, uses diagonal

    # Covariance structures should differ: off-diagonal elements will be non-zero
    # in out_sub. The sample covariance matrices will differ.
    cov_sub  = np.cov(out_sub.T)
    cov_diag = np.cov(out_diag.T)
    assert not np.allclose(cov_sub, cov_diag, atol=0.5)


def test_sample_daily_high_accepts_hour_indices():
    """sample_daily_high accepts and passes through hour_indices without error."""
    s = McSampler(n_samp=100, rng_seed=0)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((48, 48))
    s.set_cov(A @ A.T + np.eye(48))

    indices = list(range(12, 28))
    mu = np.zeros(16)
    sigma = np.ones(16)
    out = s.sample_daily_high(mu, sigma, hour_indices=indices)
    assert out.shape == (100,)


# -----------------------------------------------------------------------
# hour_indices defensive validation (Batch 4 review)
# -----------------------------------------------------------------------

def test_mc_sample_raises_when_hour_indices_length_mismatches_mu():
    """_mc_sample raises ValueError when hour_indices length != len(mu_vec)."""
    s = McSampler(n_samp=100, rng_seed=0)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((48, 48))
    s.set_cov(A @ A.T + np.eye(48))
    mu = np.zeros(4)
    sigma = np.ones(4)
    with pytest.raises(ValueError, match="hour_indices"):
        s._mc_sample(mu, sigma, hour_indices=[0, 1, 2])   # length 3 != 4


def test_mc_sample_raises_when_hour_indices_has_negative_index():
    """_mc_sample raises ValueError when any hour_index is negative."""
    s = McSampler(n_samp=100, rng_seed=0)
    mu = np.zeros(3)
    sigma = np.ones(3)
    with pytest.raises(ValueError, match="hour_indices"):
        s._mc_sample(mu, sigma, hour_indices=[0, 1, -1])


def test_mc_sample_raises_when_hour_indices_has_duplicates():
    """_mc_sample raises ValueError when hour_indices contains duplicate entries."""
    s = McSampler(n_samp=100, rng_seed=0)
    mu = np.zeros(3)
    sigma = np.ones(3)
    with pytest.raises(ValueError, match="hour_indices"):
        s._mc_sample(mu, sigma, hour_indices=[0, 1, 1])

