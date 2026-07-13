from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.conditional_residual import (
    ConditionalPathSampler,
    ConditionalResidualPathLibrary,
    InsufficientConditionalPathSupport,
)


def _library() -> ConditionalResidualPathLibrary:
    paths = pd.DataFrame(
        {
            "path_id": ["cold", "warm"],
            "init_time": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "target_date": ["2026-01-02", "2026-01-03"],
            "state_peak_f": [40.0, 80.0],
            "state_spread_f": [4.0, 4.0],
            **{
                f"residual_{hour:02d}": [float(hour), -float(hour)]
                for hour in range(1, 49)
            },
        }
    )
    return ConditionalResidualPathLibrary.fit(paths, feature_columns=("state_peak_f", "state_spread_f"))


def test_library_rejects_incomplete_or_extrapolated_path() -> None:
    frame = pd.DataFrame({
        "path_id": ["bad"], "init_time": pd.to_datetime(["2026-01-01"], utc=True),
        "target_date": ["2026-01-02"], "state_peak_f": [50.0], "state_spread_f": [3.0],
        **{f"residual_{hour:02d}": [0.0] for hour in range(1, 48)},
    })

    with pytest.raises(ValueError, match="48 residual"):
        ConditionalResidualPathLibrary.fit(frame, feature_columns=("state_peak_f", "state_spread_f"))


def test_sampler_draws_one_coherent_path_and_is_reproducible() -> None:
    library = _library()
    sampler_a = ConditionalPathSampler(library, n_samp=16, rng_seed=7, neighbors=2, min_effective_samples=1)
    sampler_b = ConditionalPathSampler(library, n_samp=16, rng_seed=7, neighbors=2, min_effective_samples=1)
    kwargs = dict(
        mu_vec=np.zeros(3), sigma_vec=np.ones(3), hour_indices=[0, 1, 2],
        state={"state_peak_f": 40.0, "state_spread_f": 4.0}, max_bias=0.0,
    )

    samples_a = sampler_a.sample_daily_high(**kwargs)
    samples_b = sampler_b.sample_daily_high(**kwargs)

    np.testing.assert_allclose(samples_a, samples_b)
    # The first three residuals always have one path's signed 1:2:3 shape.
    residual_c = (samples_a / 1.8) - (32.0 / 1.8)
    assert set(np.round(residual_c, 6)).issubset({3.0, -1.0})


def test_sampler_rejects_low_effective_support() -> None:
    library = _library()
    sampler = ConditionalPathSampler(library, n_samp=10, rng_seed=1, neighbors=2, min_effective_samples=3)

    with pytest.raises(InsufficientConditionalPathSupport, match="ESS"):
        sampler.sample_daily_high(
            np.zeros(2), np.ones(2), [0, 1],
            state={"state_peak_f": 40.0, "state_spread_f": 4.0},
        )


def test_sampler_marks_far_state_as_ood_without_using_outcomes() -> None:
    sampler = ConditionalPathSampler(_library(), n_samp=4, rng_seed=1, neighbors=2, min_effective_samples=1)

    sampler.sample_daily_high(
        np.zeros(2), np.ones(2), [0, 1],
        state={"state_peak_f": 500.0, "state_spread_f": 4.0},
    )

    assert sampler.last_diagnostics is not None
    assert sampler.last_diagnostics.is_ood is True


def test_library_load_rejects_lineage_mismatch(tmp_path) -> None:
    library = _library()
    library.save(tmp_path / "library", lineage={"crossfit_manifest": "a" * 64})

    loaded = ConditionalResidualPathLibrary.load(
        tmp_path / "library", expected_lineage={"crossfit_manifest": "a" * 64}
    )
    assert loaded.manifest_sha256 == library.manifest_sha256
    with pytest.raises(ValueError, match="lineage mismatch"):
        ConditionalResidualPathLibrary.load(
            tmp_path / "library", expected_lineage={"crossfit_manifest": "b" * 64}
        )
