from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from backtest.signal_slice_study import (
    build_contiguous_event_day_folds,
    build_signal_slice_features,
    evaluate_fixed_slice_atlas,
    load_signal_slice_artifact,
    run_nested_slice_search,
    save_signal_slice_artifact,
)


def _panel() -> pd.DataFrame:
    rows = []
    for order, (model, midpoint, outcome) in enumerate(
        [(0.6, 0.4, 1.0), (0.3, 0.4, 0.0), (0.1, 0.2, 0.0)]
    ):
        rows.append(
            {
                "event_id": "event",
                "target_date": "2026-04-01",
                "timestamp": pd.Timestamp("2026-04-01T12:00:00Z"),
                "snapshot_id": "snapshot",
                "init_time": pd.Timestamp("2026-04-01T10:00:00Z"),
                "available_at": pd.Timestamp("2026-04-01T11:00:00Z"),
                "condition_id": f"condition-{order}",
                "bucket_order": order,
                "model_probability": model,
                "midpoint_probability": midpoint,
                "midpoint_overround": 0.05,
                "yes_outcome": outcome,
                "lead_hours": 16.0,
            }
        )
    return pd.DataFrame(rows)


def test_slice_features_measure_vector_and_choose_one_model_favored_token() -> None:
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot",
                "coverage_type": "forecast_only",
                "observed_hour_count": 0,
                "forecast_hour_count": 24,
            }
        ]
    )

    vectors, tokens = build_signal_slice_features(_panel(), snapshots)

    assert len(vectors) == 1
    vector = vectors.iloc[0]
    expected_entropy = -sum(
        probability * np.log(probability)
        for probability in (0.6, 0.3, 0.1)
    ) / np.log(3)
    assert vector["model_max_probability"] == pytest.approx(0.6)
    assert vector["model_normalized_entropy"] == pytest.approx(
        expected_entropy
    )
    assert vector["model_top_two_margin"] == pytest.approx(0.3)
    assert vector["midpoint_top_two_margin"] == pytest.approx(0.0)
    assert vector["tv_distance"] == pytest.approx(0.2)
    assert bool(vector["top_bucket_agrees"])
    assert vector["max_absolute_probability_gap"] == pytest.approx(0.2)
    assert vector["snapshot_age_hours"] == pytest.approx(1.0)
    assert vector["init_age_hours"] == pytest.approx(2.0)
    assert vector["coverage_type"] == "forecast_only"

    assert len(tokens) == 1
    token = tokens.iloc[0]
    assert token["condition_id"] == "condition-0"
    assert token["token_side"] == "YES"
    assert token["model_probability"] == pytest.approx(0.6)
    assert token["benchmark_probability"] == pytest.approx(0.4)
    assert token["token_outcome"] == 1.0
    assert token["model_probability_advantage"] == pytest.approx(0.2)
    assert token["theoretical_midpoint_excess"] == pytest.approx(0.6)


def test_fixed_atlas_keeps_empty_bands_and_uses_first_signal_per_event() -> None:
    panel = pd.concat(
        [
            _panel(),
            _panel().assign(
                timestamp=pd.Timestamp("2026-04-01T12:05:00Z"),
                lead_hours=15.0,
            ),
        ],
        ignore_index=True,
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot",
                "coverage_type": "forecast_only",
                "observed_hour_count": 0,
                "forecast_hour_count": 24,
            }
        ]
    )
    vectors, tokens = build_signal_slice_features(panel, snapshots)

    atlas = evaluate_fixed_slice_atlas(
        vectors,
        tokens,
        bootstrap_samples=50,
        seed=42,
    )
    repeated = evaluate_fixed_slice_atlas(
        vectors,
        tokens,
        bootstrap_samples=50,
        seed=42,
    )

    empty = atlas.set_index("slice_id").loc["lead_band=<6h"]
    assert empty["event_count"] == 0
    assert empty["vector_count"] == 0
    active = atlas.set_index("slice_id").loc["lead_band=12-24h"]
    assert active["event_count"] == 1
    assert active["vector_count"] == 2
    assert active["first_signal_count"] == 1
    assert active["model_multiclass_brier"] == pytest.approx(0.26)
    assert active["midpoint_multiclass_brier"] == pytest.approx(0.56)
    assert active["theoretical_midpoint_excess"] == pytest.approx(0.6)
    pd.testing.assert_series_equal(
        atlas["brier_max_t_adjusted_p"],
        repeated["brier_max_t_adjusted_p"],
    )


def test_token_ties_use_condition_id_then_yes_before_no() -> None:
    panel = _panel().assign(
        model_probability=[0.4, 0.4, 0.2],
        midpoint_probability=[0.4, 0.4, 0.2],
        condition_id=["z-condition", "a-condition", "m-condition"],
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot",
                "coverage_type": "forecast_only",
                "observed_hour_count": 0,
                "forecast_hour_count": 24,
            }
        ]
    )

    _, tokens = build_signal_slice_features(panel, snapshots)

    assert tokens.iloc[0]["condition_id"] == "a-condition"
    assert tokens.iloc[0]["token_side"] == "YES"


def test_contiguous_folds_hold_out_whole_ordered_event_days() -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D").date.astype(str)

    folds = build_contiguous_event_day_folds(dates, folds=5)

    assert len(folds) == 5
    all_dates = set(dates)
    seen: set[str] = set()
    for row in folds.itertuples(index=False):
        held_out = list(row.held_out_dates)
        assert held_out == sorted(held_out)
        assert set(held_out).isdisjoint(row.fit_dates)
        assert set(row.fit_dates) | set(held_out) == all_dates
        seen.update(held_out)
    assert seen == all_dates


def test_nested_search_fits_thresholds_without_held_out_days() -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D").date.astype(str)
    vectors = pd.DataFrame(
        [
            {
                "event_id": f"event-{index}",
                "target_date": target_date,
                "timestamp": pd.Timestamp(str(target_date), tz="UTC"),
                "snapshot_id": f"snapshot-{index}",
                "lead_hours": 1_000.0 if index == 0 else float(index),
                "coverage_type": "forecast_only",
                "model_max_probability": 0.4 + index / 100,
                "model_normalized_entropy": 0.6,
                "tv_distance": 0.2,
                "top_bucket_agrees": False,
                "model_multiclass_brier": 0.2,
                "midpoint_multiclass_brier": 0.4,
                "brier_delta": -0.2,
                "model_log_score": 0.5,
                "midpoint_log_score": 0.8,
                "log_score_delta": -0.3,
                "model_normalized_rps": 0.05,
                "midpoint_normalized_rps": 0.10,
                "rps_delta": -0.05,
            }
            for index, target_date in enumerate(dates)
        ]
    )
    tokens = pd.DataFrame(
        [
            {
                "event_id": f"event-{index}",
                "target_date": target_date,
                "timestamp": pd.Timestamp(str(target_date), tz="UTC"),
                "snapshot_id": f"snapshot-{index}",
                "condition_id": "condition",
                "token_side": "YES",
                "bucket_position": "center",
                "benchmark_probability": 0.3,
                "theoretical_midpoint_excess": 0.2,
                "model_probability_residual": 0.0,
                "model_probability_advantage": 0.1,
            }
            for index, target_date in enumerate(dates)
        ]
    )

    crossfit, fold_rules, shortlist = run_nested_slice_search(
        vectors,
        tokens,
        folds=5,
        min_training_events=4,
        minimum_shortlist_events=3,
        bootstrap_samples=50,
        seed=42,
    )

    fold_zero = fold_rules[fold_rules["fold"] == 0]
    thresholds = [
        json.loads(value)["lead_hours"]
        for value in fold_zero["thresholds_json"].unique()
    ]
    assert thresholds
    assert all(max(cuts) < 1_000 for cuts in thresholds)
    assert set(crossfit["fold"]) == set(range(5))
    assert not shortlist.empty
    assert set(shortlist["evidence_tier"]).issubset(
        {"stable_candidate", "promising_candidate", "exploratory_only"}
    )


def test_slice_artifact_is_complete_and_hash_verified(tmp_path) -> None:
    frame = pd.DataFrame([{"value": 1}])
    output = tmp_path / "slices"
    summary = {
        "parameters": {
            "folds": 5,
            "bootstrap_samples": 100,
            "seed": 42,
            "min_training_events": 12,
            "minimum_shortlist_events": 15,
        }
    }

    manifest = save_signal_slice_artifact(
        output,
        parent_manifest_sha256="a" * 64,
        forecast_manifest_sha256="b" * 64,
        parent_study_id="signal-parent",
        station_id="KLGA",
        target_range=["2026-04-01", "2026-04-30"],
        summary=summary,
        vector_features=frame,
        token_candidates=frame,
        slice_atlas=frame,
        crossfit_results=frame,
        fold_rules=frame,
        shortlist=frame,
    )
    artifact = load_signal_slice_artifact(output)

    assert manifest["data_class"] == "forecast_signal_slice_study"
    assert artifact.manifest["parent_signal_manifest_sha256"] == "a" * 64
    assert set(manifest["sha256"]) == {
        "vector_features.parquet",
        "token_candidates.parquet",
        "slice_atlas.parquet",
        "crossfit_results.parquet",
        "fold_rules.parquet",
        "shortlist.parquet",
        "summary.json",
    }

    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_signal_slice_artifact(output)
