from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from backtest.peak_margin_crossfit import (
    build_peak_margin_score_panel,
    fit_expanding_peak_margin_contracts,
    summarize_peak_margin_crossfit,
    validate_peak_margin_input_lineage,
    write_peak_margin_crossfit_artifact,
)


def test_score_panel_keeps_first_legal_tick_per_forecast_snapshot() -> None:
    timestamps = pd.to_datetime(
        ["2025-06-01T00:05:00Z", "2025-06-01T00:10:00Z"], utc=True
    )
    panel = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "target_date": "2025-06-01",
                "timestamp": timestamp,
                "snapshot_id": "snapshot-1",
                "bucket_order": bucket,
                "model_probability": model,
                "midpoint_probability": midpoint,
                "yes_outcome": outcome,
                "lead_hours": 36.0,
            }
            for timestamp in timestamps
            for bucket, model, midpoint, outcome in (
                (0, 0.4, 0.5, 0.0),
                (1, 0.6, 0.5, 1.0),
            )
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "target_date": "2025-06-01",
                "timestamp": timestamp,
                "source": source,
                "normalized_rps": rps,
                "log_score": log_score,
                "multiclass_brier": brier,
                "true_bucket_probability": truth_probability,
            }
            for timestamp, offset in zip(timestamps, (0.0, 0.2), strict=True)
            for source, rps, log_score, brier, truth_probability in (
                ("model_raw", 0.10 + offset, 0.50 + offset, 0.40 + offset, 0.60),
                ("synthetic_midpoint", 0.08 + offset, 0.45 + offset, 0.35 + offset, 0.50),
            )
        ]
    )
    snapshots = pd.DataFrame(
        {
            "snapshot_id": ["snapshot-1"],
            "target_date": ["2025-06-01"],
            "available_at": pd.to_datetime(["2025-06-01T00:04:00Z"], utc=True),
            "forecast_start_at": pd.to_datetime(["2025-06-01T01:00:00Z"], utc=True),
            "forecast_through": pd.to_datetime(["2025-06-02T00:00:00Z"], utc=True),
            "forecast_hour_count": [24],
            "coverage_type": ["forecast_only"],
            "lineage_id": ["checkpoint-aligned-static-mc-fold-01"],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "snapshot_id": ["snapshot-1"],
            "target_date": ["2025-06-01"],
            "tft_peak_margin_f": [1.2],
            "tft_peak_slot_at": pd.to_datetime(["2025-06-01T12:00:00Z"], utc=True),
            "tft_near_peak_hour_count_1f": [1],
        }
    )

    result = build_peak_margin_score_panel(
        signal_panel=panel,
        scores_by_tick=scores,
        snapshots=snapshots,
        peak_diagnostics=diagnostics,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["timestamp"] == timestamps[0]
    assert row["fold_id"] == "fold-01"
    assert row["horizon_band"] == "36-42h"
    assert row["relative_rps"] == pytest.approx(0.02)
    assert row["relative_log_score"] == pytest.approx(0.05)
    assert row["relative_multiclass_brier"] == pytest.approx(0.05)
    assert row["true_bucket_probability_delta"] == pytest.approx(0.10)
    assert not bool(row["peak_at_boundary"])


def test_score_panel_marks_cross_boundary_checkpoint_rows_ineligible() -> None:
    timestamp = pd.Timestamp("2025-06-01T00:05:00Z")
    panel = pd.DataFrame(
        {
            "event_id": ["event-1"], "target_date": ["2025-06-01"],
            "timestamp": [timestamp], "snapshot_id": ["snapshot-1"],
            "bucket_order": [0], "model_probability": [1.0],
            "midpoint_probability": [1.0], "yes_outcome": [1.0],
            "lead_hours": [36.0],
        }
    )
    scores = pd.DataFrame(
        {
            "event_id": ["event-1", "event-1"],
            "target_date": ["2025-06-01", "2025-06-01"],
            "timestamp": [timestamp, timestamp],
            "source": ["model_raw", "synthetic_midpoint"],
            "normalized_rps": [0.0, 0.0], "log_score": [0.0, 0.0],
            "multiclass_brier": [0.0, 0.0],
            "true_bucket_probability": [1.0, 1.0],
        }
    )
    snapshots = pd.DataFrame(
        {
            "snapshot_id": ["snapshot-1"], "target_date": ["2025-06-01"],
            "available_at": [pd.Timestamp("2025-06-01T00:04:00Z")],
            "forecast_start_at": [pd.Timestamp("2025-06-01T01:00:00Z")],
            "forecast_through": [pd.Timestamp("2025-06-02T00:00:00Z")],
            "forecast_hour_count": [24], "coverage_type": ["forecast_only"],
            "lineage_id": ["checkpoint-aligned-static-mc-fold-00"],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "snapshot_id": ["snapshot-1"], "target_date": ["2025-06-01"],
            "tft_peak_margin_f": [1.0],
            "tft_peak_slot_at": [pd.Timestamp("2025-06-01T12:00:00Z")],
            "tft_near_peak_hour_count_1f": [1],
        }
    )

    result = build_peak_margin_score_panel(
        signal_panel=panel, scores_by_tick=scores, snapshots=snapshots,
        peak_diagnostics=diagnostics,
    )

    assert result.loc[0, "fold_id"] == "fold-01"
    assert result.loc[0, "lineage_fold_id"] == "fold-00"
    assert not bool(result.loc[0, "strict_fold_eligible"])


def test_expanding_contracts_use_only_earlier_forecast_only_folds() -> None:
    panel = pd.DataFrame(
        {
            "snapshot_id": [f"s{index}" for index in range(9)],
            "target_date": [
                "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04",
                "2025-06-01", "2025-06-02", "2025-06-03",
                "2025-08-01", "2025-08-02",
            ],
            "fold_id": ["fold-00"] * 4 + ["fold-01"] * 3 + ["fold-02"] * 2,
            "coverage_type": ["forecast_only"] * 8 + ["observed_plus_forecast"],
            "strict_fold_eligible": [True] * 6 + [False, True, True],
            "tft_peak_margin_f": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 6.0, 200.0],
        }
    )

    scored, contracts = fit_expanding_peak_margin_contracts(panel)

    assert set(scored["fold_id"]) == {"fold-01", "fold-02"}
    fold_one = contracts.set_index("evaluation_fold_id").loc["fold-01"]
    fold_two = contracts.set_index("evaluation_fold_id").loc["fold-02"]
    assert fold_one["fit_fold_ids"] == ["fold-00"]
    assert fold_one["expanding_q75_f"] == pytest.approx(2.25)
    assert fold_two["fit_fold_ids"] == ["fold-00", "fold-01"]
    assert fold_two["expanding_q75_f"] == pytest.approx(3.75)
    assert fold_two["fixed_fold00_q75_f"] == pytest.approx(2.25)

    evaluation = scored.set_index("snapshot_id")
    assert evaluation.loc["s7", "peak_margin_percentile"] == pytest.approx(1.0)
    assert bool(evaluation.loc["s7", "peak_margin_high"])
    assert bool(evaluation.loc["s8", "peak_margin_high"])


def test_summary_weights_event_days_instead_of_repeated_snapshots() -> None:
    scored = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "a"],
            "snapshot_id": ["a1", "a2", "b1", "a3"],
            "target_date": ["2025-06-01", "2025-06-01", "2025-06-02", "2025-06-01"],
            "fold_id": ["fold-01"] * 4,
            "coverage_type": ["forecast_only"] * 4,
            "peak_margin_high": [True, True, True, False],
            "peak_margin_high_fixed_fold00": [True, True, True, False],
            "peak_margin_percentile": [0.9, 0.8, 0.95, 0.2],
            "horizon_band": ["36-42h"] * 4,
            "peak_at_boundary": [False] * 4,
            "relative_rps": [0.0, 2.0, 3.0, 4.0],
            "relative_log_score": [0.0, 2.0, 3.0, 4.0],
            "relative_multiclass_brier": [0.0, 2.0, 3.0, 4.0],
            "true_bucket_probability_delta": [0.0, -2.0, -3.0, -4.0],
        }
    )
    contracts = pd.DataFrame(
        {
            "evaluation_fold_id": ["fold-01"],
            "fit_fold_ids": [["fold-00"]],
            "expanding_q75_f": [0.8],
            "fixed_fold00_q75_f": [0.8],
        }
    )

    result = summarize_peak_margin_crossfit(
        scored,
        contracts=contracts,
        bootstrap_samples=50,
        seed=9,
    )

    high = result.fold_summaries.query(
        "threshold_mode == 'expanding_q75' and coverage_scope == 'forecast_only_primary' "
        "and fold_id == 'pooled' and stratum_type == 'overall' "
        "and margin_group == 'high'"
    ).iloc[0]
    assert high["event_days"] == 2
    assert high["snapshots"] == 3
    assert high["relative_rps"] == pytest.approx(2.0)
    assert {
        "overall", "horizon", "month", "peak_boundary",
    }.issubset(set(result.fold_summaries["stratum_type"]))
    repeated = summarize_peak_margin_crossfit(
        scored,
        contracts=contracts,
        bootstrap_samples=50,
        seed=9,
    )
    pd.testing.assert_frame_equal(result.fold_summaries, repeated.fold_summaries)
    pd.testing.assert_frame_equal(result.continuous_effects, repeated.continuous_effects)


def test_report_distinguishes_absolute_advantage_from_risk_reduction() -> None:
    folds = [f"fold-{index:02d}" for index in range(1, 6)]
    rows = []
    for index, fold_id in enumerate(folds, start=1):
        for group, relative_rps, percentile in (
            (True, 0.1, 0.9),
            (False, 0.3, 0.2),
        ):
            rows.append(
                {
                    "event_id": f"event-{index}",
                    "snapshot_id": f"snapshot-{index}-{group}",
                    "target_date": f"2025-{index + 5:02d}-01",
                    "fold_id": fold_id,
                    "coverage_type": "forecast_only",
                    "peak_margin_high": group,
                    "peak_margin_high_fixed_fold00": group,
                    "peak_margin_percentile": percentile,
                    "horizon_band": "36-42h",
                    "peak_at_boundary": False,
                    "relative_rps": relative_rps,
                    "relative_log_score": relative_rps,
                    "relative_multiclass_brier": relative_rps,
                    "true_bucket_probability_delta": -relative_rps,
                }
            )
    contracts = pd.DataFrame(
        {
            "evaluation_fold_id": folds,
            "fit_fold_ids": [["fold-00"]] * len(folds),
            "expanding_q75_f": [0.8] * len(folds),
            "fixed_fold00_q75_f": [0.8] * len(folds),
        }
    )

    result = summarize_peak_margin_crossfit(
        pd.DataFrame(rows),
        contracts=contracts,
        bootstrap_samples=20,
        seed=5,
    )

    questions = {
        row["question_id"]: row for row in result.report["decision_ledger"]
    }
    assert questions["high_peak_margin_absolute_advantage"]["status"] == "contradicted"
    assert questions["high_peak_margin_risk_screening"]["status"] == "supported"
    selectivity = result.fold_summaries.query(
        "threshold_mode == 'expanding_q75' and coverage_scope == 'forecast_only_primary' "
        "and fold_id == 'pooled' and stratum_type == 'overall' "
        "and margin_group == 'high_minus_non_high'"
    ).iloc[0]
    assert selectivity["relative_rps"] == pytest.approx(-0.2)


def test_adjusted_continuous_effect_is_event_day_clustered_and_reported() -> None:
    folds = [f"fold-{index:02d}" for index in range(1, 6)]
    rows = []
    for fold_index, fold_id in enumerate(folds, start=1):
        for day_index, percentile in enumerate((0.1, 0.3, 0.7, 0.9), start=1):
            rows.append(
                {
                    "event_id": f"event-{fold_index}-{day_index}",
                    "snapshot_id": f"snapshot-{fold_index}-{day_index}",
                    "target_date": f"2025-{fold_index + 5:02d}-{day_index:02d}",
                    "fold_id": fold_id,
                    "coverage_type": "forecast_only",
                    "peak_margin_high": percentile >= 0.75,
                    "peak_margin_high_fixed_fold00": percentile >= 0.75,
                    "peak_margin_percentile": percentile,
                    "horizon_band": "36-42h" if day_index % 2 else "30-36h",
                    "peak_at_boundary": day_index == 4,
                    "relative_rps": 1.0 - percentile,
                    "relative_log_score": 1.0 - percentile,
                    "relative_multiclass_brier": 1.0 - percentile,
                    "true_bucket_probability_delta": percentile - 1.0,
                }
            )
    contracts = pd.DataFrame(
        {
            "evaluation_fold_id": folds,
            "fit_fold_ids": [["fold-00"]] * len(folds),
            "expanding_q75_f": [0.8] * len(folds),
            "fixed_fold00_q75_f": [0.8] * len(folds),
        }
    )

    result = summarize_peak_margin_crossfit(
        pd.DataFrame(rows),
        contracts=contracts,
        bootstrap_samples=20,
        seed=11,
    )

    pooled = result.continuous_effects.query("fold_id == 'pooled'").iloc[0]
    assert pooled["percentile_slope"] == pytest.approx(-1.0)
    question = next(
        row for row in result.report["decision_ledger"]
        if row["question_id"] == "peak_margin_adjusted_independence"
    )
    assert question["status"] == "supported"


def test_peak_margin_artifact_is_immutable_and_hash_complete(tmp_path) -> None:
    output = tmp_path / "peak-margin"
    frames = {
        "snapshot_scores": pd.DataFrame({"snapshot_id": ["s1"]}),
        "fold_contracts": pd.DataFrame({"evaluation_fold_id": ["fold-01"]}),
        "fold_summaries": pd.DataFrame({"fold_id": ["fold-01"]}),
        "continuous_effects": pd.DataFrame({"fold_id": ["pooled"]}),
    }

    manifest = write_peak_margin_crossfit_artifact(
        output=output,
        frames=frames,
        report={"schema_version": 1},
        input_sha256={"forecast_manifest": "abc"},
    )

    assert manifest["data_class"] == "peak_margin_historical_crossfit"
    assert set(manifest["sha256"]) == {
        "snapshot_scores.parquet", "fold_contracts.parquet",
        "fold_summaries.parquet", "continuous_effects.parquet", "report.json",
    }
    with pytest.raises(FileExistsError):
        write_peak_margin_crossfit_artifact(
            output=output,
            frames=frames,
            report={},
            input_sha256={},
        )


def test_input_lineage_rejects_mismatched_peak_and_signal_artifacts(tmp_path) -> None:
    forecast = tmp_path / "forecast"
    signal = tmp_path / "signal"
    peak = tmp_path / "peak"
    for root in (forecast, signal, peak):
        root.mkdir()
    forecast_manifest = forecast / "manifest.json"
    forecast_manifest.write_text(
        json.dumps({"data_class": "point_in_time_forecast_cache"}), encoding="ascii"
    )
    forecast_hash = hashlib.sha256(forecast_manifest.read_bytes()).hexdigest()
    signal_manifest = signal / "manifest.json"
    signal_manifest.write_text(
        json.dumps(
            {
                "data_class": "forecast_signal_study",
                "split_role": "oof_development",
                "input_sha256": {"forecast_manifest": forecast_hash},
            }
        ),
        encoding="ascii",
    )
    signal_hash = hashlib.sha256(signal_manifest.read_bytes()).hexdigest()
    peak_manifest = peak / "manifest.json"
    peak_manifest.write_text(
        json.dumps(
            {
                "data_class": "peak_union_diagnostics",
                "input_sha256": {
                    "forecast_cache_manifest": forecast_hash,
                    "signal_manifest": signal_hash,
                },
            }
        ),
        encoding="ascii",
    )

    hashes = validate_peak_margin_input_lineage(
        forecast_cache_root=forecast,
        signal_root=signal,
        peak_diagnostics_root=peak,
    )
    assert hashes["forecast_manifest"] == forecast_hash
    assert hashes["signal_manifest"] == signal_hash

    peak_manifest.write_text(
        json.dumps(
            {
                "data_class": "peak_union_diagnostics",
                "input_sha256": {
                    "forecast_cache_manifest": forecast_hash,
                    "signal_manifest": "wrong",
                },
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="signal manifest"):
        validate_peak_margin_input_lineage(
            forecast_cache_root=forecast,
            signal_root=signal,
            peak_diagnostics_root=peak,
        )
