from __future__ import annotations

import pandas as pd
import pytest

from backtest.two_bucket_probability_stack import (
    apply_expanding_probability_stacks,
    CURRENT_MODEL_ONLY,
    DEFAULT_STACK_SPECS,
    build_probability_stack_report,
    fit_expanding_probability_stacks,
    score_probability_stack_samples,
    write_probability_stack_artifact,
)
from backtest.two_bucket_union_study import fit_expanding_union_calibration
from scripts.run_two_bucket_probability_stack import _assert_current_a2_identity


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d", "e", "f"],
            "snapshot_id": ["sa", "sb", "sc", "sd", "se", "sf"],
            "target_date": [
                "2025-04-01", "2025-04-02", "2025-06-01",
                "2025-06-02", "2025-08-01", "2025-08-02",
            ],
            "fold_id": ["fold-00", "fold-00", "fold-01", "fold-01", "fold-02", "fold-02"],
            "model_union_probability": [0.55, 0.80, 0.60, 0.75, 0.65, 0.85],
            "midpoint_union_probability": [0.45, 0.60, 0.55, 0.65, 0.50, 0.70],
            "lead_hours": [30.0, 38.0, 30.0, 38.0, 30.0, 38.0],
            "union_outcome": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


def test_current_model_stack_matches_existing_a2_expanding_calibration() -> None:
    samples = _samples()

    existing, _ = fit_expanding_union_calibration(
        samples,
        bootstrap_samples=20,
        seed=7,
    )
    result = fit_expanding_probability_stacks(
        samples,
        specs=(CURRENT_MODEL_ONLY,),
        bootstrap_samples=20,
        seed=7,
    )
    current = result.scored_samples.sort_values(["fold_id", "event_id"])
    existing = existing.sort_values(["fold_id", "event_id"])

    assert current["probability_source"].unique().tolist() == ["current_model_only"]
    assert current["calibrated_union_probability"].tolist() == pytest.approx(
        existing["calibrated_union_probability"].tolist()
    )
    assert current["union_probability_lower_90"].tolist() == pytest.approx(
        existing["union_probability_lower_90"].tolist()
    )
    assert current["union_probability_upper_90"].tolist() == pytest.approx(
        existing["union_probability_upper_90"].tolist()
    )


def test_expanding_stacks_never_read_the_evaluation_fold_outcome() -> None:
    samples = _samples()
    result = fit_expanding_probability_stacks(
        samples,
        specs=DEFAULT_STACK_SPECS,
        bootstrap_samples=10,
        seed=3,
    )
    changed = samples.copy()
    changed.loc[changed["fold_id"] == "fold-02", "union_outcome"] = (
        1.0 - changed.loc[changed["fold_id"] == "fold-02", "union_outcome"]
    )
    changed_result = fit_expanding_probability_stacks(
        changed,
        specs=DEFAULT_STACK_SPECS,
        bootstrap_samples=10,
        seed=3,
    )
    columns = [
        "fold_id", "event_id", "probability_source",
        "calibrated_union_probability", "union_probability_lower_90",
    ]
    left = result.scored_samples.loc[:, columns].sort_values(columns[:3]).reset_index(drop=True)
    right = changed_result.scored_samples.loc[:, columns].sort_values(columns[:3]).reset_index(drop=True)
    fold_two = left["fold_id"] == "fold-02"

    assert left.loc[fold_two, "calibrated_union_probability"].tolist() == pytest.approx(
        right.loc[fold_two, "calibrated_union_probability"].tolist()
    )
    assert left.loc[fold_two, "union_probability_lower_90"].tolist() == pytest.approx(
        right.loc[fold_two, "union_probability_lower_90"].tolist()
    )
    contracts = result.coefficients[
        result.coefficients["coefficient_kind"] == "point"
    ]
    assert set(contracts["evaluation_fold_id"]) == {"fold-01", "fold-02"}
    assert all(
        fit_folds == ["fold-00"]
        for fit_folds in contracts.loc[
            contracts["evaluation_fold_id"] == "fold-01", "fit_fold_ids"
        ]
    )
    assert all(
        fit_folds == ["fold-00", "fold-01"]
        for fit_folds in contracts.loc[
            contracts["evaluation_fold_id"] == "fold-02", "fit_fold_ids"
        ]
    )


def test_expanding_models_apply_to_full_candidate_ticks_by_matching_fold() -> None:
    samples = _samples()
    result = fit_expanding_probability_stacks(
        samples,
        specs=DEFAULT_STACK_SPECS,
        bootstrap_samples=5,
        seed=2,
    )
    candidates = pd.concat(
        [samples, samples.assign(snapshot_id=lambda frame: frame["snapshot_id"] + "-later")],
        ignore_index=True,
    )

    applied = apply_expanding_probability_stacks(candidates, result.models)

    assert set(applied["fold_id"]) == {"fold-01", "fold-02"}
    assert set(applied["probability_source"]) == {
        "current_model_only", "market_only", "model_plus_market",
    }
    assert len(applied) == 2 * 4 * 3


def test_report_uses_paired_event_scores_to_measure_incremental_model_information() -> None:
    base = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "snapshot_id": ["sa", "sb", "sc", "sd"],
            "target_date": ["2025-06-01", "2025-06-02", "2025-08-01", "2025-08-02"],
            "fold_id": ["fold-01", "fold-01", "fold-02", "fold-02"],
            "union_outcome": [1.0, 0.0, 1.0, 0.0],
            "lead_hours": [38.0, 38.0, 30.0, 30.0],
            "horizon_band": ["36-42h", "36-42h", "30-36h", "30-36h"],
            "model_market_gap_band": ["<0.15"] * 4,
        }
    )
    probabilities = {
        "market_only": [0.5, 0.5, 0.5, 0.5],
        "current_model_only": [0.7, 0.3, 0.7, 0.3],
        "model_plus_market": [0.9, 0.1, 0.8, 0.2],
    }
    scored = pd.concat(
        [
            base.assign(
                probability_source=source,
                calibrated_union_probability=values,
                union_probability_lower_90=[max(0.0, value - 0.05) for value in values],
            )
            for source, values in probabilities.items()
        ],
        ignore_index=True,
    )
    scores = score_probability_stack_samples(scored)
    settlements = pd.concat(
        [
            base.assign(
                probability_source=source,
                realized_pnl_usd=value,
                cost_basis_usd=5.0,
            )
            for source, value in {
                "market_only": 0.0,
                "current_model_only": 1.0,
                "model_plus_market": 2.0,
            }.items()
        ],
        ignore_index=True,
    )
    coefficients = pd.DataFrame(
        {
            "evaluation_fold_id": ["fold-01", "fold-02"],
            "probability_source": ["model_plus_market"] * 2,
            "coefficient_kind": ["point"] * 2,
            "model_logit": [0.4, 0.3],
            "midpoint_logit": [0.5, 0.6],
            "is_36_42h": [0.1, 0.1],
        }
    )

    report = build_probability_stack_report(
        scores=scores,
        coefficients=coefficients,
        candidates=base,
        settlements=settlements,
        bootstrap_samples=50,
        seed=5,
    )

    incremental = report["proper_scores"]["model_plus_market_minus_market_only"]
    assert incremental["log_loss_delta"] < 0.0
    assert incremental["log_loss_delta_ci90_high"] < 0.0
    assert incremental["brier_delta"] < 0.0
    question = next(
        row for row in report["decision_ledger"]
        if row["question_id"] == "incremental_model_information"
    )
    assert question["status"] == "supported"


def test_probability_stack_artifact_is_immutable_and_hash_complete(tmp_path) -> None:
    output = tmp_path / "stack"
    frames = {
        "probability_scores": pd.DataFrame({"event_id": ["a"]}),
        "calibration_coefficients": pd.DataFrame({"evaluation_fold_id": ["fold-01"]}),
        "probability_panel": pd.DataFrame({"event_id": ["a"]}),
        "decisions": pd.DataFrame({"event_id": ["a"]}),
        "execution_legs": pd.DataFrame({"event_id": ["a"]}),
        "event_pnl": pd.DataFrame({"event_id": ["a"]}),
    }

    manifest = write_probability_stack_artifact(
        output=output,
        frames=frames,
        report={"schema_version": 1},
        input_sha256={"historical_a2_manifest": "abc"},
    )

    assert manifest["data_class"] == "two_bucket_probability_stack_historical_oof"
    assert set(manifest["sha256"]) == {
        "probability_scores.parquet", "calibration_coefficients.parquet",
        "probability_panel.parquet", "decisions.parquet",
        "execution_legs.parquet", "event_pnl.parquet", "report.json",
    }
    with pytest.raises(FileExistsError):
        write_probability_stack_artifact(
            output=output,
            frames=frames,
            report={},
            input_sha256={},
        )


def test_current_a2_identity_checks_intermediate_probabilities(tmp_path) -> None:
    root = tmp_path / "a2"
    root.mkdir()
    panel = pd.DataFrame(
        {
            "event_id": ["training", "evaluation"],
            "timestamp": pd.to_datetime(
                ["2025-04-01T00:00:00Z", "2025-06-01T00:00:00Z"], utc=True
            ),
            "snapshot_id": ["s0", "s1"],
            "fold_id": ["fold-00", "fold-01"],
            "calibrated_union_probability": [0.6, 0.7],
            "union_probability_lower_90": [0.5, 0.6],
            "union_probability_upper_90": [0.7, 0.8],
        }
    )
    panel.to_parquet(root / "selection_panel.parquet", index=False)
    settlement = pd.DataFrame(
        {
            "stage": ["frozen"],
            "execution_mode": ["taker"],
            "edge_threshold": [0.0],
            "event_id": ["evaluation"],
            "timestamp": pd.to_datetime(["2025-06-01T00:00:00Z"], utc=True),
            "snapshot_id": ["s1"],
            "cost_basis_usd": [4.0],
            "realized_pnl_usd": [6.0],
        }
    )
    settlement.to_parquet(root / "event_pnl.parquet", index=False)
    actual_panel = panel[panel["fold_id"] == "fold-01"].assign(
        probability_source="current_model_only"
    )
    actual_settlement = settlement.assign(probability_source="current_model_only")

    _assert_current_a2_identity(
        root,
        probability_panel=actual_panel,
        settlements=actual_settlement,
    )

    changed = actual_panel.copy()
    changed["union_probability_lower_90"] += 0.01
    with pytest.raises(ValueError, match="union_probability_lower_90"):
        _assert_current_a2_identity(
            root,
            probability_panel=changed,
            settlements=actual_settlement,
        )
