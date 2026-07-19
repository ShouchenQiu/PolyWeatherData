from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.two_bucket_probability_stack import (
    DEFAULT_STACK_SPECS,
    fit_probability_stack,
    score_probability_stack_samples,
)
from backtest.two_bucket_probability_stack_frozen import (
    apply_frozen_probability_stacks,
    build_frozen_probability_stack_report,
    load_frozen_probability_stacks,
    write_frozen_probability_stack_artifact,
)


def _training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "target_date": [
                "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04",
            ],
            "model_union_probability": [0.7, 0.4, 0.8, 0.3],
            "midpoint_union_probability": [0.6, 0.5, 0.7, 0.4],
            "lead_hours": [30.0, 38.0, 30.0, 38.0],
            "union_outcome": [1.0, 0.0, 1.0, 0.0],
        }
    )


def test_frozen_probability_stacks_round_trip_without_refitting() -> None:
    training = _training()
    fitted = [
        fit_probability_stack(
            training,
            spec=spec,
            bootstrap_samples=5,
            seed=11,
            fit_fold_ids=("fold-00",),
        )
        for spec in DEFAULT_STACK_SPECS
    ]
    coefficients = pd.concat(
        [model.coefficient_frame(evaluation_fold_id="frozen_pre_april") for model in fitted],
        ignore_index=True,
    )

    loaded = load_frozen_probability_stacks(
        coefficients,
        expected_bootstrap_samples=5,
        training_cutoff="2026-03-31",
    )

    assert set(loaded) == {spec.name for spec in DEFAULT_STACK_SPECS}
    for original in fitted:
        restored = loaded[original.spec.name]
        assert restored.coefficients == pytest.approx(original.coefficients)
        assert np.asarray(restored.bootstrap_coefficients) == pytest.approx(
            np.asarray(original.bootstrap_coefficients)
        )
        assert restored.fit_target_end <= "2026-03-31"


def test_frozen_stacks_apply_all_sources_without_reading_diagnostic_outcomes() -> None:
    fitted = {
        spec.name: fit_probability_stack(
            _training(),
            spec=spec,
            bootstrap_samples=5,
            seed=13,
            fit_fold_ids=("fold-00",),
        )
        for spec in DEFAULT_STACK_SPECS
    }
    diagnostic = _training().iloc[:2].assign(
        event_id=["april", "may"],
        target_date=["2026-04-01", "2026-05-01"],
        snapshot_id=["sa", "sm"],
        evaluation_id=["april_2026", "may_2026"],
    )

    applied = apply_frozen_probability_stacks(diagnostic, fitted)
    changed = diagnostic.assign(union_outcome=1.0 - diagnostic["union_outcome"])
    applied_after_outcome_change = apply_frozen_probability_stacks(changed, fitted)
    order = ["event_id", "probability_source"]
    columns = [
        "calibrated_union_probability",
        "union_probability_lower_90",
        "union_probability_upper_90",
    ]

    assert len(applied) == len(diagnostic) * len(DEFAULT_STACK_SPECS)
    assert set(applied["probability_source"]) == {
        spec.name for spec in DEFAULT_STACK_SPECS
    }
    left = applied.sort_values(order).reset_index(drop=True)
    right = applied_after_outcome_change.sort_values(order).reset_index(drop=True)
    assert left[columns].to_numpy() == pytest.approx(right[columns].to_numpy())


def test_frozen_report_answers_april_and_may_increment_and_pnl_separately() -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"a{i}" for i in range(4)] + [f"m{i}" for i in range(4)],
            "snapshot_id": [f"s{i}" for i in range(8)],
            "timestamp": pd.to_datetime(
                [f"2026-04-{i + 1:02d}T00:00:00Z" for i in range(4)]
                + [f"2026-05-{i + 1:02d}T00:00:00Z" for i in range(4)],
                utc=True,
            ),
            "target_date": [f"2026-04-{i + 1:02d}" for i in range(4)]
            + [f"2026-05-{i + 1:02d}" for i in range(4)],
            "evaluation_id": ["april_2026"] * 4 + ["may_2026"] * 4,
            "union_outcome": [1.0, 0.0, 1.0, 0.0] * 2,
            "lead_hours": [38.0] * 8,
            "horizon_band": ["36-42h"] * 8,
            "model_market_gap_band": ["<0.15"] * 8,
        }
    )
    probabilities = {
        "current_model_only": [0.7, 0.3, 0.7, 0.3] + [0.5] * 4,
        "market_only": [0.5] * 4 + [0.8, 0.2, 0.8, 0.2],
        "model_plus_market": [0.8, 0.2, 0.8, 0.2]
        + [0.6, 0.4, 0.6, 0.4],
    }
    scores = score_probability_stack_samples(
        pd.concat(
            [
                events.assign(
                    probability_source=source,
                    calibrated_union_probability=values,
                )
                for source, values in probabilities.items()
            ],
            ignore_index=True,
        )
    )
    pnl = {
        "current_model_only": [1.0] * 4 + [-1.0] * 4,
        "market_only": [0.0] * 8,
        "model_plus_market": [2.0] * 4 + [0.5] * 4,
    }
    settlements = pd.concat(
        [
            events.assign(
                probability_source=source,
                realized_pnl_usd=values,
                cost_basis_usd=5.0,
            )
            for source, values in pnl.items()
        ],
        ignore_index=True,
    )

    report = build_frozen_probability_stack_report(
        scores=scores,
        candidates=events,
        settlements=settlements,
        exclusions=pd.DataFrame(columns=["evaluation_id", "target_date", "reason", "source"]),
        bootstrap_samples=100,
        seed=7,
    )

    ledger = {row["question_id"]: row for row in report["decision_ledger"]}
    assert ledger["april_incremental_model_information"]["status"] == "supported"
    assert ledger["may_incremental_model_information"]["status"] == "contradicted"
    assert ledger["april_market_aware_pnl_vs_current"]["status"] == "supported"
    assert ledger["may_market_aware_pnl_vs_current"]["status"] == "supported"
    assert report["proper_scores"]["by_evaluation"][0]["evaluation_id"] == "april_2026"
    transitions = report["execution"][
        "entry_transitions_model_plus_market_vs_current_model_only"
    ]
    assert transitions[0]["both_traded_event_days"] == 4
    assert transitions[0]["same_entry_timestamp_event_days"] == 4


def test_frozen_probability_stack_artifact_is_immutable_and_hash_complete(tmp_path) -> None:
    output = tmp_path / "frozen"
    frames = {
        name: pd.DataFrame({"event_id": ["a"]})
        for name in (
            "probability_scores", "calibration_coefficients", "probability_panel",
            "decisions", "execution_legs", "event_pnl", "exclusions",
        )
    }

    manifest = write_frozen_probability_stack_artifact(
        output=output,
        frames=frames,
        report={"schema_version": 1},
        input_sha256={"historical_stack_manifest": "abc", "diagnostic_manifest": "def"},
    )

    assert manifest["data_class"] == (
        "frozen_two_bucket_probability_stack_april_may_diagnostic"
    )
    assert set(manifest["sha256"]) == {
        *(f"{name}.parquet" for name in frames),
        "report.json",
    }
    with pytest.raises(FileExistsError):
        write_frozen_probability_stack_artifact(
            output=output,
            frames=frames,
            report={},
            input_sha256={},
        )
