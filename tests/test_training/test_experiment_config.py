from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from training.config import (
    ExperimentTemplate,
    ResolvedTrialSnapshot,
)


def _minimal_template() -> dict:
    return json.loads(
        Path("experiments/klga/tft_optuna_v1.json").read_text(encoding="ascii")
    )


def test_experiment_template_requires_explicit_schema_version() -> None:
    payload = _minimal_template()
    del payload["schema_version"]

    with pytest.raises(ValueError, match=r"missing required field.*schema_version"):
        ExperimentTemplate.from_dict(payload)


def test_experiment_template_rejects_unknown_top_level_fields() -> None:
    payload = _minimal_template()
    payload["staton"] = "KLGA"

    with pytest.raises(ValueError, match=r"unknown field.*staton"):
        ExperimentTemplate.from_dict(payload)


def test_experiment_template_rejects_unknown_nested_fields() -> None:
    payload = _minimal_template()
    payload["model"] = {"hidden_sze": 64}

    with pytest.raises(ValueError, match=r"model contains unknown field.*hidden_sze"):
        ExperimentTemplate.from_dict(payload)


def test_versioned_klga_template_has_stable_canonical_hash() -> None:
    path = "experiments/klga/tft_optuna_v1.json"

    first = ExperimentTemplate.from_json(path)
    second = ExperimentTemplate.from_json(path)

    assert first.template_id == "tft_optuna_v1"
    assert first.station == "KLGA"
    assert first.study.budget_gpu_hours == 48.0
    assert first.study.objective == "validation_quantile_loss"
    assert first.config_hash() == second.config_hash()


def test_experiment_template_rejects_workspace_for_another_station() -> None:
    template = ExperimentTemplate.from_json(
        "experiments/klga/tft_optuna_v1.json"
    )

    with pytest.raises(ValueError, match=r"station.*KJFK.*KLGA"):
        template.validate_audit(
            {
                "ready": True,
                "station_id": "KJFK",
                "data_order_hash": "audit-hash",
            }
        )


def test_schema_v1_rejects_unproven_batch_sizes_in_search_space() -> None:
    payload = _minimal_template()
    payload["study"]["search_space"]["batch_size"]["choices"] = [4, 16]

    with pytest.raises(ValueError, match=r"batch_size.*4 and 8"):
        ExperimentTemplate.from_dict(payload)


def test_resolved_trial_snapshot_is_explicit_and_hashable() -> None:
    payload = {
        "schema_version": 1,
        "study_id": "klga-tft-study",
        "trial_number": 3,
        "execution_type": "trial",
        "template_hash": "template-hash",
        "data_order_hash": "data-hash",
        "parameters": {"hidden_size": 32, "batch_size": 4},
    }

    first = ResolvedTrialSnapshot.from_dict(payload)
    second = ResolvedTrialSnapshot.from_dict(payload)

    assert first.snapshot_hash() == second.snapshot_hash()
    assert first.parameters["hidden_size"] == 32


def test_parsed_experiment_template_is_deeply_immutable() -> None:
    template = ExperimentTemplate.from_json(
        "experiments/klga/tft_optuna_v1.json"
    )

    with pytest.raises(FrozenInstanceError):
        template.model.hidden_size = 64
    with pytest.raises(TypeError):
        template.splits["train"] = ("2025-01-01", "2025-12-31")
    with pytest.raises(TypeError):
        template.study.search_space["hidden_size"] = template.study.search_space[
            "batch_size"
        ]
