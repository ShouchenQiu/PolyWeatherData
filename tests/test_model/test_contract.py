from __future__ import annotations

import json

import pytest

from model.contract import validate_model_contract
from model.features import OBS_UNKNOWN_REALS, TFT_KNOWN_REALS


def _write_contract(tmp_path, schema_version: int, unknown_reals: list[str]):
    path = tmp_path / "model_contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "target": "temp_error",
                "known_reals": TFT_KNOWN_REALS,
                "unknown_reals": unknown_reals,
                "max_encoder_length": 72,
                "max_prediction_length": 48,
                "quantiles": [0.1, 0.5, 0.9],
            }
        ),
        encoding="ascii",
    )
    return path


def test_contract_validator_keeps_v1_models_loadable(tmp_path) -> None:
    contract = validate_model_contract(
        _write_contract(tmp_path, 1, list(OBS_UNKNOWN_REALS))
    )

    assert contract["schema_version"] == 1


def test_contract_validator_accepts_baseline_features_in_calibrated_v2(
    tmp_path,
) -> None:
    contract = validate_model_contract(
        _write_contract(tmp_path, 2, list(OBS_UNKNOWN_REALS))
    )

    assert contract["schema_version"] == 2


def test_contract_validator_rejects_retired_history_feature_in_v2(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown_reals mismatch"):
        validate_model_contract(
            _write_contract(
                tmp_path,
                2,
                list(OBS_UNKNOWN_REALS) + ["hist_temp_error"],
            )
        )
