"""Production compatibility checks for trained TFT model artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model.features import TARGET, TFT_KNOWN_REALS, TFT_UNKNOWN_REALS


SUPPORTED_CONTRACT_SCHEMA = 1
SUPPORTED_ENCODER_LENGTH = 72
SUPPORTED_PREDICTION_LENGTH = 48
SUPPORTED_QUANTILES = [0.1, 0.5, 0.9]


def validate_model_contract(path: Path | str) -> dict[str, Any]:
    """Load and validate that a model contract matches production inference."""
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"model_contract.json not found at {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _require_equal(contract, "schema_version", SUPPORTED_CONTRACT_SCHEMA)
    _require_equal(contract, "target", TARGET)
    _require_equal(contract, "known_reals", list(TFT_KNOWN_REALS))
    _require_equal(contract, "unknown_reals", list(TFT_UNKNOWN_REALS))
    _require_equal(contract, "max_encoder_length", SUPPORTED_ENCODER_LENGTH)
    _require_equal(contract, "max_prediction_length", SUPPORTED_PREDICTION_LENGTH)
    _require_equal(contract, "quantiles", SUPPORTED_QUANTILES)
    return contract


def _require_equal(contract: dict[str, Any], field: str, expected: Any) -> None:
    actual = contract.get(field)
    if actual != expected:
        raise ValueError(
            f"model_contract.json {field} mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )
