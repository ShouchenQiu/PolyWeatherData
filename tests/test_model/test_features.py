"""Tests for model/features.py -- canonical feature column lists."""
from __future__ import annotations

import ast
from pathlib import Path


# ---------------------------------------------------------------------------
# Imports exist
# ---------------------------------------------------------------------------


def test_imports_are_available() -> None:
    from model.features import (  # noqa: F401
        HRRR_KNOWN_REALS,
        MODEL_GENERATED_COLUMNS,
        OBS_UNKNOWN_REALS,
        REQUIRED_DECODER_SOURCE_COLUMNS,
        REQUIRED_ENCODER_SOURCE_COLUMNS,
        TARGET,
    )


# ---------------------------------------------------------------------------
# HRRR_KNOWN_REALS
# ---------------------------------------------------------------------------


def test_hrrr_known_reals_is_nonempty_list() -> None:
    from model.features import HRRR_KNOWN_REALS
    assert isinstance(HRRR_KNOWN_REALS, list)
    assert len(HRRR_KNOWN_REALS) > 0


def test_hrrr_known_reals_contains_expected_weather_vars() -> None:
    from model.features import HRRR_KNOWN_REALS
    for col in ("TMP_2m", "DPT_2m", "RH_2m", "WS_10m", "CAPE", "SNOD"):
        assert col in HRRR_KNOWN_REALS, f"{col} missing from HRRR_KNOWN_REALS"


def test_hrrr_known_reals_has_no_duplicates() -> None:
    from model.features import HRRR_KNOWN_REALS
    assert len(HRRR_KNOWN_REALS) == len(set(HRRR_KNOWN_REALS))


# ---------------------------------------------------------------------------
# OBS_UNKNOWN_REALS
# ---------------------------------------------------------------------------


def test_obs_unknown_reals_is_nonempty_list() -> None:
    from model.features import OBS_UNKNOWN_REALS
    assert isinstance(OBS_UNKNOWN_REALS, list)
    assert len(OBS_UNKNOWN_REALS) > 0


def test_obs_unknown_reals_contains_expected_obs_vars() -> None:
    from model.features import OBS_UNKNOWN_REALS
    for col in ("raw_tmpc_tenths", "obs_WS", "is_rain", "is_snow"):
        assert col in OBS_UNKNOWN_REALS, f"{col} missing from OBS_UNKNOWN_REALS"


def test_obs_unknown_reals_has_no_duplicates() -> None:
    from model.features import OBS_UNKNOWN_REALS
    assert len(OBS_UNKNOWN_REALS) == len(set(OBS_UNKNOWN_REALS))


# ---------------------------------------------------------------------------
# TARGET
# ---------------------------------------------------------------------------


def test_target_is_temp_error() -> None:
    from model.features import TARGET
    assert TARGET == "temp_error"


# ---------------------------------------------------------------------------
# Derived column sets
# ---------------------------------------------------------------------------


def test_required_encoder_source_columns_is_hrrr_plus_obs_plus_target() -> None:
    from model.features import (
        HRRR_KNOWN_REALS,
        OBS_UNKNOWN_REALS,
        REQUIRED_ENCODER_SOURCE_COLUMNS,
        TARGET,
    )
    assert REQUIRED_ENCODER_SOURCE_COLUMNS == HRRR_KNOWN_REALS + OBS_UNKNOWN_REALS + [TARGET]


def test_required_decoder_source_columns_is_hrrr_only() -> None:
    from model.features import HRRR_KNOWN_REALS, REQUIRED_DECODER_SOURCE_COLUMNS
    assert REQUIRED_DECODER_SOURCE_COLUMNS == HRRR_KNOWN_REALS


def test_model_generated_columns_contains_time_features() -> None:
    from model.features import MODEL_GENERATED_COLUMNS
    assert isinstance(MODEL_GENERATED_COLUMNS, list)
    for col in ("time_idx", "local_idx", "sin_hour", "cos_hour", "sin_doy", "cos_doy"):
        assert col in MODEL_GENERATED_COLUMNS, f"{col} missing from MODEL_GENERATED_COLUMNS"


def test_hrrr_and_obs_sets_are_disjoint() -> None:
    from model.features import HRRR_KNOWN_REALS, OBS_UNKNOWN_REALS
    overlap = set(HRRR_KNOWN_REALS) & set(OBS_UNKNOWN_REALS)
    assert overlap == set(), f"Overlap between HRRR and OBS: {overlap}"


# ---------------------------------------------------------------------------
# Leaf module: no imports from project packages
# ---------------------------------------------------------------------------


def test_features_module_has_no_project_package_imports() -> None:
    src = Path(__file__).parent.parent.parent / "model" / "features.py"
    tree = ast.parse(src.read_text())
    project_packages = {"config", "core", "data", "db", "events", "loops", "model", "snipers", "ui"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in project_packages, (
                    f"model/features.py must not import from '{top}'"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                assert top not in project_packages, (
                    f"model/features.py must not import from '{top}'"
                )
