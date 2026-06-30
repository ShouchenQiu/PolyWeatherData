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
        TFT_KNOWN_REALS,
        TFT_UNKNOWN_REALS,
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


def test_history_target_feature_is_encoder_only_copy() -> None:
    import pandas as pd

    from model.features import (
        HISTORICAL_TARGET_FEATURE,
        add_decoder_history_features,
        add_encoder_history_features,
    )

    encoder = add_encoder_history_features(
        pd.DataFrame({"temp_error": [1.25, -0.5]})
    )
    decoder = add_decoder_history_features(
        pd.DataFrame({"temp_error": [99.0, -99.0]})
    )

    assert HISTORICAL_TARGET_FEATURE == "hist_temp_error"
    assert encoder[HISTORICAL_TARGET_FEATURE].tolist() == [1.25, -0.5]
    assert decoder[HISTORICAL_TARGET_FEATURE].tolist() == [0.0, 0.0]


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
    assert REQUIRED_DECODER_SOURCE_COLUMNS is not HRRR_KNOWN_REALS


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
# TFT_KNOWN_REALS / TFT_UNKNOWN_REALS
# ---------------------------------------------------------------------------


def test_tft_known_reals_starts_with_time_encoding_then_hrrr() -> None:
    from model.features import HRRR_KNOWN_REALS, TFT_KNOWN_REALS
    for col in ("local_idx", "sin_hour", "cos_hour", "sin_doy", "cos_doy"):
        assert col in TFT_KNOWN_REALS, f"{col} missing from TFT_KNOWN_REALS"
    for col in HRRR_KNOWN_REALS:
        assert col in TFT_KNOWN_REALS, f"{col} missing from TFT_KNOWN_REALS"


def test_tft_unknown_reals_adds_history_target_to_observations() -> None:
    from model.features import (
        HISTORICAL_TARGET_FEATURE,
        OBS_UNKNOWN_REALS,
        TARGET,
        TFT_UNKNOWN_REALS,
    )
    assert TFT_UNKNOWN_REALS == OBS_UNKNOWN_REALS + [HISTORICAL_TARGET_FEATURE]
    assert TFT_UNKNOWN_REALS is not OBS_UNKNOWN_REALS
    assert TARGET not in TFT_UNKNOWN_REALS


def test_tft_known_reals_has_no_duplicates() -> None:
    from model.features import TFT_KNOWN_REALS
    assert len(TFT_KNOWN_REALS) == len(set(TFT_KNOWN_REALS))


# ---------------------------------------------------------------------------
# Smoke tests: tft.py and train_tft.py use the same contract
# ---------------------------------------------------------------------------


def test_tft_known_reals_matches_model_tft_known_reals() -> None:
    from model.features import TFT_KNOWN_REALS
    from model.tft import KNOWN_REALS
    assert KNOWN_REALS == TFT_KNOWN_REALS


def test_tft_unknown_reals_matches_model_tft_unknown_reals() -> None:
    from model.features import TFT_UNKNOWN_REALS
    from model.tft import UNKNOWN_REALS
    assert UNKNOWN_REALS == TFT_UNKNOWN_REALS


def test_train_tft_known_reals_matches_tft_contract() -> None:
    from model.features import TFT_KNOWN_REALS
    from training.trainer import _KNOWN_REALS
    assert _KNOWN_REALS == TFT_KNOWN_REALS


def test_train_tft_unknown_reals_matches_tft_contract() -> None:
    from model.features import TFT_UNKNOWN_REALS
    from training.trainer import _UNKNOWN_REALS
    assert _UNKNOWN_REALS == TFT_UNKNOWN_REALS


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
