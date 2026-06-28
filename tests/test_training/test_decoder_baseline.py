from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.storage.parquet_store import ParquetStore
from training.decoder_baseline import build_month_baseline, verify_decoder_baseline


def _decoder_rows(
    init: pd.Timestamp,
    errors: list[float],
    *,
    extrapolated: set[int] | None = None,
) -> pd.DataFrame:
    extrapolated = extrapolated or set()
    return pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 10.0 + error,
                "is_extrapolated": fxx in extrapolated,
            }
            for fxx, error in enumerate(errors, start=1)
        ]
    )


def _obs_for_decoder(decoder: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time": decoder["valid_time"],
            "raw_tmpc_tenths": 10.0,
        }
    )


def test_month_baseline_aligns_cells_and_computes_per_init_error_metrics():
    init = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    errors = [1.0, -1.0, *([0.0] * 46)]
    decoder = _decoder_rows(init, errors)

    result = build_month_baseline(
        decoder,
        _obs_for_decoder(decoder),
        station_id="KLGA",
        month="2026-01",
    )

    assert result.cells["forecast_error_c"].tolist() == pytest.approx(errors)
    assert result.cells["absolute_error_c"].tolist() == pytest.approx(
        np.abs(errors)
    )
    row = result.per_init.iloc[0]
    assert row["data_status"] == "complete"
    assert row["matched_steps"] == 48
    assert row["bias_c"] == pytest.approx(0.0)
    assert row["mae_c"] == pytest.approx(2.0 / 48.0)
    assert row["rmse_c"] == pytest.approx(np.sqrt(2.0 / 48.0))
    assert row["error_variance_c2"] == pytest.approx(2.0 / 48.0)
    assert row["error_std_c"] == pytest.approx(np.sqrt(2.0 / 48.0))


def test_month_baseline_separates_invalid_decoder_from_incomplete_observations():
    missing_init = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    duplicate_init = pd.Timestamp("2026-01-10 00:00", tz="UTC")
    incomplete_init = pd.Timestamp("2026-01-20 00:00", tz="UTC")
    nan_init = pd.Timestamp("2026-01-25 00:00", tz="UTC")
    missing = _decoder_rows(missing_init, [0.0] * 47)
    duplicate = _decoder_rows(duplicate_init, [0.0] * 48)
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    incomplete = _decoder_rows(incomplete_init, [0.0] * 48)
    nan_decoder = _decoder_rows(nan_init, [0.0] * 48)
    nan_decoder.loc[nan_decoder["fxx"] == 12, "TMP_2m"] = np.nan
    decoder = pd.concat(
        [missing, duplicate, incomplete, nan_decoder],
        ignore_index=True,
    )
    observations = _obs_for_decoder(decoder).drop_duplicates("valid_time")
    observations = observations[
        observations["valid_time"] != incomplete_init + pd.Timedelta(hours=48)
    ]

    result = build_month_baseline(
        decoder,
        observations,
        station_id="KLGA",
        month="2026-01",
    )
    statuses = result.per_init.set_index("init_time")

    assert statuses.loc[missing_init, "data_status"] == "invalid_decoder"
    assert statuses.loc[missing_init, "missing_forecast_steps"] == 1
    assert statuses.loc[duplicate_init, "data_status"] == "invalid_decoder"
    assert statuses.loc[duplicate_init, "duplicate_steps"] == 1
    assert (
        statuses.loc[incomplete_init, "data_status"]
        == "incomplete_observations"
    )
    assert statuses.loc[incomplete_init, "missing_observation_steps"] == 1
    assert statuses.loc[nan_init, "data_status"] == "invalid_decoder"
    assert statuses.loc[nan_init, "invalid_forecast_steps"] == 1
    assert not statuses["performance_outlier"].any()


def test_month_baseline_flags_month_relative_outlier_and_excludes_incomplete_init():
    base = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    frames = [
        _decoder_rows(base + pd.Timedelta(days=i), [0.5 + i * 0.1] * 48)
        for i in range(10)
    ]
    outlier_init = base + pd.Timedelta(days=12)
    incomplete_init = base + pd.Timedelta(days=20)
    frames.extend(
        [
            _decoder_rows(outlier_init, [5.0] * 48),
            _decoder_rows(incomplete_init, [10.0] * 48),
        ]
    )
    decoder = pd.concat(frames, ignore_index=True)
    observations = _obs_for_decoder(decoder).drop_duplicates("valid_time")
    observations = observations[
        observations["valid_time"] != incomplete_init + pd.Timedelta(hours=48)
    ]

    result = build_month_baseline(
        decoder,
        observations,
        station_id="KLGA",
        month="2026-01",
    )
    per_init = result.per_init.set_index("init_time")

    assert per_init.loc[outlier_init, "mae_robust_z"] > 3.5
    assert per_init.loc[outlier_init, "performance_outlier"]
    assert pd.isna(per_init.loc[incomplete_init, "mae_robust_z"])
    assert not per_init.loc[incomplete_init, "performance_outlier"]
    assert result.summary["performance_outlier_count"] == 1


def test_month_baseline_does_not_flag_outliers_when_monthly_mad_is_zero():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    decoder = pd.concat(
        [
            _decoder_rows(base + pd.Timedelta(days=i), [1.0] * 48)
            for i in range(3)
        ],
        ignore_index=True,
    )

    result = build_month_baseline(
        decoder,
        _obs_for_decoder(decoder).drop_duplicates("valid_time"),
        station_id="KLGA",
        month="2026-01",
    )

    assert result.per_init["mae_robust_z"].isna().all()
    assert not result.per_init["performance_outlier"].any()


def test_month_baseline_marks_missing_temperature_column_as_invalid_decoder():
    init = pd.Timestamp("2026-01-01", tz="UTC")
    decoder = _decoder_rows(init, [0.0] * 48).drop(columns="TMP_2m")

    result = build_month_baseline(
        decoder,
        _obs_for_decoder(_decoder_rows(init, [0.0] * 48)),
        station_id="KLGA",
        month="2026-01",
    )

    assert result.per_init["data_status"].tolist() == ["invalid_decoder"]
    assert result.per_init["invalid_forecast_steps"].tolist() == [48]
    assert result.summary["invalid_decoder_init_count"] == 1


def test_month_baseline_marks_month_relative_outlier_forecast_hours():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    errors = [0.1 * fxx for fxx in range(1, 49)]
    errors[-1] = 20.0
    decoder = pd.concat(
        [
            _decoder_rows(base + pd.Timedelta(hours=i), errors)
            for i in range(10)
        ],
        ignore_index=True,
    )

    result = build_month_baseline(
        decoder,
        _obs_for_decoder(decoder).drop_duplicates("valid_time"),
        station_id="KLGA",
        month="2026-01",
    )
    by_fxx = result.by_fxx.set_index("fxx")

    assert by_fxx.loc[48, "mae_robust_z"] > 3.5
    assert by_fxx.loc[48, "performance_outlier"]
    assert not by_fxx.loc[1:47, "performance_outlier"].any()


def test_verify_decoder_baseline_writes_month_artifacts_and_is_rerunnable(tmp_path):
    run_dir = tmp_path / "run"
    decoder_dir = run_dir / "data_store" / "_decoder_cache" / "KLGA"
    decoder_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-01-31 23:00", tz="UTC")
    decoder = _decoder_rows(init, [0.5] * 48, extrapolated={48})
    decoder.to_parquet(decoder_dir / "2026-01.parquet", index=False)
    ParquetStore(run_dir / "data_store" / "obs").append(
        "KLGA",
        _obs_for_decoder(decoder),
    )

    first = verify_decoder_baseline(run_dir, "KLGA", month="2026-01")
    second = verify_decoder_baseline(run_dir, "KLGA", month="2026-01")

    baseline_dir = run_dir / "artifacts" / "decoder_baseline" / "KLGA"
    assert (baseline_dir / "cells" / "2026-01.parquet").exists()
    assert (baseline_dir / "per_init" / "2026-01.parquet").exists()
    assert (baseline_dir / "by_fxx" / "2026-01.parquet").exists()
    cells = pd.read_parquet(baseline_dir / "cells" / "2026-01.parquet")
    assert cells.loc[cells["fxx"] == 48, "source_kind"].iloc[0] == "extrapolated"
    by_fxx = pd.read_parquet(baseline_dir / "by_fxx" / "2026-01.parquet")
    assert by_fxx["fxx"].tolist() == list(range(1, 49))
    assert by_fxx["sample_count"].eq(1).all()
    monthly = pd.read_parquet(baseline_dir / "monthly_summary.parquet")
    assert monthly["month"].tolist() == ["2026-01"]
    assert monthly["complete_init_count"].tolist() == [1]
    manifest = pd.read_json(baseline_dir / "manifest.json", typ="series")
    assert manifest["schema_version"] == 2
    assert manifest["error_definition"] == "forecast_c - observed_c"
    assert "2026-01" in manifest["input_fingerprints"]
    assert first["processed_months"] == ["2026-01"]
    assert second["processed_months"] == ["2026-01"]
