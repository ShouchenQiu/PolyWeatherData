from __future__ import annotations

import pandas as pd
import pytest

from backtest.forecast_cache import ForecastCache


def _cache() -> ForecastCache:
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_id": "old",
                "station_id": "KLGA",
                "init_time": "2026-06-05T12:00:00Z",
                "target_date": "2026-06-06",
                "available_at": "2026-06-05T13:55:00Z",
                "p10_f": 70.0,
                "p50_f": 74.0,
                "p90_f": 78.0,
                "lineage_id": "fixed",
            },
            {
                "snapshot_id": "future",
                "station_id": "KLGA",
                "init_time": "2026-06-05T13:00:00Z",
                "target_date": "2026-06-06",
                "available_at": "2026-06-05T14:30:00Z",
                "p10_f": 71.0,
                "p50_f": 75.0,
                "p90_f": 79.0,
                "lineage_id": "fixed",
            },
            {
                "snapshot_id": "other-day",
                "station_id": "KLGA",
                "init_time": "2026-06-05T12:00:00Z",
                "target_date": "2026-06-07",
                "available_at": "2026-06-05T13:55:00Z",
                "p10_f": 68.0,
                "p50_f": 72.0,
                "p90_f": 77.0,
                "lineage_id": "fixed",
            },
        ]
    )
    probabilities = pd.DataFrame(
        [
            {
                "snapshot_id": "old",
                "condition_id": "condition-a",
                "yes_probability": 0.40,
            },
            {
                "snapshot_id": "future",
                "condition_id": "condition-a",
                "yes_probability": 0.55,
            },
        ]
    )
    lineages = pd.DataFrame(
        [
            {
                "lineage_id": "fixed",
                "checkpoint_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
                "split_role": "post_test_holdout",
                "mc_seed": 42,
                "mc_samples": 2000,
            }
        ]
    )
    return ForecastCache(snapshots, probabilities, lineages)


def test_latest_at_hides_future_snapshot_and_keeps_last_available() -> None:
    cache = _cache()

    visible = cache.latest_at(
        "2026-06-05T14:20:00Z",
        station_id="KLGA",
        target_date="2026-06-06",
    )

    assert visible["snapshot_id"].tolist() == ["old"]
    assert cache.probabilities_at(
        "2026-06-05T14:20:00Z",
        station_id="KLGA",
        target_date="2026-06-06",
    )["yes_probability"].tolist() == [0.40]


def test_latest_at_selects_one_snapshot_per_target_day() -> None:
    visible = _cache().latest_at(
        "2026-06-05T14:35:00Z",
        station_id="KLGA",
    )

    assert set(visible["snapshot_id"]) == {"future", "other-day"}


def test_forecast_cache_is_immutable_and_hash_verified(tmp_path) -> None:
    root = tmp_path / "cache"
    cache = _cache()
    manifest = cache.save(root, created_at="2026-07-02T00:00:00Z")

    assert manifest["schema_version"] == 2
    assert manifest["data_class"] == "point_in_time_forecast_cache"
    loaded = ForecastCache.load(root)
    assert loaded.snapshots["snapshot_id"].tolist() == [
        "old",
        "other-day",
        "future",
    ]
    with pytest.raises(FileExistsError):
        cache.save(root)

    with (root / "snapshots.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ForecastCache.load(root)


def test_forecast_cache_rejects_unknown_lineage_and_snapshot() -> None:
    cache = _cache()
    snapshots = cache.snapshots.copy()
    snapshots.loc[0, "lineage_id"] = "missing"
    with pytest.raises(ValueError, match="unknown lineage"):
        ForecastCache(snapshots, cache.probabilities, cache.lineages)

    probabilities = cache.probabilities.copy()
    probabilities.loc[0, "snapshot_id"] = "missing"
    with pytest.raises(ValueError, match="unknown snapshot"):
        ForecastCache(cache.snapshots, probabilities, cache.lineages)
