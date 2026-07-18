from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.mc_market_edge import (
    horizon_band,
    score_market_edge_panel,
    selected_fold_ids,
)
from scripts.run_mc_market_edge_attribution import _evaluation_panels
from scripts.run_mc_market_edge_attribution import _tune_cache_source_hashes
from training.historical_market_panel import filter_evaluation_prediction_window


def _panel(*, model: list[float], midpoint: list[float], truth: int) -> pd.DataFrame:
    rows = []
    for order, (model_p, midpoint_p) in enumerate(zip(model, midpoint)):
        rows.append({
            "event_id": "event-1",
            "target_date": "2025-04-01",
            "timestamp": pd.Timestamp("2025-04-01T12:00:00Z"),
            "snapshot_id": "snapshot-1",
            "bucket_order": order,
            "model_probability": model_p,
            "midpoint_probability": midpoint_p,
            "yes_outcome": float(order == truth),
            "coverage_type": "observed_plus_forecast",
            "forecast_hour_count": 12,
        })
    return pd.DataFrame(rows)


def test_market_edge_identity_is_exactly_zero_against_itself() -> None:
    panel = _panel(model=[0.1, 0.7, 0.2], midpoint=[0.1, 0.7, 0.2], truth=1)

    scores, summary = score_market_edge_panel(
        panel,
        candidate_probability=panel["model_probability"],
        bootstrap_samples=50,
    )

    assert scores["normalized_rps_delta"].iloc[0] == pytest.approx(0.0)
    assert scores["multiclass_brier_delta"].iloc[0] == pytest.approx(0.0)
    assert scores["log_score_delta"].iloc[0] == pytest.approx(0.0)
    assert summary["event_day_equal"]["normalized_rps_delta"] == pytest.approx(0.0)


def test_market_edge_uses_event_day_equal_vectors() -> None:
    good = _panel(model=[1.0, 0.0], midpoint=[0.0, 1.0], truth=0)
    bad = _panel(model=[0.0, 1.0], midpoint=[1.0, 0.0], truth=0).assign(
        event_id="event-2", target_date="2025-04-02"
    )
    # A second tick on event-1 must not make that day weigh twice as much.
    duplicated_good = good.assign(timestamp=pd.Timestamp("2025-04-01T12:05:00Z"))
    panel = pd.concat([good, duplicated_good, bad], ignore_index=True)

    _, summary = score_market_edge_panel(
        panel,
        candidate_probability=panel["model_probability"],
        bootstrap_samples=50,
    )

    assert summary["event_day_equal"]["normalized_rps_delta"] == pytest.approx(0.0)


def test_frozen_mode_excludes_fold_zero_and_oracle_keeps_every_fold() -> None:
    folds = ("fold-00", "fold-01", "fold-02")
    assert selected_fold_ids("oracle_market_edge", folds) == folds
    assert selected_fold_ids("frozen_market_edge", folds) == ("fold-01", "fold-02")
    with pytest.raises(ValueError, match="selection mode"):
        selected_fold_ids("unknown", folds)


def test_horizon_bands_use_remaining_forecast_hours() -> None:
    assert [horizon_band(value) for value in (1, 6, 7, 12, 13, 18, 19, 24)] == [
        "1-6h", "1-6h", "7-12h", "7-12h", "13-18h", "13-18h", "19-24h", "19-24h"
    ]
    assert horizon_band(25) == "19-25h"


def test_static_signal_snapshots_are_assigned_to_their_lineage_fold() -> None:
    panel = _panel(model=[0.5, 0.5], midpoint=[0.5, 0.5], truth=0)
    snapshots = pd.DataFrame({
        "snapshot_id": ["snapshot-1"],
        "lineage_id": ["checkpoint-aligned-static-mc-fold-00"],
        "coverage_type": ["forecast_only"],
        "forecast_hour_count": [24],
    })

    assigned = _evaluation_panels(panel, snapshots)

    assert list(assigned) == ["fold-00"]
    assert "fold_id" not in assigned["fold-00"]


def test_historical_panel_drops_crossfit_tail_before_building_snapshots() -> None:
    frame = pd.DataFrame({
        "valid_time": pd.to_datetime([
            "2025-06-01T04:00:00Z",  # right-closed local target: May 31
            "2025-06-01T05:00:00Z",  # right-closed local target: June 1
        ]),
        "value": [1, 2],
    })

    filtered = filter_evaluation_prediction_window(
        frame,
        predict=("2025-04-01", "2025-05-31"),
        station_timezone="America/New_York",
    )

    assert filtered["value"].tolist() == [1]


def test_frozen_tune_cache_binds_the_same_clob_manifest_as_its_market_panel(
    tmp_path,
) -> None:
    import argparse

    roots = {}
    for name in (
        "crossfit_root", "tune_predictions_root", "observations_root",
        "availability_root", "clob_root", "quotes_root",
    ):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(name, encoding="ascii")
        roots[name] = root
    actuals = tmp_path / "actuals.csv"
    actuals.write_text("target_date,actual_high_f\n", encoding="ascii")
    actuals.with_suffix(".csv.manifest.json").write_text("actuals", encoding="ascii")

    hashes = _tune_cache_source_hashes(
        argparse.Namespace(**roots, actual_highs=actuals)
    )

    assert hashes["clob_manifest"]
    assert hashes["quotes_manifest"]
    assert hashes["actuals_manifest"]
