from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model.mc_calibration import McCalibration
from ui.model_evaluation import (
    daily_target_summary,
    load_daily_high_pmf_pilot,
    load_peak_union_diagnostics,
    discover_evaluations,
    evaluation_metrics,
    load_evaluation,
    model_performance_table,
)
from ui.pages import model_evaluation


def _complete_evaluation(tmp_path):
    version_dir = tmp_path / "runs" / "run-1" / "checkpoints" / "a-v1"
    version_dir.mkdir(parents=True)
    checkpoint = version_dir / "tft_best.ckpt"
    checkpoint.write_bytes(b"model")
    predictions = pd.DataFrame(
        {
            "init_time": pd.to_datetime(["2026-04-01T00:00:00Z"] * 2),
            "valid_time": pd.to_datetime(
                ["2026-04-01T01:00:00Z", "2026-04-01T02:00:00Z"]
            ),
            "fxx": [1, 2],
            "target": [1.0, -1.0],
            "p10": [0.0, -2.0],
            "p50": [0.5, -0.5],
            "p90": [1.0, 0.0],
            "base_temp_c": [10.0, 12.0],
            "weight": [1.0, 1.0],
        }
    )
    predictions.to_parquet(version_dir / "validation_predictions.parquet")
    pd.DataFrame(
        {
            "init_time": pd.to_datetime(["2026-04-01T00:00:00Z"]),
            "target_date": ["2026-04-01"],
            "actual_high_f": [70.0],
            "hrrr_high_f": [72.0],
            "tft_high_f": [71.0],
            "mc_p10_f": [68.0],
            "mc_p50_f": [70.5],
            "mc_p90_f": [73.0],
            "min_fxx": [1],
            "max_fxx": [24],
        }
    ).to_parquet(version_dir / "mc_calibration_predictions.parquet")
    (version_dir / "mc_calibration_evidence.json").write_text(
        json.dumps({"p10_p90_coverage": 1.0}),
        encoding="ascii",
    )
    McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.ones(5),
        max_bias_f=0.5,
        model_sha256=McCalibration.sha256(checkpoint),
        fit_split="val",
        fit_start="2026-04-01",
        fit_end="2026-04-30",
        sample_count=1,
        station_timezone="America/New_York",
    ).save(version_dir)
    return version_dir


def test_discovers_complete_evaluation_and_computes_hrrr_tft_mc_metrics(
    tmp_path,
) -> None:
    version_dir = _complete_evaluation(tmp_path)

    found = discover_evaluations(tmp_path / "runs")
    data = load_evaluation(found[0])
    metrics = evaluation_metrics(data)

    assert found[0].version == "a-v1"
    assert found[0].path == version_dir
    assert metrics["hrrr_hourly_mae_f"] == pytest.approx(1.8)
    assert metrics["tft_hourly_mae_f"] == pytest.approx(0.9)
    assert metrics["hourly_skill_pct"] == pytest.approx(50.0)
    assert metrics["hrrr_high_mae_f"] == 2.0
    assert metrics["tft_high_mae_f"] == 1.0
    assert metrics["mc_high_mae_f"] == 0.5
    assert metrics["mc_coverage"] == 1.0


def test_daily_target_summary_collapses_cycles_to_one_row_per_target_day() -> None:
    daily = pd.DataFrame(
        {
            "target_date": ["2026-04-02", "2026-04-02", "2026-04-03"],
            "actual_high_f": [70.0, 70.0, 72.0],
            "hrrr_high_f": [68.0, 72.0, 71.0],
            "tft_high_f": [69.0, 73.0, 72.0],
            "mc_p10_f": [66.0, 68.0, 69.0],
            "mc_p50_f": [69.0, 71.0, 72.0],
            "mc_p90_f": [72.0, 74.0, 75.0],
        }
    )

    summary = daily_target_summary(daily)

    assert summary["target_date"].tolist() == list(
        pd.to_datetime(["2026-04-02", "2026-04-03"])
    )
    assert summary["cycle_count"].tolist() == [2, 1]
    assert summary.loc[0, "actual_high_f"] == 70.0
    assert summary.loc[0, "hrrr_high_f"] == 70.0
    assert summary.loc[0, "tft_high_f"] == 71.0
    assert summary.loc[0, "mc_p50_f"] == 70.0


def test_model_evaluation_exposes_strict_april_only_when_artifact_is_complete(tmp_path) -> None:
    project = tmp_path / "project"
    root = (
        project / "data_store" / "backtest" / "preapril_fixed_epoch_mc"
        / "KLGA" / "20260401_20260430_v1"
    )
    root.mkdir(parents=True)

    options = model_evaluation._default_backtest_evaluations(project)
    assert "April 2026 strict pre-April MC" not in options

    (root / "report.json").write_text("{}", encoding="ascii")
    (root / "april_hourly_predictions.parquet").write_bytes(b"artifact")
    (root / "forecast_cache").mkdir()
    (root / "forecast_cache" / "manifest.json").write_text("{}", encoding="ascii")

    options = model_evaluation._default_backtest_evaluations(project)
    april = options["April 2026 strict pre-April MC"]
    assert april["target_start"] == "2026-04-01"
    assert april["target_end"] == "2026-04-30"


def test_model_performance_table_compares_tft_and_mc_with_their_baselines(
    tmp_path,
) -> None:
    version_dir = _complete_evaluation(tmp_path)
    data = load_evaluation(discover_evaluations(tmp_path / "runs")[0])

    table = model_performance_table(data)

    assert table["Model"].tolist() == ["TFT", "MC"]
    assert table["Target"].tolist() == [
        "Hourly temperature",
        "Daily high (OOF cycle-day)",
    ]
    assert table["Baseline"].tolist() == ["HRRR", "TFT"]
    assert table["MAE (°F)"].tolist() == pytest.approx([0.9, 0.5])
    assert table["Skill vs baseline (%)"].tolist() == pytest.approx([50.0, 50.0])
    assert table["Samples"].tolist() == [2, 1]


def test_load_daily_high_pmf_pilot_requires_unconsumed_june_and_tables(tmp_path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "june_consumed": False,
                "gates": {"eligible_for_cross_fit_goal": False},
            }
        ),
        encoding="ascii",
    )
    pd.DataFrame([{"stratum": "overall", "candidate_rps": 0.2, "mc_rps": 0.1}]).to_csv(
        root / "april_summary.csv", index=False
    )
    pd.DataFrame([{"stratum": "overall", "candidate_rps": 0.3, "mc_rps": 0.1}]).to_csv(
        root / "may_summary.csv", index=False
    )
    pd.DataFrame([{"target_date": "2026-05-01", "candidate_rps": 0.3, "mc_rps": 0.1}]).to_parquet(
        root / "may_parallel_scores.parquet", index=False
    )

    pilot = load_daily_high_pmf_pilot(root)

    assert pilot.report["gates"]["eligible_for_cross_fit_goal"] is False
    assert pilot.may_summary.iloc[0]["candidate_rps"] == 0.3


def test_load_peak_union_diagnostics_requires_complete_hashed_artifact(tmp_path) -> None:
    root = tmp_path / "peak-union"
    root.mkdir()
    payloads = {
        "snapshot_diagnostics.parquet": pd.DataFrame([{"snapshot_id": "s1"}]),
        "peak_outcomes.parquet": pd.DataFrame([{"snapshot_id": "s1"}]),
        "union_scores.parquet": pd.DataFrame([{"snapshot_id": "s1", "brier_delta": -0.01}]),
        "union_strata.parquet": pd.DataFrame([{"union_width": 1}]),
    }
    import hashlib
    hashes = {}
    for name, frame in payloads.items():
        path = root / name
        frame.to_parquet(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = root / "report.json"
    report.write_text(json.dumps({"schema_version": 1, "scope": "normalized_synthetic_midpoint_only"}), encoding="ascii")
    hashes[report.name] = hashlib.sha256(report.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "data_class": "peak_union_diagnostics", "sha256": hashes}), encoding="ascii")

    diagnostic = load_peak_union_diagnostics(root)

    assert diagnostic.union_scores.iloc[0]["brier_delta"] == pytest.approx(-0.01)


def test_model_evaluation_page_renders_complete_artifacts(tmp_path) -> None:
    _complete_evaluation(tmp_path)

    with ExitStack() as stack:
        for name in (
            "header",
            "caption",
            "info",
            "warning",
            "subheader",
        ):
            stack.enter_context(patch(f"streamlit.{name}"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(
            patch(
                "streamlit.selectbox",
                side_effect=lambda _label, options, **_kwargs: options[0],
            )
        )
        stack.enter_context(
            patch(
                "streamlit.columns",
                side_effect=lambda n, **_kwargs: [MagicMock() for _ in range(n)],
            )
        )
        stack.enter_context(
            patch(
                "streamlit.tabs",
                side_effect=lambda labels: [MagicMock() for _ in labels],
            )
        )
        chart = stack.enter_context(patch("streamlit.plotly_chart"))

        model_evaluation.render(runs_root=tmp_path / "runs")

    assert chart.call_count >= 7
    figures = [call.args[0] for call in chart.call_args_list]
    figures_by_title = {
        figure.layout.title.text: figure
        for figure in figures
        if figure.layout.title.text
    }
    assert "Daily high forecasts by target day" in figures_by_title
    assert "Daily MC interval and actual high" in figures_by_title
    assert "Daily absolute error by target day" in figures_by_title
    assert "Daily-high absolute-error CDF" not in figures_by_title
    daily_forecasts = figures_by_title["Daily high forecasts by target day"]
    assert {trace.name for trace in daily_forecasts.data} == {
        "Actual",
        "HRRR",
        "TFT",
        "MC",
    }
    performance_tables = [
        call.args[0]
        for call in dataframe.call_args_list
        if isinstance(call.args[0], pd.DataFrame)
        and "Model" in call.args[0].columns
    ]
    assert performance_tables[0]["Model"].tolist() == ["TFT", "MC"]
