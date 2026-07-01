from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model.mc_calibration import McCalibration
from ui.model_evaluation import (
    discover_evaluations,
    evaluation_metrics,
    load_evaluation,
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


def test_model_evaluation_page_renders_complete_artifacts(tmp_path) -> None:
    _complete_evaluation(tmp_path)

    with ExitStack() as stack:
        for name in (
            "header",
            "caption",
            "info",
            "warning",
            "subheader",
            "dataframe",
        ):
            stack.enter_context(patch(f"streamlit.{name}"))
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
