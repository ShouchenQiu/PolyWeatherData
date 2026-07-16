from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.forecast_cache import ForecastCache
from backtest.signal_study import (
    build_signal_panel,
    load_signal_study_artifact,
    run_forecast_signal_study,
    run_forecast_signal_validation_study,
    score_signal_panel,
)
from backtest.study import BacktestStudy, validate_signal_inputs
from backtest.synthetic_quotes import SyntheticQuoteDataset
from tests.test_backtest.test_study import _study_inputs


def test_signal_panel_requires_complete_vectors_and_normalizes_midpoints(
    tmp_path,
) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path),
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    validation = study.validate(record.study_id)
    high_probability = validation.forecast.probabilities.iloc[[0]].assign(
        condition_id="high",
        yes_probability=0.2,
    )
    incomplete_quote = validation.quotes.quotes.iloc[[0]].assign(
        timestamp="2026-04-01T12:00:00Z",
    )
    high_quote = validation.quotes.quotes.iloc[[0]].assign(
        condition_id="high",
        mid_price_yes=0.7,
    )
    future_high_quote = high_quote.assign(
        timestamp="2026-04-01T12:00:00Z",
    )
    expanded = replace(
        validation,
        forecast=ForecastCache(
            snapshots=validation.forecast.snapshots,
            probabilities=pd.concat(
                [
                    validation.forecast.probabilities,
                    high_probability,
                ],
                ignore_index=True,
            ),
            lineages=validation.forecast.lineages,
        ),
        quotes=SyntheticQuoteDataset(
                quotes=pd.concat(
                    [
                        incomplete_quote,
                        future_high_quote,
                        validation.quotes.quotes,
                        high_quote,
                ],
                ignore_index=True,
            ),
            start_at="2026-04-01T12:00:00Z",
            end_at=validation.quotes.end_at,
            decision_tick=validation.quotes.decision_tick,
            lookback=validation.quotes.lookback,
            max_staleness=validation.quotes.max_staleness,
        ),
    )

    panel = build_signal_panel(expanded)

    assert len(panel) == 2
    assert panel["timestamp"].nunique() == 1
    assert panel["snapshot_id"].nunique() == 1
    assert panel["model_probability"].sum() == pytest.approx(1.0)
    assert panel["midpoint_probability"].sum() == pytest.approx(1.0)
    assert panel["midpoint_overround"].unique() == pytest.approx([0.1])
    assert panel.set_index("condition_id").loc[
        "low", "midpoint_probability"
    ] == pytest.approx(0.4 / 1.1)
    assert panel["yes_outcome"].sum() == 1.0


def test_signal_validation_study_accepts_frozen_model_test(tmp_path, monkeypatch) -> None:
    validation = validate_signal_inputs(
        study_id="frozen-model-test",
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path, split_role="model_test"),
        expected_split_role="model_test",
    )

    expected = {"split_role": "model_test"}
    monkeypatch.setattr(
        "backtest.signal_study._run_forecast_signal_study",
        lambda actual, **kwargs: expected if actual is validation else None,
    )

    manifest = run_forecast_signal_validation_study(
        validation,
        output=tmp_path / "model-test-signal",
        bootstrap_samples=20,
    )

    assert manifest["split_role"] == "model_test"


def _metric_panel(
    *,
    model: list[float],
    midpoint: list[float],
    truth: int,
    event_id: str = "event-1",
    timestamp: str = "2026-04-01T12:00:00Z",
) -> pd.DataFrame:
    rows = []
    for order, (model_p, midpoint_p) in enumerate(zip(model, midpoint)):
        rows.append(
            {
                "event_id": event_id,
                "target_date": "2026-04-01",
                "timestamp": pd.Timestamp(timestamp),
                "condition_id": f"bucket-{order}",
                "bucket_order": order,
                "model_probability": model_p,
                "midpoint_probability": midpoint_p,
                "midpoint_sum": 1.0,
                "midpoint_overround": 0.0,
                "yes_outcome": float(order == truth),
                "lead_band": "12-24h",
            }
        )
    return pd.DataFrame(rows)


def test_signal_scores_match_perfect_uniform_and_ordered_fixtures() -> None:
    perfect = score_signal_panel(
        _metric_panel(
            model=[0.0, 1.0, 0.0],
            midpoint=[1 / 3, 1 / 3, 1 / 3],
            truth=1,
        ),
        bootstrap_samples=100,
        seed=42,
    )

    model = perfect.summary["sources"]["model_raw"]
    midpoint = perfect.summary["sources"]["synthetic_midpoint"]
    assert model["multiclass_brier"] == pytest.approx(0.0, abs=1e-5)
    assert model["log_score"] == pytest.approx(0.0, abs=1e-5)
    assert model["normalized_rps"] == pytest.approx(0.0, abs=1e-5)
    assert model["top1_accuracy"] == 1.0
    assert model["normalized_entropy"] == pytest.approx(0.0, abs=1e-4)
    assert midpoint["multiclass_brier"] == pytest.approx(2 / 3)
    assert midpoint["log_score"] == pytest.approx(-np.log(1 / 3))
    assert midpoint["normalized_rps"] == pytest.approx(1 / 9)
    assert midpoint["normalized_entropy"] == pytest.approx(1.0)

    near = score_signal_panel(
        _metric_panel(
            model=[0.7, 0.2, 0.1],
            midpoint=[0.1, 0.2, 0.7],
            truth=0,
        ),
        bootstrap_samples=100,
        seed=42,
    )
    assert (
        near.summary["sources"]["model_raw"]["normalized_rps"]
        < near.summary["sources"]["synthetic_midpoint"]["normalized_rps"]
    )


def test_signal_summary_is_event_equal_and_bootstrap_reproducible() -> None:
    event_one = _metric_panel(
        model=[0.8, 0.2],
        midpoint=[0.4, 0.6],
        truth=0,
        event_id="event-1",
    )
    event_two = _metric_panel(
        model=[0.1, 0.9],
        midpoint=[0.8, 0.2],
        truth=0,
        event_id="event-2",
    )
    duplicated_ticks = pd.concat(
        [
            event_one,
            event_one.assign(
                timestamp=pd.Timestamp("2026-04-01T12:05:00Z")
            ),
            event_two,
        ],
        ignore_index=True,
    )

    first = score_signal_panel(
        duplicated_ticks, bootstrap_samples=200, seed=42
    )
    second = score_signal_panel(
        duplicated_ticks, bootstrap_samples=200, seed=42
    )

    expected = (0.08 + 1.62) / 2
    assert first.summary["sources"]["model_raw"][
        "multiclass_brier"
    ] == pytest.approx(expected)
    assert first.summary["comparison"] == second.summary["comparison"]
    comparison = first.summary["comparison"]["multiclass_brier"]
    assert comparison["ci90_low"] <= comparison["model_minus_midpoint"]
    assert comparison["ci90_high"] == pytest.approx(
        comparison["model_minus_midpoint"]
    )


def test_signal_artifact_is_complete_and_hash_verified(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path),
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    validation = study.validate(record.study_id)
    validation = replace(
        validation,
        forecast=ForecastCache(
            snapshots=validation.forecast.snapshots,
            probabilities=pd.concat(
                [
                    validation.forecast.probabilities,
                    validation.forecast.probabilities.iloc[[0]].assign(
                        condition_id="high",
                        yes_probability=0.2,
                    ),
                ],
                ignore_index=True,
            ),
            lineages=validation.forecast.lineages,
        ),
        quotes=SyntheticQuoteDataset(
            quotes=pd.concat(
                [
                    validation.quotes.quotes,
                    validation.quotes.quotes.iloc[[0]].assign(
                        condition_id="high",
                        mid_price_yes=0.6,
                    ),
                ],
                ignore_index=True,
            ),
            start_at=validation.quotes.start_at,
            end_at=validation.quotes.end_at,
            decision_tick=validation.quotes.decision_tick,
            lookback=validation.quotes.lookback,
            max_staleness=validation.quotes.max_staleness,
        ),
    )
    output = tmp_path / "signal"

    manifest = run_forecast_signal_study(
        validation,
        output=output,
        bootstrap_samples=20,
    )
    artifact = load_signal_study_artifact(output)

    assert manifest["data_class"] == "forecast_signal_study"
    assert manifest["split_role"] == "oof_development"
    assert len(artifact.panel) == 2
    assert artifact.summary["sample"]["requested_event_count"] == 1
    assert artifact.summary["sample"]["excluded_event_count"] == 0
    assert set(artifact.by_lead["lead_band"].astype(str)) == {
        "<6h",
        "6-12h",
        "12-24h",
        ">=24h",
    }
    assert set(manifest["sha256"]) == {
        "panel.parquet",
        "scores_by_tick.parquet",
        "by_event.parquet",
        "by_lead.parquet",
        "reliability.parquet",
        "summary.json",
    }

    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_signal_study_artifact(output)
