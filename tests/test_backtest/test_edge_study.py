from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from backtest.forecast_cache import ForecastCache
from backtest.edge_study import (
    build_edge_study_report,
    build_edge_observations,
    cross_fit_edge_calibration,
    fit_edge_calibrator,
    load_edge_calibrator,
    run_diagnostic_replay,
    save_edge_study_artifact,
)
from backtest.study import BacktestStudy
from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset
from tests.test_backtest.test_study import _study_inputs


def test_edge_observations_bind_point_in_time_signal_truth_and_execution(
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

    observations = build_edge_observations(
        study.validate(record.study_id),
        risk_usd=50.0,
    )

    assert len(observations) == 1
    row = observations.iloc[0]
    assert row["event_id"] == "event"
    assert row["condition_id"] == "low"
    assert row["yes_probability"] == 0.8
    assert row["yes_outcome"] == 1.0
    assert row["forecast_width_f"] == 10.0
    assert row["lead_hours"] > 0.0
    assert row["yes_effective_price"] >= row["yes_vwap"]
    assert row["no_effective_price"] >= row["no_vwap"]


def test_edge_observations_never_leave_the_study_target_range(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path),
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    validation = study.validate(record.study_id)
    next_event = validation.clob.events.iloc[[0]].assign(
        event_id="event-next",
        target_date="2026-04-02",
        event_end_at="2026-04-03T00:00:00Z",
    )
    next_market = validation.clob.markets.iloc[[0]].assign(
        event_id="event-next",
        condition_id="next",
        yes_token_id="next-yes",
        no_token_id="next-no",
    )
    next_snapshot = validation.forecast.snapshots.iloc[[0]].assign(
        snapshot_id="snapshot-next",
        init_time="2026-04-02T11:00:00Z",
        target_date="2026-04-02",
        available_at="2026-04-02T12:05:00Z",
    )
    next_probability = validation.forecast.probabilities.iloc[[0]].assign(
        snapshot_id="snapshot-next",
        condition_id="next",
    )
    next_quote = validation.quotes.quotes.iloc[[0]].assign(
        condition_id="next",
        timestamp="2026-04-02T12:05:00Z",
    )
    expanded = replace(
        validation,
        forecast=ForecastCache(
            snapshots=pd.concat(
                [validation.forecast.snapshots, next_snapshot],
                ignore_index=True,
            ),
            probabilities=pd.concat(
                [validation.forecast.probabilities, next_probability],
                ignore_index=True,
            ),
            lineages=validation.forecast.lineages,
        ),
        quotes=SyntheticQuoteDataset(
            quotes=pd.concat(
                [validation.quotes.quotes, next_quote],
                ignore_index=True,
            ),
            start_at=validation.quotes.start_at,
            end_at="2026-04-02T12:05:00Z",
            decision_tick=validation.quotes.decision_tick,
            lookback=validation.quotes.lookback,
            max_staleness=validation.quotes.max_staleness,
        ),
        clob=HistoricalClobDataset(
            events=pd.concat(
                [validation.clob.events, next_event],
                ignore_index=True,
            ),
            markets=pd.concat(
                [validation.clob.markets, next_market],
                ignore_index=True,
            ),
            trades=validation.clob.trades,
            start_date=validation.clob.start_date,
            end_date="2026-04-02",
            station_id=validation.clob.station_id,
        ),
    )

    observations = build_edge_observations(expanded)

    assert set(observations["target_date"]) == {"2026-04-01"}


def test_edge_observations_use_latest_forecast_available_at_each_tick(
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
    later_snapshot = validation.forecast.snapshots.iloc[[0]].assign(
        snapshot_id="snapshot-later",
        init_time="2026-04-01T12:00:00Z",
        available_at="2026-04-01T12:10:00Z",
    )
    later_probability = validation.forecast.probabilities.iloc[[0]].assign(
        snapshot_id="snapshot-later",
        yes_probability=0.6,
    )
    later_quote = validation.quotes.quotes.iloc[[0]].assign(
        timestamp="2026-04-01T12:10:00Z",
    )
    expanded = replace(
        validation,
        forecast=ForecastCache(
            snapshots=pd.concat(
                [validation.forecast.snapshots, later_snapshot],
                ignore_index=True,
            ),
            probabilities=pd.concat(
                [validation.forecast.probabilities, later_probability],
                ignore_index=True,
            ),
            lineages=validation.forecast.lineages,
        ),
        quotes=SyntheticQuoteDataset(
            quotes=pd.concat(
                [validation.quotes.quotes, later_quote],
                ignore_index=True,
            ),
            start_at=validation.quotes.start_at,
            end_at="2026-04-01T12:10:00Z",
            decision_tick=validation.quotes.decision_tick,
            lookback=validation.quotes.lookback,
            max_staleness=validation.quotes.max_staleness,
        ),
    )

    observations = build_edge_observations(expanded)

    assert observations["yes_probability"].tolist() == [0.8, 0.6]


def test_edge_calibration_weights_events_equally_and_falls_back_globally() -> None:
    observations = pd.DataFrame(
        [
            {
                "event_id": "many-ticks",
                "target_date": "2026-04-01",
                "yes_probability": 0.4,
                "yes_outcome": 1.0,
                "yes_effective_price": 0.4,
                "lead_hours": 8.0,
                "forecast_width_f": 5.0,
            }
            for _ in range(10)
        ]
        + [
            {
                "event_id": "one-tick",
                "target_date": "2026-04-02",
                "yes_probability": 0.4,
                "yes_outcome": 0.0,
                "yes_effective_price": 0.4,
                "lead_hours": 8.0,
                "forecast_width_f": 5.0,
            }
        ]
    )

    first = fit_edge_calibrator(
        observations,
        bootstrap_samples=200,
        seed=42,
    )
    second = fit_edge_calibrator(
        observations,
        bootstrap_samples=200,
        seed=42,
    )
    unseen = pd.DataFrame(
        [
            {
                "yes_probability": 0.4,
                "yes_effective_price": 0.95,
                "lead_hours": 100.0,
                "forecast_width_f": 99.0,
            }
        ]
    )
    calibrated = first.calibrate(unseen).iloc[0]

    assert first.global_correction == pytest.approx(0.1)
    assert calibrated["calibration_correction"] == pytest.approx(0.1)
    assert calibrated["calibrated_yes_probability"] == pytest.approx(0.5)
    assert first.to_dict() == second.to_dict()


def test_cross_fit_calibration_holds_out_contiguous_event_days() -> None:
    observations = pd.DataFrame(
        [
            {
                "event_id": f"event-{day}",
                "target_date": f"2026-04-{day:02d}",
                "yes_probability": 0.4,
                "yes_outcome": float(day % 2),
                "yes_effective_price": 0.4,
                "lead_hours": 8.0,
                "forecast_width_f": float(day),
            }
            for day in range(1, 9)
        ]
    )

    result = cross_fit_edge_calibration(
        observations,
        folds=4,
        bootstrap_samples=50,
        seed=42,
    )

    assert len(result.observations) == len(observations)
    for row in result.folds.itertuples(index=False):
        assert set(row.fit_dates).isdisjoint(row.held_out_dates)
        assert list(row.held_out_dates) == sorted(row.held_out_dates)
    assert result.observations["calibration_fold"].notna().all()


def test_diagnostic_replay_enters_each_event_once_at_fixed_risk() -> None:
    rows = [
        {
            "timestamp": "2026-04-01T10:00:00Z",
            "event_id": "event-1",
            "target_date": "2026-04-01",
            "condition_id": "a",
            "yes_token_id": "a-yes",
            "no_token_id": "a-no",
            "yes_outcome": 1.0,
            "yes_probability_lower": 0.50,
            "yes_probability_upper": 0.55,
            "yes_vwap": 0.50,
            "yes_fee_usd": 0.0,
            "yes_effective_price": 0.50,
            "no_vwap": 0.50,
            "no_fee_usd": 0.0,
            "no_effective_price": 0.50,
        },
        {
            "timestamp": "2026-04-01T10:05:00Z",
            "event_id": "event-1",
            "target_date": "2026-04-01",
            "condition_id": "a",
            "yes_token_id": "a-yes",
            "no_token_id": "a-no",
            "yes_outcome": 1.0,
            "yes_probability_lower": 0.65,
            "yes_probability_upper": 0.70,
            "yes_vwap": 0.50,
            "yes_fee_usd": 0.0,
            "yes_effective_price": 0.50,
            "no_vwap": 0.50,
            "no_fee_usd": 0.0,
            "no_effective_price": 0.50,
        },
        {
            "timestamp": "2026-04-01T10:05:00Z",
            "event_id": "event-1",
            "target_date": "2026-04-01",
            "condition_id": "b",
            "yes_token_id": "b-yes",
            "no_token_id": "b-no",
            "yes_outcome": 0.0,
            "yes_probability_lower": 0.80,
            "yes_probability_upper": 0.85,
            "yes_vwap": 0.50,
            "yes_fee_usd": 0.0,
            "yes_effective_price": 0.50,
            "no_vwap": 0.50,
            "no_fee_usd": 0.0,
            "no_effective_price": 0.50,
        },
        {
            "timestamp": "2026-04-02T10:00:00Z",
            "event_id": "event-2",
            "target_date": "2026-04-02",
            "condition_id": "c",
            "yes_token_id": "c-yes",
            "no_token_id": "c-no",
            "yes_outcome": 0.0,
            "yes_probability_lower": 0.10,
            "yes_probability_upper": 0.20,
            "yes_vwap": 0.50,
            "yes_fee_usd": 0.0,
            "yes_effective_price": 0.50,
            "no_vwap": 0.50,
            "no_fee_usd": 0.0,
            "no_effective_price": 0.50,
        },
    ]

    result = run_diagnostic_replay(
        pd.DataFrame(rows),
        min_edge=0.04,
        risk_usd=50.0,
    )

    assert result.trades.groupby("event_id").size().max() == 1
    assert result.trades["event_id"].tolist() == ["event-1", "event-2"]
    assert result.trades["condition_id"].tolist() == ["b", "c"]
    assert result.trades["size_usd"].tolist() == [50.0, 50.0]
    assert set(result.trades["side"]) == {"YES", "NO"}


def test_edge_report_separates_signal_quality_from_money_and_gates_paper(
    tmp_path,
) -> None:
    observations = pd.DataFrame(
        [
            {
                "timestamp": f"2026-04-{day:02d}T10:{minute:02d}:00Z",
                "event_id": f"event-{day}-{index}",
                "target_date": f"2026-04-{day:02d}",
                "condition_id": f"condition-{day}-{index}",
                "yes_token_id": f"yes-{day}-{index}",
                "no_token_id": f"no-{day}-{index}",
                "yes_probability": 0.8,
                "calibrated_yes_probability": 0.8,
                "yes_probability_lower": 0.7,
                "yes_probability_upper": 0.85,
                "yes_outcome": 1.0,
                "yes_vwap": 0.5,
                "yes_fee_usd": 0.0,
                "yes_effective_price": 0.5,
                "no_vwap": 0.5,
                "no_fee_usd": 0.0,
                "no_effective_price": 0.5,
                "calibration_cell": "2|1|1",
            }
            for day in range(1, 11)
            for index, minute in enumerate((0, 5))
        ]
    )
    replay = run_diagnostic_replay(observations)

    report = build_edge_study_report(
        observations,
        replay,
        bootstrap_samples=200,
        seed=42,
    )

    assert report.signal_summary["traded_events"] == 20
    assert report.signal_summary["realized_edge_lcb_90"] > 0.0
    assert report.money_summary["net_pnl_usd"] == pytest.approx(
        replay.settlements["realized_pnl_usd"].sum()
    )
    assert report.money_summary["event_day_pnl_lcb_90"] > 0.0
    assert report.paper_eligible
    calibrator = fit_edge_calibrator(
        observations.assign(
            yes_effective_price=0.5,
            lead_hours=8.0,
            forecast_width_f=5.0,
        ),
        bootstrap_samples=50,
    )
    manifest = save_edge_study_artifact(
        tmp_path / "artifact",
        study_id="edge-study",
        split_role="oof_development",
        input_sha256={"clob_manifest": "a" * 64},
        observations=observations,
        calibrator=calibrator,
        replay=replay,
        report=report,
    )

    assert manifest["study_id"] == "edge-study"
    assert manifest["input_sha256"]["clob_manifest"] == "a" * 64
    assert load_edge_calibrator(tmp_path / "artifact").to_dict() == (
        calibrator.to_dict()
    )
