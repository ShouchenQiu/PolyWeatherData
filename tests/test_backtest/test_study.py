from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from backtest.forecast_cache import ForecastCache
from backtest.study import (
    BacktestStudy,
    StudyInputs,
    validate_signal_inputs,
    validate_study_inputs,
)
from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset
from data.clob.quality import audit_historical_clob


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _study_inputs(tmp_path, *, split_role: str = "oof_development") -> StudyInputs:
    clob_root = tmp_path / f"clob-{split_role}"
    clob = HistoricalClobDataset(
        events=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "target_date": "2026-04-01",
                    "station_id": "KLGA",
                    "event_end_at": "2026-04-02T00:00:00Z",
                }
            ]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "condition_id": "low",
                    "yes_token_id": "low-yes",
                    "no_token_id": "low-no",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": 70.0,
                },
                {
                    "event_id": "event",
                    "condition_id": "high",
                    "yes_token_id": "high-yes",
                    "no_token_id": "high-no",
                    "bucket_lo_f": 70.0,
                    "bucket_hi_f": float("inf"),
                },
            ]
        ),
        trades=pd.DataFrame(
            [
                {
                    "condition_id": "low",
                    "token_id": "low-yes",
                    "timestamp": "2026-04-01T12:00:00Z",
                    "outcome": "YES",
                    "side": "BUY",
                    "price": 0.4,
                    "size": 10.0,
                    "usdc": 4.0,
                }
            ]
        ),
        start_date="2026-04-01",
        end_date="2026-04-01",
        station_id="KLGA",
    )
    clob.save(clob_root, fetched_at="2026-07-01T00:00:00Z")
    clob_digest = _sha256(clob_root / "manifest.json")
    quality_path = clob_root.with_suffix(".quality.json")
    quality = audit_historical_clob(clob, exclusions={})
    quality["source_manifest_sha256"] = clob_digest
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    quotes_root = tmp_path / f"quotes-{split_role}"
    SyntheticQuoteDataset(
        quotes=pd.DataFrame(
            [
                {
                    "condition_id": "low",
                    "timestamp": "2026-04-01T12:05:00Z",
                    "bid_price_yes": 0.39,
                    "mid_price_yes": 0.40,
                    "ask_price_yes": 0.41,
                    "estimated_spread": 0.02,
                    "impact_per_usdc": 0.0,
                    "capacity_usdc": 100.0,
                }
            ]
        ),
        start_at="2026-04-01T12:05:00Z",
        end_at="2026-04-01T12:05:00Z",
        decision_tick="5min",
        lookback="60min",
        max_staleness="30min",
    ).save(
        quotes_root,
        source_sha256=clob_digest,
        created_at="2026-07-01T00:00:00Z",
    )

    forecast_root = tmp_path / f"forecast-{split_role}"
    ForecastCache(
        snapshots=pd.DataFrame(
            [
                {
                    "snapshot_id": "snapshot",
                    "station_id": "KLGA",
                    "init_time": "2026-04-01T11:00:00Z",
                    "target_date": "2026-04-01",
                    "available_at": "2026-04-01T12:05:00Z",
                    "p10_f": 65.0,
                    "p50_f": 70.0,
                    "p90_f": 75.0,
                    "lineage_id": "lineage",
                }
            ]
        ),
        probabilities=pd.DataFrame(
            [
                {
                    "snapshot_id": "snapshot",
                    "condition_id": "low",
                    "yes_probability": 0.8,
                }
            ]
        ),
        lineages=pd.DataFrame(
            [
                {
                    "lineage_id": "lineage",
                    "checkpoint_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64,
                    "split_role": split_role,
                    "mc_seed": 42,
                    "mc_samples": 2000,
                }
            ]
        ),
    ).save(
        forecast_root,
        created_at="2026-07-01T00:00:00Z",
        source_sha256={"clob_manifest": clob_digest},
    )

    actuals_path = tmp_path / f"actuals-{split_role}.csv"
    pd.DataFrame(
        [{"target_date": "2026-04-01", "actual_high_f": 69.0}]
    ).to_csv(actuals_path, index=False)
    actuals_manifest = {
        "schema_version": 1,
        "data_class": "market_resolution_actual_highs",
        "station_id": "KLGA",
        "requested_range": ["2026-04-01", "2026-04-01"],
        "row_count": 1,
        "source_sha256": "c" * 64,
        "sha256": {actuals_path.name: _sha256(actuals_path)},
    }
    actuals_path.with_suffix(".csv.manifest.json").write_text(
        json.dumps(actuals_manifest),
        encoding="utf-8",
    )
    return StudyInputs(
        forecast_root=forecast_root,
        quotes_root=quotes_root,
        clob_root=clob_root,
        actuals_path=actuals_path,
    )


def test_development_study_registers_one_validated_input_contract(
    tmp_path,
) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    inputs = _study_inputs(tmp_path)

    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=inputs,
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    validated = study.validate(record.study_id)

    assert record.status == "development"
    assert validated.split_role == "oof_development"
    assert validated.input_sha256 == record.input_sha256


def test_public_input_contract_validates_oof_without_catalog(tmp_path) -> None:
    inputs = _study_inputs(tmp_path)

    validated = validate_study_inputs(
        study_id="signal-study",
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=inputs,
        expected_split_role="oof_development",
    )

    assert validated.study_id == "signal-study"
    assert validated.split_role == "oof_development"
    assert not (tmp_path / "studies.sqlite").exists()


def test_input_contract_excludes_only_quality_declared_market_unavailable_days(
    tmp_path,
) -> None:
    inputs = _study_inputs(tmp_path)
    quality_path = inputs.clob_root.with_suffix(".quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["status"] = "accepted_with_exclusions"
    quality["excluded_days"] = {
        "2026-04-02": "event_absent_from_provider_series"
    }
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    validated = validate_study_inputs(
        study_id="signal-study",
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-02",
        inputs=inputs,
        expected_split_role="oof_development",
    )

    assert validated.actuals["target_date"].tolist() == ["2026-04-01"]
    assert validated.excluded_target_dates == ("2026-04-02",)


def test_development_study_rejects_model_test_forecasts(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    inputs = _study_inputs(tmp_path, split_role="model_test")

    with pytest.raises(ValueError, match="model_test is forbidden"):
        study.create_development(
            station_id="KLGA",
            start_target="2026-04-01",
            end_target="2026-04-01",
            inputs=inputs,
            strategy={"min_edge": 0.04, "risk_usd": 50.0},
        )


def test_signal_validation_allows_model_test_without_enabling_trading(tmp_path) -> None:
    inputs = _study_inputs(tmp_path, split_role="model_test")

    validated = validate_signal_inputs(
        study_id="frozen-model-test-diagnostic",
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=inputs,
        expected_split_role="model_test",
    )

    assert validated.split_role == "model_test"


def test_holdout_requires_freeze_and_can_only_be_consumed_once(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    development = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path / "development"),
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    holdout = _study_inputs(
        tmp_path / "holdout",
        split_role="post_test_holdout",
    )

    with pytest.raises(ValueError, match="must be frozen"):
        study.consume_holdout(development.study_id, inputs=holdout)

    frozen = study.freeze(development.study_id, approved_by="john")
    consumed = study.consume_holdout(frozen.study_id, inputs=holdout)

    assert frozen.status == "frozen"
    assert consumed.split_role == "post_test_holdout"
    with pytest.raises(ValueError, match="already consumed"):
        study.consume_holdout(frozen.study_id, inputs=holdout)


def test_concurrent_holdout_consumers_have_one_winner(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=_study_inputs(tmp_path / "development"),
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )
    study.freeze(record.study_id, approved_by="john")
    holdout = _study_inputs(
        tmp_path / "holdout",
        split_role="post_test_holdout",
    )

    def consume() -> str:
        try:
            study.consume_holdout(record.study_id, inputs=holdout)
            return "consumed"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: consume(), range(2)))

    assert results.count("consumed") == 1
    assert sum("already consumed" in result for result in results) == 1


def test_study_rejects_quotes_from_another_clob_version(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    inputs = _study_inputs(tmp_path)
    manifest_path = inputs.quotes_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="another CLOB"):
        study.create_development(
            station_id="KLGA",
            start_target="2026-04-01",
            end_target="2026-04-01",
            inputs=inputs,
            strategy={"min_edge": 0.04, "risk_usd": 50.0},
        )


def test_study_rejects_duplicate_truth_even_with_a_valid_manifest(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    inputs = _study_inputs(tmp_path)
    pd.DataFrame(
        [
            {"target_date": "2026-04-01", "actual_high_f": 69.0},
            {"target_date": "2026-04-01", "actual_high_f": 70.0},
        ]
    ).to_csv(inputs.actuals_path, index=False)
    manifest_path = inputs.actuals_path.with_suffix(".csv.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 2
    manifest["sha256"][inputs.actuals_path.name] = _sha256(
        inputs.actuals_path
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate target_date"):
        study.create_development(
            station_id="KLGA",
            start_target="2026-04-01",
            end_target="2026-04-01",
            inputs=inputs,
            strategy={"min_edge": 0.04, "risk_usd": 50.0},
        )


def test_study_accepts_truth_artifact_that_covers_later_dates(tmp_path) -> None:
    study = BacktestStudy(tmp_path / "studies.sqlite")
    inputs = _study_inputs(tmp_path)
    pd.DataFrame(
        [
            {"target_date": "2026-04-01", "actual_high_f": 69.0},
            {"target_date": "2026-04-02", "actual_high_f": 70.0},
        ]
    ).to_csv(inputs.actuals_path, index=False)
    manifest_path = inputs.actuals_path.with_suffix(".csv.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requested_range"] = ["2026-04-01", "2026-04-02"]
    manifest["row_count"] = 2
    manifest["sha256"][inputs.actuals_path.name] = _sha256(
        inputs.actuals_path
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record = study.create_development(
        station_id="KLGA",
        start_target="2026-04-01",
        end_target="2026-04-01",
        inputs=inputs,
        strategy={"min_edge": 0.04, "risk_usd": 50.0},
    )

    assert record.status == "development"
