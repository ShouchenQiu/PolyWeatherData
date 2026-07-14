"""Governed input contract and lifecycle for point-in-time trading studies."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset


@dataclass(frozen=True)
class StudyInputs:
    forecast_root: Path
    quotes_root: Path
    clob_root: Path
    actuals_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "forecast_root": str(Path(self.forecast_root).resolve()),
            "quotes_root": str(Path(self.quotes_root).resolve()),
            "clob_root": str(Path(self.clob_root).resolve()),
            "actuals_path": str(Path(self.actuals_path).resolve()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "StudyInputs":
        return cls(**{name: Path(value) for name, value in payload.items()})


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    status: str
    station_id: str
    start_target: str
    end_target: str
    split_role: str
    strategy_sha256: str
    input_sha256: dict[str, str]


@dataclass(frozen=True)
class StudyValidation:
    study_id: str
    station_id: str
    start_target: str
    end_target: str
    split_role: str
    input_sha256: dict[str, str]
    inputs: StudyInputs
    forecast: ForecastCache
    quotes: SyntheticQuoteDataset
    clob: HistoricalClobDataset
    actuals: pd.DataFrame
    excluded_target_dates: tuple[str, ...] = ()


class BacktestStudy:
    """Single façade for validated development and holdout studies."""

    def __init__(self, catalog_path: Path | str) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_development(
        self,
        *,
        station_id: str,
        start_target: str,
        end_target: str,
        inputs: StudyInputs,
        strategy: dict,
    ) -> StudyRecord:
        validated = self._validate_inputs(
            study_id="pending",
            station_id=station_id,
            start_target=start_target,
            end_target=end_target,
            inputs=inputs,
            expected_split_role="oof_development",
        )
        strategy_json = _canonical_json(strategy)
        strategy_sha256 = _digest_bytes(strategy_json.encode("utf-8"))
        identity = {
            "station_id": station_id,
            "start_target": start_target,
            "end_target": end_target,
            "split_role": validated.split_role,
            "strategy_sha256": strategy_sha256,
            "input_sha256": validated.input_sha256,
        }
        study_id = "edge-" + _digest_bytes(
            _canonical_json(identity).encode("utf-8")
        )[:16]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO studies (
                    study_id, status, station_id, start_target, end_target,
                    split_role, strategy_json, strategy_sha256, inputs_json,
                    input_sha256_json
                ) VALUES (?, 'development', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    station_id,
                    start_target,
                    end_target,
                    validated.split_role,
                    strategy_json,
                    strategy_sha256,
                    _canonical_json(inputs.as_dict()),
                    _canonical_json(validated.input_sha256),
                ),
            )
        return self.get(study_id)

    def get(self, study_id: str) -> StudyRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown backtest study {study_id!r}")
        return _record(row)

    def validate(self, study_id: str) -> StudyValidation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown backtest study {study_id!r}")
        validated = self._validate_inputs(
            study_id=study_id,
            station_id=str(row["station_id"]),
            start_target=str(row["start_target"]),
            end_target=str(row["end_target"]),
            inputs=StudyInputs.from_dict(json.loads(row["inputs_json"])),
            expected_split_role=str(row["split_role"]),
        )
        expected = json.loads(row["input_sha256_json"])
        if validated.input_sha256 != expected:
            raise ValueError("backtest study input hashes changed after registration")
        return validated

    def run_development(
        self,
        study_id: str,
        *,
        output: Path | str,
        folds: int = 4,
        bootstrap_samples: int = 5_000,
        seed: int = 42,
    ) -> dict:
        from backtest.edge_study import (
            build_edge_observations,
            build_edge_study_report,
            cross_fit_edge_calibration,
            run_diagnostic_replay,
            save_edge_study_artifact,
        )

        record = self.get(study_id)
        if record.status != "development":
            raise ValueError("only a development study can run development replay")
        validated = self.validate(study_id)
        strategy = self._strategy(study_id)
        observations = build_edge_observations(
            validated,
            risk_usd=float(strategy.get("risk_usd", 50.0)),
        )
        calibrated = cross_fit_edge_calibration(
            observations,
            folds=folds,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        replay = run_diagnostic_replay(
            calibrated.observations,
            min_edge=float(strategy.get("min_edge", 0.04)),
            risk_usd=float(strategy.get("risk_usd", 50.0)),
        )
        report = build_edge_study_report(
            calibrated.observations,
            replay,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        manifest = save_edge_study_artifact(
            output,
            study_id=study_id,
            split_role=validated.split_role,
            input_sha256=validated.input_sha256,
            observations=calibrated.observations,
            calibrator=calibrated.frozen_calibrator,
            replay=replay,
            report=report,
            folds=calibrated.folds,
        )
        calibration_json = _canonical_json(
            calibrated.frozen_calibrator.to_dict()
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE studies
                SET calibration_json = ?, calibration_sha256 = ?,
                    development_manifest_sha256 = ?
                WHERE study_id = ? AND status = 'development'
                """,
                (
                    calibration_json,
                    manifest["calibration_sha256"],
                    _sha256(Path(output) / "manifest.json"),
                    study_id,
                ),
            )
        return manifest

    def run_holdout(
        self,
        study_id: str,
        *,
        inputs: StudyInputs,
        output: Path | str,
        bootstrap_samples: int = 5_000,
        seed: int = 42,
    ) -> dict:
        from backtest.edge_study import (
            EdgeCalibrator,
            build_edge_observations,
            build_edge_study_report,
            run_diagnostic_replay,
            save_edge_study_artifact,
        )

        if Path(output).exists():
            raise FileExistsError(output)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT calibration_json FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown backtest study {study_id!r}")
        if not row["calibration_json"]:
            raise ValueError("development calibration must be frozen before holdout")
        calibrator = EdgeCalibrator.from_dict(
            json.loads(row["calibration_json"])
        )
        validated = self.consume_holdout(study_id, inputs=inputs)
        strategy = self._strategy(study_id)
        observations = build_edge_observations(
            validated,
            risk_usd=float(strategy.get("risk_usd", 50.0)),
        )
        calibrated_observations = calibrator.calibrate(observations)
        replay = run_diagnostic_replay(
            calibrated_observations,
            min_edge=float(strategy.get("min_edge", 0.04)),
            risk_usd=float(strategy.get("risk_usd", 50.0)),
        )
        report = build_edge_study_report(
            calibrated_observations,
            replay,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        return save_edge_study_artifact(
            output,
            study_id=study_id,
            split_role=validated.split_role,
            input_sha256=validated.input_sha256,
            observations=calibrated_observations,
            calibrator=calibrator,
            replay=replay,
            report=report,
        )

    def freeze(self, study_id: str, *, approved_by: str) -> StudyRecord:
        if not approved_by.strip():
            raise ValueError("freezing a study requires an approver")
        frozen_at = pd.Timestamp.now(tz="UTC").isoformat()
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE studies
                SET status = 'frozen', approved_by = ?, frozen_at = ?
                WHERE study_id = ? AND status = 'development'
                """,
                (approved_by.strip(), frozen_at, study_id),
            ).rowcount
        if changed != 1:
            record = self.get(study_id)
            if record.status == "frozen":
                raise ValueError("backtest study is already frozen")
            raise ValueError(
                f"backtest study cannot be frozen from status {record.status!r}"
            )
        return self.get(study_id)

    def consume_holdout(
        self,
        study_id: str,
        *,
        inputs: StudyInputs,
    ) -> StudyValidation:
        record = self.get(study_id)
        if record.status == "holdout_consumed":
            raise ValueError("post_test_holdout was already consumed")
        if record.status != "frozen":
            raise ValueError("backtest study must be frozen before holdout")
        validated = self._validate_inputs(
            study_id=study_id,
            station_id=record.station_id,
            start_target=record.start_target,
            end_target=record.end_target,
            inputs=inputs,
            expected_split_role="post_test_holdout",
        )
        consumed_at = pd.Timestamp.now(tz="UTC").isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown backtest study {study_id!r}")
            if current["status"] != "frozen":
                raise ValueError("post_test_holdout was already consumed")
            connection.execute(
                """
                UPDATE studies
                SET status = 'holdout_consumed',
                    holdout_consumed_at = ?,
                    holdout_inputs_json = ?,
                    holdout_input_sha256_json = ?
                WHERE study_id = ?
                """,
                (
                    consumed_at,
                    _canonical_json(inputs.as_dict()),
                    _canonical_json(validated.input_sha256),
                    study_id,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return validated

    @staticmethod
    def _validate_inputs(
        *,
        study_id: str,
        station_id: str,
        start_target: str,
        end_target: str,
        inputs: StudyInputs,
        expected_split_role: str,
        allow_model_test: bool = False,
    ) -> StudyValidation:
        allowed_splits = {"oof_development", "post_test_holdout"}
        if allow_model_test:
            allowed_splits.add("model_test")
        if expected_split_role == "model_test" and not allow_model_test:
            raise ValueError("model_test is forbidden for trading studies")
        if expected_split_role not in allowed_splits:
            raise ValueError(f"unsupported trading-study split {expected_split_role!r}")
        forecast_manifest_path = Path(inputs.forecast_root) / "manifest.json"
        quotes_manifest_path = Path(inputs.quotes_root) / "manifest.json"
        clob_manifest_path = Path(inputs.clob_root) / "manifest.json"
        actuals_manifest_path = Path(inputs.actuals_path).with_suffix(
            Path(inputs.actuals_path).suffix + ".manifest.json"
        )
        manifests = {
            "forecast_manifest": _load_manifest(
                forecast_manifest_path,
                schema_version=2,
                data_class="point_in_time_forecast_cache",
            ),
            "quotes_manifest": _load_manifest(
                quotes_manifest_path,
                schema_version=1,
                data_class="past_only_synthetic_quotes",
            ),
            "clob_manifest": _load_manifest(
                clob_manifest_path,
                schema_version=1,
                data_class="raw_point_in_time",
            ),
            "actuals_manifest": _load_manifest(
                actuals_manifest_path,
                schema_version=1,
                data_class="market_resolution_actual_highs",
            ),
        }
        input_sha256 = {
            name: _sha256(path)
            for name, path in {
                "forecast_manifest": forecast_manifest_path,
                "quotes_manifest": quotes_manifest_path,
                "clob_manifest": clob_manifest_path,
                "actuals_manifest": actuals_manifest_path,
            }.items()
        }
        clob_digest = input_sha256["clob_manifest"]
        if (
            manifests["quotes_manifest"].get("source_sha256")
            != clob_digest
        ):
            raise ValueError("synthetic quotes are bound to another CLOB artifact")
        if (
            manifests["forecast_manifest"]
            .get("source_sha256", {})
            .get("clob_manifest")
            != clob_digest
        ):
            raise ValueError("forecast cache is bound to another CLOB artifact")

        forecast = ForecastCache.load(inputs.forecast_root)
        quotes = SyntheticQuoteDataset.load(inputs.quotes_root)
        clob = HistoricalClobDataset.load(inputs.clob_root)
        actuals = _load_actuals(inputs.actuals_path, manifests["actuals_manifest"])
        split_roles = set(forecast.lineages["split_role"].astype(str))
        if "model_test" in split_roles and not allow_model_test:
            raise ValueError("model_test is forbidden for trading studies")
        if split_roles != {expected_split_role}:
            raise ValueError(
                f"study requires split_role={expected_split_role}, "
                f"found {sorted(split_roles)}"
            )
        if set(forecast.snapshots["station_id"].astype(str)) != {station_id}:
            raise ValueError("forecast station does not match study station")
        if clob.station_id != station_id:
            raise ValueError("CLOB station does not match study station")
        if manifests["actuals_manifest"].get("station_id") != station_id:
            raise ValueError("actual-highs station does not match study station")

        quality_path = Path(inputs.clob_root).with_suffix(".quality.json")
        quality = _read_json(quality_path, "CLOB eligibility contract")
        if quality.get("source_manifest_sha256") != clob_digest:
            raise ValueError("CLOB eligibility contract is stale")
        if quality.get("status") not in {
            "accepted_complete",
            "accepted_with_exclusions",
        }:
            raise ValueError("CLOB eligibility contract is not accepted")

        requested = {
            stamp.date().isoformat()
            for stamp in pd.date_range(start_target, end_target, freq="D")
        }
        if not requested:
            raise ValueError("study target range is empty")
        declared_exclusions = _quality_exclusions(quality)
        excluded_target_dates = tuple(sorted(requested & set(declared_exclusions)))
        eligible_requested = requested - set(excluded_target_dates)
        if not eligible_requested:
            raise ValueError("study target range has no market-eligible dates")
        coverage = clob.coverage()
        covered = set(
            coverage.loc[coverage["complete"], "target_date"].astype(str)
        )
        if not eligible_requested.issubset(covered):
            missing = sorted(eligible_requested - covered)
            raise ValueError(f"CLOB coverage missing target dates: {missing}")
        event_dates = set(clob.events["target_date"].astype(str))
        if not eligible_requested.issubset(event_dates):
            raise ValueError("CLOB events do not cover the study target range")
        event_ids = set(
            clob.events.loc[
                clob.events["target_date"].astype(str).isin(eligible_requested),
                "event_id",
            ].astype(str)
        )
        if not event_ids or not set(
            clob.markets.loc[
                clob.markets["event_id"].astype(str).isin(event_ids),
                "event_id",
            ].astype(str)
        ) == event_ids:
            raise ValueError("CLOB Markets do not cover every study Event")
        if actuals["target_date"].duplicated().any():
            raise ValueError("actual highs contain duplicate target_date")
        actual_dates = set(actuals["target_date"])
        if not eligible_requested.issubset(actual_dates):
            missing = sorted(eligible_requested - actual_dates)
            raise ValueError(
                f"actual-highs coverage missing target dates: {missing}"
            )
        actuals = actuals[actuals["target_date"].isin(eligible_requested)].reset_index(
            drop=True
        )
        snapshot_dates = set(forecast.snapshots["target_date"].astype(str))
        if not eligible_requested.issubset(snapshot_dates):
            raise ValueError("forecast cache does not cover the study target range")
        return StudyValidation(
            study_id=study_id,
            station_id=station_id,
            start_target=start_target,
            end_target=end_target,
            split_role=expected_split_role,
            input_sha256=input_sha256,
            inputs=inputs,
            forecast=forecast,
            quotes=quotes,
            clob=clob,
            actuals=actuals,
            excluded_target_dates=excluded_target_dates,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    start_target TEXT NOT NULL,
                    end_target TEXT NOT NULL,
                    split_role TEXT NOT NULL,
                    strategy_json TEXT NOT NULL,
                    strategy_sha256 TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    input_sha256_json TEXT NOT NULL,
                    approved_by TEXT,
                    frozen_at TEXT,
                    holdout_consumed_at TEXT,
                    holdout_inputs_json TEXT,
                    holdout_input_sha256_json TEXT,
                    calibration_json TEXT,
                    calibration_sha256 TEXT,
                    development_manifest_sha256 TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.catalog_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _strategy(self, study_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT strategy_json FROM studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown backtest study {study_id!r}")
        return json.loads(row["strategy_json"])


def validate_study_inputs(
    *,
    study_id: str,
    station_id: str,
    start_target: str,
    end_target: str,
    inputs: StudyInputs,
    expected_split_role: str,
) -> StudyValidation:
    """Validate immutable replay inputs without creating a study catalog."""
    return BacktestStudy._validate_inputs(
        study_id=study_id,
        station_id=station_id,
        start_target=start_target,
        end_target=end_target,
        inputs=inputs,
        expected_split_role=expected_split_role,
    )


def validate_signal_inputs(
    *,
    study_id: str,
    station_id: str,
    start_target: str,
    end_target: str,
    inputs: StudyInputs,
    expected_split_role: str,
) -> StudyValidation:
    """Validate forecast-signal inputs without trading-study split bans."""
    return BacktestStudy._validate_inputs(
        study_id=study_id,
        station_id=station_id,
        start_target=start_target,
        end_target=end_target,
        inputs=inputs,
        expected_split_role=expected_split_role,
        allow_model_test=True,
    )


def _record(row: sqlite3.Row) -> StudyRecord:
    return StudyRecord(
        study_id=str(row["study_id"]),
        status=str(row["status"]),
        station_id=str(row["station_id"]),
        start_target=str(row["start_target"]),
        end_target=str(row["end_target"]),
        split_role=str(row["split_role"]),
        strategy_sha256=str(row["strategy_sha256"]),
        input_sha256=json.loads(row["input_sha256_json"]),
    )


def _load_manifest(
    path: Path,
    *,
    schema_version: int,
    data_class: str,
) -> dict:
    manifest = _read_json(path, f"{data_class} manifest")
    if manifest.get("schema_version") != schema_version:
        raise ValueError(f"unsupported {data_class} schema_version")
    if manifest.get("data_class") != data_class:
        raise ValueError(f"invalid artifact data_class for {path}")
    return manifest


def _load_actuals(path: Path, manifest: dict) -> pd.DataFrame:
    expected = manifest.get("sha256", {}).get(path.name)
    if expected != _sha256(path):
        raise ValueError("actual-highs checksum mismatch")
    frame = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path)
    )
    missing = {"target_date", "actual_high_f"} - set(frame)
    if missing:
        raise ValueError(f"actual highs missing columns: {sorted(missing)}")
    work = frame[["target_date", "actual_high_f"]].copy()
    work["target_date"] = pd.to_datetime(work["target_date"]).dt.date.astype(str)
    work["actual_high_f"] = pd.to_numeric(
        work["actual_high_f"], errors="raise"
    )
    if len(work) != int(manifest.get("row_count", -1)):
        raise ValueError("actual-highs row count does not match manifest")
    return work


def _quality_exclusions(quality: dict) -> dict[str, str]:
    """Normalize accepted market-unavailable dates from the CLOB audit."""
    raw = quality.get("excluded_days", {})
    if not isinstance(raw, dict):
        raise ValueError("CLOB eligibility exclusions must be a date/reason mapping")
    exclusions: dict[str, str] = {}
    for value, reason in raw.items():
        try:
            date = pd.Timestamp(value).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CLOB exclusion date: {value!r}") from exc
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"CLOB exclusion requires a reason: {date}")
        exclusions[date] = reason
    return exclusions


def _read_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _sha256(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
