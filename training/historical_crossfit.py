"""Immutable, chronological TFT cross-fit artifacts for offline research."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HistoricalCrossfitFold:
    fold_id: str
    train: tuple[str, str]
    tune: tuple[str, str]
    predict: tuple[str, str]

    @property
    def predict_months(self) -> tuple[str, ...]:
        start_year, start_month, _ = (int(value) for value in self.predict[0].split("-"))
        end_year, end_month, _ = (int(value) for value in self.predict[1].split("-"))
        result = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            result.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return tuple(result)


@dataclass(frozen=True)
class HistoricalCrossfitPlan:
    plan_id: str
    station_id: str
    folds: tuple[HistoricalCrossfitFold, ...]


def historical_six_fold_plan() -> HistoricalCrossfitPlan:
    """Return the fixed rolling-origin plan through 2026-03."""
    return HistoricalCrossfitPlan(
        plan_id="historical_6fold_20250401_20260331",
        station_id="KLGA",
        folds=(
            HistoricalCrossfitFold("fold-00", ("2024-10-07", "2025-01-31"), ("2025-02-01", "2025-03-31"), ("2025-04-01", "2025-05-31")),
            HistoricalCrossfitFold("fold-01", ("2024-10-07", "2025-03-31"), ("2025-04-01", "2025-05-31"), ("2025-06-01", "2025-07-31")),
            HistoricalCrossfitFold("fold-02", ("2024-10-07", "2025-05-31"), ("2025-06-01", "2025-07-31"), ("2025-08-01", "2025-09-30")),
            HistoricalCrossfitFold("fold-03", ("2024-10-07", "2025-07-31"), ("2025-08-01", "2025-09-30"), ("2025-10-01", "2025-11-30")),
            HistoricalCrossfitFold("fold-04", ("2024-10-07", "2025-09-30"), ("2025-10-01", "2025-11-30"), ("2025-12-01", "2026-01-31")),
            HistoricalCrossfitFold("fold-05", ("2024-10-07", "2025-11-30"), ("2025-12-01", "2026-01-31"), ("2026-02-01", "2026-03-31")),
        ),
    )


def crossfit_template_payload(
    reference: dict[str, object],
    fold: HistoricalCrossfitFold,
) -> dict[str, object]:
    """Freeze a reference TFT configuration onto one chronological fold."""
    payload = json.loads(json.dumps(reference))
    if payload.get("schema_version") == 2:
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("use_history_target") is not False:
            raise ValueError(
                "cross-fit accepts schema-v2 references only with "
                "use_history_target=false"
            )
        model.pop("use_history_target")
        payload["schema_version"] = 1
    payload["template_id"] = f"historical-crossfit-{fold.fold_id}"
    payload["splits"] = {
        "train": list(fold.train),
        "val": list(fold.tune),
        "test": list(fold.predict),
    }
    return payload


@dataclass(frozen=True)
class FoldExecutionArtifacts:
    checkpoint: Path
    training_dataset: Path
    predictions: Path
    source_sha256: dict[str, str] | None = None


@dataclass(frozen=True)
class HistoricalCrossfitResult:
    completed: tuple[str, ...]
    skipped: tuple[str, ...]


class HistoricalCrossfitRunner:
    """Run only incomplete or hash-invalid folds through an injected executor."""

    def __init__(
        self,
        *,
        execute_fold: Callable[
            [HistoricalCrossfitFold, Path, Path], FoldExecutionArtifacts
        ],
        plan: HistoricalCrossfitPlan | None = None,
    ) -> None:
        self._execute_fold = execute_fold
        self._plan = plan or historical_six_fold_plan()

    def run(
        self,
        *,
        source_run: Path | str,
        output: Path | str,
        resume: bool,
    ) -> HistoricalCrossfitResult:
        source = Path(source_run)
        if not source.is_dir():
            raise FileNotFoundError(f"cross-fit source run does not exist: {source}")
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        self._write_or_validate_root_manifest(root, source)
        completed: list[str] = []
        skipped: list[str] = []
        for fold in self._plan.folds:
            fold_root = root / fold.fold_id
            if resume and self._is_complete(fold_root, fold):
                skipped.append(fold.fold_id)
                continue
            fold_root.mkdir(parents=True, exist_ok=True)
            artifacts = self._execute_fold(fold, source, fold_root)
            self._write_fold_manifest(fold_root, fold, artifacts)
            completed.append(fold.fold_id)
        return HistoricalCrossfitResult(tuple(completed), tuple(skipped))

    def _write_or_validate_root_manifest(self, root: Path, source: Path) -> None:
        path = root / "manifest.json"
        payload = {
            "schema_version": 1,
            "data_class": "historical_crossfit_tft",
            "plan_id": self._plan.plan_id,
            "station_id": self._plan.station_id,
            "source_run": str(source),
            "fold_ids": [fold.fold_id for fold in self._plan.folds],
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="ascii"))
            if existing != payload:
                raise ValueError("cross-fit output manifest does not match requested plan")
            return
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")

    def _is_complete(self, root: Path, fold: HistoricalCrossfitFold) -> bool:
        manifest_path = _fold_manifest_path(root)
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            if manifest.get("fold") != _fold_payload(fold):
                return False
            hashes = manifest["sha256"]
            return all(
                _sha256(root / filename) == expected
                for filename, expected in hashes.items()
            )
        except (KeyError, OSError, ValueError):
            return False

    @staticmethod
    def _write_fold_manifest(
        root: Path,
        fold: HistoricalCrossfitFold,
        artifacts: FoldExecutionArtifacts,
    ) -> None:
        paths = {
            "checkpoint": artifacts.checkpoint,
            "training_dataset": artifacts.training_dataset,
            "predictions": artifacts.predictions,
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("cross-fit fold did not produce: " + ", ".join(missing))
        canonical_root = root.resolve()
        hashes = {
            str(path.resolve().relative_to(canonical_root)): _sha256(path)
            for path in paths.values()
        }
        if len(hashes) != len(paths):
            raise ValueError("cross-fit artifact filenames must be unique")
        payload = {
            "schema_version": 1,
            "fold": _fold_payload(fold),
            "artifacts": {
                name: str(path.resolve().relative_to(canonical_root))
                for name, path in paths.items()
            },
            "sha256": hashes,
            "source_sha256": dict(sorted((artifacts.source_sha256 or {}).items())),
        }
        _fold_manifest_path(root).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="ascii"
        )


def _fold_payload(fold: HistoricalCrossfitFold) -> dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "train": list(fold.train),
        "tune": list(fold.tune),
        "predict": list(fold.predict),
    }


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_historical_crossfit_predictions(root: Path | str) -> pd.DataFrame:
    """Load only complete, hash-validated fold predictions in plan order."""
    directory = Path(root)
    plan = historical_six_fold_plan()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
    if manifest.get("plan_id") != plan.plan_id:
        raise ValueError("cross-fit prediction root has another plan")
    frames: list[pd.DataFrame] = []
    for fold in plan.folds:
        fold_root = directory / fold.fold_id
        fold_manifest = json.loads(
            _fold_manifest_path(fold_root).read_text(encoding="ascii")
        )
        if fold_manifest.get("fold") != _fold_payload(fold):
            raise ValueError(f"cross-fit fold boundary mismatch: {fold.fold_id}")
        artifacts = fold_manifest.get("artifacts", {})
        if set(artifacts) != {"checkpoint", "training_dataset", "predictions"}:
            raise ValueError(f"cross-fit artifact contract invalid: {fold.fold_id}")
        hashes = fold_manifest.get("sha256", {})
        if set(hashes) != set(artifacts.values()):
            raise ValueError(f"cross-fit artifact hashes incomplete: {fold.fold_id}")
        for relative, expected in hashes.items():
            if _sha256(fold_root / relative) != expected:
                raise ValueError(f"cross-fit artifact hash mismatch: {fold.fold_id}/{relative}")
        prediction_name = artifacts.get("predictions")
        if not isinstance(prediction_name, str):
            raise ValueError(f"cross-fit predictions missing: {fold.fold_id}")
        frame = pd.read_parquet(fold_root / prediction_name)
        if not {"init_time", "valid_time"}.issubset(frame.columns):
            raise ValueError(f"cross-fit prediction schema invalid: {fold.fold_id}")
        init_dates = pd.to_datetime(frame["init_time"], utc=True).dt.strftime("%Y-%m-%d")
        if not init_dates.between(*fold.predict).all():
            raise ValueError(f"cross-fit prediction init outside fold: {fold.fold_id}")
        frame = frame.copy()
        frame["fold_id"] = fold.fold_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fold_manifest_path(root: Path) -> Path:
    return root / "crossfit_manifest.json"


class DefaultHistoricalFoldExecutor:
    """Training-factory adapter for one immutable historical fold workspace."""

    def __init__(self, *, reference_template: Path | str) -> None:
        self._reference_template = Path(reference_template)

    def __call__(
        self,
        fold: HistoricalCrossfitFold,
        source_run: Path,
        output: Path,
    ) -> FoldExecutionArtifacts:
        from training.audit import audit_workspace
        from training.audit_frames import AuditedFrameCache
        from training.predictions import build_quantile_prediction_frame
        from training.trainer import train_workspace

        source_data = source_run / "data_store"
        if not source_data.is_dir():
            raise FileNotFoundError(f"source run has no data_store: {source_data}")
        if not self._reference_template.is_file():
            raise FileNotFoundError(
                f"reference template does not exist: {self._reference_template}"
            )
        existing = _existing_fold_artifacts(output, self._reference_template)
        if existing is not None:
            return existing
        data_link = output / "data_store"
        if not data_link.exists():
            data_link.symlink_to(source_data.resolve(), target_is_directory=True)
        elif not data_link.is_symlink() or data_link.resolve() != source_data.resolve():
            raise ValueError("cross-fit fold data_store must link to the source run")
        template_path = output / "experiment_config.json"
        template_path.write_text(
            json.dumps(
                crossfit_template_payload(
                    json.loads(self._reference_template.read_text(encoding="ascii")),
                    fold,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="ascii",
        )
        audit_workspace(
            output,
            "KLGA",
            {"train": fold.train, "val": fold.tune, "test": fold.predict},
            progress=True,
            progress_every=250,
        )
        result = train_workspace(
            output,
            station_id="KLGA",
            version=fold.fold_id,
            experiment_config=template_path,
        )
        checkpoint = Path(str(result["train_result"]["checkpoint_path"]))
        training_dataset = checkpoint.parent / "training_dataset.pkl"
        test_frame = AuditedFrameCache(
            output / "artifacts" / "audit_frames", source_data
        ).load("test", "KLGA", *fold.predict)
        if test_frame is None:
            raise RuntimeError("cross-fit prediction frame is missing after audit")
        frame, _ = test_frame
        predictions = output / "predictions.parquet"
        build_quantile_prediction_frame(
            checkpoint_path=checkpoint,
            training_dataset=_load_pickle(training_dataset),
            frame=frame,
            batch_size=256,
        ).to_parquet(predictions, index=False)
        audit = json.loads((output / "audit_summary.json").read_text(encoding="ascii"))
        return FoldExecutionArtifacts(
            checkpoint,
            training_dataset,
            predictions,
            source_sha256={
                "reference_template": _sha256(self._reference_template),
                "train_source_fingerprint": str(
                    audit["splits"]["train"]["source_fingerprint"]
                ),
                "tune_source_fingerprint": str(
                    audit["splits"]["val"]["source_fingerprint"]
                ),
                "predict_source_fingerprint": str(
                    audit["splits"]["test"]["source_fingerprint"]
                ),
            },
        )


def _load_pickle(path: Path) -> object:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def _existing_fold_artifacts(
    output: Path,
    reference_template: Path,
) -> FoldExecutionArtifacts | None:
    """Recover a validated training workspace interrupted before cross-fit sealing."""
    result_path = output / "validation_result.json"
    predictions = output / "predictions.parquet"
    audit_path = output / "audit_summary.json"
    if not (result_path.is_file() and predictions.is_file() and audit_path.is_file()):
        return None
    result = json.loads(result_path.read_text(encoding="ascii"))
    train_result = result.get("train_result", result)
    if not isinstance(train_result, dict):
        return None
    checkpoint_value = train_result.get("checkpoint_path")
    if not isinstance(checkpoint_value, str):
        return None
    checkpoint = Path(checkpoint_value)
    dataset = checkpoint.parent / "training_dataset.pkl"
    if not checkpoint.is_file() or not dataset.is_file():
        return None
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    if not audit.get("ready", False):
        return None
    return FoldExecutionArtifacts(
        checkpoint,
        dataset,
        predictions,
        source_sha256={
            "reference_template": _sha256(reference_template),
            "train_source_fingerprint": str(
                audit["splits"]["train"]["source_fingerprint"]
            ),
            "tune_source_fingerprint": str(
                audit["splits"]["val"]["source_fingerprint"]
            ),
            "predict_source_fingerprint": str(
                audit["splits"]["test"]["source_fingerprint"]
            ),
        },
    )
