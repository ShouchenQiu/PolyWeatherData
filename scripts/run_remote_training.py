"""Remote candidate-training run orchestration.

The remote machine is a candidate training factory: it may download data,
audit readiness, train, and package evidence, but it must not activate a
production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def run_remote_training(
    run_root: Path | str,
    run_id: str,
    station: str,
    start: str,
    end: str,
    split_ranges: dict[str, tuple[str, str] | list[str]],
    version: str,
    workers: int = 12,
    min_epochs: int = 1,
    max_epochs: int = 1,
    batch_size: int = 16,
    seed: int = 42,
    deterministic: bool = True,
    precision: str = "32",
    num_workers: int = 0,
    download_fn: Callable[..., dict[str, Any]] | None = None,
    audit_fn: Callable[..., dict[str, Any]] | None = None,
    train_fn: Callable[..., dict[str, Any]] | None = None,
    package_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    root = Path(run_root)
    run_dir = root / run_id
    data_store = run_dir / "data_store"
    model_store = run_dir / "model_store"
    checkpoints = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    db_path = run_dir / "polyweather.db"
    for path in (data_store, model_store, checkpoints, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    download = download_fn or download_weather_data
    audit = audit_fn or _default_audit
    train = train_fn or _default_train
    package = package_fn or package_candidate_model

    download_summary = download(
        run_dir=run_dir,
        data_store=data_store,
        station=station,
        start=start,
        end=end,
        workers=workers,
    )
    audit_summary = audit(
        data_store=data_store,
        station_id=station,
        split_ranges=split_ranges,
    )
    audit_path = run_dir / "audit_summary.json"
    audit_path.write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    if not audit_summary.get("ready", False):
        failed_path = run_dir / "failed_audit_summary.json"
        failed_path.write_text(audit_path.read_text(encoding="ascii"), encoding="ascii")
        result = {
            "status": "audit_failed",
            "run_id": run_id,
            "audit_summary_path": str(failed_path),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="ascii",
        )
        return result

    train_result = train(
        run_dir=run_dir,
        data_store=data_store,
        model_store=model_store,
        checkpoints=checkpoints,
        db=db_path,
        station=station,
        version=version,
        split_ranges=split_ranges,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
        deterministic=deterministic,
        precision=precision,
        num_workers=num_workers,
    )
    train_result_path = run_dir / "train_result.json"
    train_result_path.write_text(
        json.dumps(train_result, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "version": version,
        "station": station,
        "start": start,
        "end": end,
        "split_ranges": split_ranges,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment_summary(),
        "download": download_summary,
        "audit": audit_summary,
        "train_result": train_result,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    package_summary = package(run_dir=run_dir, version=version)
    weather_package_summary = package_production_weather_data(run_dir=run_dir)
    manifest["candidate_model_package"] = package_summary
    manifest["production_weather_data_package"] = weather_package_summary
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    return manifest


def download_weather_data(**kwargs: Any) -> dict[str, Any]:
    from scripts.smoke_bootstrap import run_smoke_bootstrap

    return run_smoke_bootstrap(
        smoke_dir=kwargs["run_dir"],
        station_id=kwargs["station"],
        start_date=kwargs["start"],
        end_date=kwargs["end"],
        workers=kwargs["workers"],
        skip_polymarket=True,
    )


def package_candidate_model(run_dir: Path | str, version: str) -> dict[str, Any]:
    root = Path(run_dir)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_path = artifact_dir / f"{version}.tar.gz"
    required = [
        root / "model_store" / version / "model.ckpt",
        root / "model_store" / version / "training_dataset.pkl",
        root / "model_store" / version / "meta.json",
        root / "model_store" / version / "training_metadata.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"candidate package missing required artifact: {path}")

    candidates = [
        root / "model_store" / version,
        root / "manifest.json",
        root / "audit_summary.json",
        root / "train_result.json",
    ]
    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as tar:
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_dir():
                for path in sorted(candidate.rglob("*")):
                    if path.is_file() and "_decoder_cache" not in path.parts:
                        arcname = path.relative_to(root).as_posix()
                        tar.add(path, arcname=arcname)
                        members.append(arcname)
            elif "_decoder_cache" not in candidate.parts:
                arcname = candidate.relative_to(root).as_posix()
                tar.add(candidate, arcname=arcname)
                members.append(arcname)
    return {
        "package_path": str(package_path),
        "sha256": _sha256_file(package_path),
        "members": sorted(members),
    }


def package_production_weather_data(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_path = artifact_dir / "production_weather_data.tar.gz"
    data_root = root / "data_store"
    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as tar:
        for subdir in ("hrrr", "obs"):
            source = data_root / subdir
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(root).as_posix()
                    tar.add(path, arcname=arcname)
                    members.append(arcname)
        manifest_path = root / "weather_data_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "included_stores": ["hrrr", "obs"],
                    "excluded_stores": ["_decoder_cache"],
                    "members": sorted(members),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="ascii",
        )
        tar.add(manifest_path, arcname="weather_data_manifest.json")
        members.append("weather_data_manifest.json")
    return {
        "package_path": str(package_path),
        "sha256": _sha256_file(package_path),
        "members": sorted(members),
    }


def _default_audit(**kwargs: Any) -> dict[str, Any]:
    from scripts.audit_weather_backfill import audit_training_readiness

    return audit_training_readiness(
        data_store=kwargs["data_store"],
        station_id=kwargs["station_id"],
        split_ranges=kwargs["split_ranges"],
    )


def _default_train(**kwargs: Any) -> dict[str, Any]:
    from config.settings import Settings
    from config.stations import STATIONS
    from data.manager import DataManager
    from data.storage.parquet_store import ParquetStore
    from db.schema import DbClient
    from model.store.registry import ModelRegistry
    from model.training_data import TrainingDecoderCache
    from scripts.train_tft import TftTrainer

    settings = Settings(data_store_dir=Path(kwargs["data_store"]), model_store_dir=Path(kwargs["model_store"]))
    settings.train_start, settings.train_end = kwargs["split_ranges"]["train"]
    settings.val_start, settings.val_end = kwargs["split_ranges"]["val"]
    settings.test_start, settings.test_end = kwargs["split_ranges"]["test"]
    settings.tft_min_epochs = int(kwargs["min_epochs"])
    settings.tft_max_epochs = int(kwargs["max_epochs"])
    settings.tft_batch_size = int(kwargs["batch_size"])
    forecast_store = ParquetStore(settings.forecast_store_dir)
    obs_store = ParquetStore(settings.obs_store_dir)
    data_manager = DataManager([], [], forecast_store, obs_store, settings=settings)
    decoder_cache = TrainingDecoderCache(settings.training_decoder_cache_dir)
    db = DbClient(str(kwargs["db"]))
    try:
        registry = ModelRegistry(settings.model_store_dir, db)
        trainer = TftTrainer(
            forecast_store,
            registry,
            output_dir=Path(kwargs["checkpoints"]),
            settings=settings,
            data_manager=data_manager,
            decoder_cache=decoder_cache,
            station_config=STATIONS[kwargs["station"]],
            random_seed=kwargs["seed"],
            torch_deterministic=kwargs["deterministic"],
            precision=kwargs["precision"],
            num_workers=kwargs["num_workers"],
            activate_model=False,
        )
        return trainer.run(station_id=kwargs["station"], version=kwargs["version"])
    finally:
        db.close()


def _environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_split(value: str) -> tuple[str, str]:
    start, end = value.split(":", 1)
    return start, end


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run a remote candidate TFT training batch")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--train", required=True, help="start:end")
    parser.add_argument("--val", required=True, help="start:end")
    parser.add_argument("--test", required=True, help="start:end")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    result = run_remote_training(
        run_root=args.run_root,
        run_id=args.run_id,
        station=args.station,
        start=args.start,
        end=args.end,
        split_ranges={
            "train": _parse_split(args.train),
            "val": _parse_split(args.val),
            "test": _parse_split(args.test),
        },
        version=args.version,
        workers=args.workers,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=args.deterministic,
        precision=args.precision,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
