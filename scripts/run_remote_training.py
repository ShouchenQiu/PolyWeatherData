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


def run_download_stage(
    run_root: Path | str,
    run_id: str,
    station: str,
    start: str,
    end: str,
    workers: int = 12,
    download_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    download = download_fn or download_weather_data
    paths = _run_paths(run_root, run_id)
    _ensure_run_dirs(paths)

    download_summary = download(
        run_dir=paths["run_dir"],
        data_store=paths["data_store"],
        station=station,
        start=start,
        end=end,
        workers=workers,
    )
    result = {
        "status": "downloaded",
        "run_id": run_id,
        "station": station,
        "start": start,
        "end": end,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "download": download_summary,
    }
    _write_json(paths["run_dir"] / "download_summary.json", download_summary)
    _record_stage(paths["run_dir"], "download", result)
    _merge_manifest(paths["run_dir"], result)
    return result


def run_audit_stage(
    run_root: Path | str,
    run_id: str,
    station: str,
    split_ranges: dict[str, tuple[str, str] | list[str]],
    audit_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    paths = _run_paths(run_root, run_id)
    if not paths["data_store"].exists():
        raise FileNotFoundError(
            f"audit stage requires data_store at {paths['data_store']}; "
            "run download first"
        )
    audit = audit_fn or _default_audit
    audit_summary = audit(
        data_store=paths["data_store"],
        station_id=station,
        split_ranges=split_ranges,
    )
    audit_path = paths["run_dir"] / "audit_summary.json"
    _write_json(audit_path, audit_summary)
    status = "audited" if audit_summary.get("ready", False) else "audit_failed"
    result = {
        "status": status,
        "run_id": run_id,
        "station": station,
        "split_ranges": split_ranges,
        "audit_summary_path": str(audit_path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "audit": audit_summary,
    }
    if status == "audit_failed":
        failed_path = paths["run_dir"] / "failed_audit_summary.json"
        failed_path.write_text(audit_path.read_text(encoding="ascii"), encoding="ascii")
        result["failed_audit_summary_path"] = str(failed_path)
    _record_stage(paths["run_dir"], "audit", result)
    _merge_manifest(paths["run_dir"], result)
    return result


def run_train_stage(
    run_root: Path | str,
    run_id: str,
    station: str,
    version: str,
    split_ranges: dict[str, tuple[str, str] | list[str]],
    min_epochs: int = 1,
    max_epochs: int = 1,
    batch_size: int = 16,
    seed: int = 42,
    deterministic: bool = True,
    precision: str = "32",
    num_workers: int = 0,
    train_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    paths = _run_paths(run_root, run_id)
    audit_path = paths["run_dir"] / "audit_summary.json"
    if not audit_path.exists():
        raise RuntimeError("train stage requires audit_summary.json with ready=true")
    audit_summary = _read_json(audit_path)
    if not audit_summary.get("ready", False):
        raise RuntimeError("train stage requires audit_summary.json with ready=true")
    _ensure_run_dirs(paths)
    train = train_fn or _default_train

    train_result = train(
        run_dir=paths["run_dir"],
        data_store=paths["data_store"],
        model_store=paths["model_store"],
        checkpoints=paths["checkpoints"],
        db=paths["db_path"],
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
    train_result_path = paths["run_dir"] / "train_result.json"
    _write_json(train_result_path, train_result)
    result = {
        "status": "trained",
        "run_id": run_id,
        "version": version,
        "station": station,
        "split_ranges": split_ranges,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "train_result": train_result,
        "train_result_path": str(train_result_path),
    }
    _record_stage(paths["run_dir"], "train", result)
    _merge_manifest(paths["run_dir"], result)
    return result


def run_package_stage(
    run_root: Path | str,
    run_id: str,
    version: str,
    model_only: bool = False,
    package_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    paths = _run_paths(run_root, run_id)
    if not (paths["run_dir"] / "train_result.json").exists():
        raise FileNotFoundError("package stage requires train_result.json")
    if not (paths["model_store"] / version).exists():
        raise FileNotFoundError(f"package stage requires model_store/{version}")
    package = package_fn or package_candidate_model
    manifest_path = paths["run_dir"] / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest["status"] = "packaging"
    manifest["latest_packaged_version"] = version
    _write_json(manifest_path, manifest)
    package_summary = package(run_dir=paths["run_dir"], version=version)
    result = {
        "status": "completed",
        "run_id": run_id,
        "version": version,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "candidate_model_package": package_summary,
    }
    if not model_only:
        result["production_weather_data_package"] = package_production_weather_data(
            run_dir=paths["run_dir"]
        )
    _record_stage(paths["run_dir"], "package", result)
    _merge_manifest(paths["run_dir"], result)
    return _read_json(manifest_path)


def run_export_lab_input_stage(
    run_root: Path | str,
    run_id: str,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    paths = _run_paths(run_root, run_id)
    audit_path = paths["run_dir"] / "audit_summary.json"
    if not audit_path.exists():
        raise RuntimeError("export-lab-input requires audit_summary.json with ready=true")
    audit_summary = _read_json(audit_path)
    if not audit_summary.get("ready", False):
        raise RuntimeError("export-lab-input requires audit_summary.json with ready=true")
    package_summary = export_lab_input_package(paths["run_dir"])
    result = {
        "status": "lab_input_exported",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "lab_input_package": package_summary,
    }
    _record_stage(paths["run_dir"], "export_lab_input", result)
    _merge_manifest(paths["run_dir"], result)
    return result


def download_weather_data(**kwargs: Any) -> dict[str, Any]:
    from scripts.batch_prepare_training_data import run_batch_prepare_training_data

    return run_batch_prepare_training_data(
        run_dir=kwargs["run_dir"],
        station_id=kwargs["station"],
        start_date=kwargs["start"],
        end_date=kwargs["end"],
        workers=kwargs["workers"],
        skip_polymarket=True,
    )


def _run_paths(run_root: Path | str, run_id: str) -> dict[str, Path]:
    root = Path(run_root)
    run_dir = root / run_id
    return {
        "run_dir": run_dir,
        "data_store": run_dir / "data_store",
        "model_store": run_dir / "model_store",
        "checkpoints": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "db_path": run_dir / "polyweather.db",
    }


def _ensure_run_dirs(paths: dict[str, Path]) -> None:
    for key in ("data_store", "model_store", "checkpoints", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def _record_stage(run_dir: Path, stage: str, result: dict[str, Any]) -> None:
    state_path = run_dir / "stage_state.json"
    state = _read_json(state_path) if state_path.exists() else {"stages": {}}
    state.setdefault("stages", {})[stage] = {
        "status": result.get("status"),
        "version": result.get("version"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
    }
    _write_json(state_path, state)


def _merge_manifest(run_dir: Path, result: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest.setdefault("run_id", result.get("run_id"))
    manifest["status"] = result.get("status", manifest.get("status"))
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["environment"] = manifest.get("environment") or _environment_summary()
    for key in (
        "station",
        "start",
        "end",
        "split_ranges",
        "version",
        "download",
        "audit",
        "train_result",
        "candidate_model_package",
        "production_weather_data_package",
        "lab_input_package",
    ):
        if key in result:
            manifest[key] = result[key]
    if result.get("status") == "completed" and result.get("version"):
        manifest["latest_packaged_version"] = result["version"]
    _write_json(manifest_path, manifest)


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


def export_lab_input_package(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_path = artifact_dir / f"{root.name}_lab_training_input.tar.gz"
    required = [
        root / "data_store",
        root / "audit_summary.json",
        root / "stage_state.json",
        root / "manifest.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"lab input package missing required artifact: {path}")

    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as tar:
        for candidate in required:
            if candidate.is_dir():
                for path in sorted(candidate.rglob("*")):
                    if path.is_file():
                        arcname = path.relative_to(root).as_posix()
                        tar.add(path, arcname=arcname)
                        members.append(arcname)
            else:
                arcname = candidate.relative_to(root).as_posix()
                tar.add(candidate, arcname=arcname)
                members.append(arcname)
    return {
        "package_path": str(package_path),
        "sha256": _sha256_file(package_path),
        "members": sorted(members),
    }


def _default_audit(**kwargs: Any) -> dict[str, Any]:
    from scripts.audit_training_data import audit_training_readiness

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
    parser = argparse.ArgumentParser(description="Run a staged remote candidate TFT training batch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-root", default="runs")
        p.add_argument("--run-id", required=True)

    def add_station_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--station", default="KLGA")

    def add_date_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", required=True)
        p.add_argument("--end", required=True)

    def add_split_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--train", required=True, help="start:end")
        p.add_argument("--val", required=True, help="start:end")
        p.add_argument("--test", required=True, help="start:end")

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--version", required=True)
        p.add_argument("--min-epochs", type=int, default=1)
        p.add_argument("--max-epochs", type=int, default=1)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--precision", default="32")
        p.add_argument("--num-workers", type=int, default=0)

    p_download = subparsers.add_parser("download", help="download and clean weather data")
    add_run_args(p_download)
    add_station_arg(p_download)
    add_date_args(p_download)
    p_download.add_argument("--workers", type=int, default=12)

    p_audit = subparsers.add_parser("audit", help="audit local run data readiness")
    add_run_args(p_audit)
    add_station_arg(p_audit)
    add_split_args(p_audit)

    p_train = subparsers.add_parser("train", help="train a candidate TFT model")
    add_run_args(p_train)
    add_station_arg(p_train)
    add_split_args(p_train)
    add_train_args(p_train)

    p_package = subparsers.add_parser("package", help="package a trained candidate model")
    add_run_args(p_package)
    p_package.add_argument("--version", required=True)
    p_package.add_argument("--model-only", action="store_true")

    p_export_lab = subparsers.add_parser(
        "export-lab-input",
        help="export audited training input for a lab training machine",
    )
    add_run_args(p_export_lab)

    p_all = subparsers.add_parser("all", help="download, audit, train, and package")
    add_run_args(p_all)
    add_station_arg(p_all)
    add_date_args(p_all)
    add_split_args(p_all)
    add_train_args(p_all)
    p_all.add_argument("--workers", type=int, default=12)

    args = parser.parse_args()
    if args.command == "download":
        result = run_download_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            station=args.station,
            start=args.start,
            end=args.end,
            workers=args.workers,
        )
    elif args.command == "audit":
        result = run_audit_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            station=args.station,
            split_ranges={
                "train": _parse_split(args.train),
                "val": _parse_split(args.val),
                "test": _parse_split(args.test),
            },
        )
    elif args.command == "train":
        result = run_train_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            station=args.station,
            version=args.version,
            split_ranges={
                "train": _parse_split(args.train),
                "val": _parse_split(args.val),
                "test": _parse_split(args.test),
            },
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            deterministic=args.deterministic,
            precision=args.precision,
            num_workers=args.num_workers,
        )
    elif args.command == "package":
        result = run_package_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            version=args.version,
            model_only=args.model_only,
        )
    elif args.command == "export-lab-input":
        result = run_export_lab_input_stage(
            run_root=args.run_root,
            run_id=args.run_id,
        )
    else:
        split_ranges = {
            "train": _parse_split(args.train),
            "val": _parse_split(args.val),
            "test": _parse_split(args.test),
        }
        run_download_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            station=args.station,
            start=args.start,
            end=args.end,
            workers=args.workers,
        )
        result = run_audit_stage(
            run_root=args.run_root,
            run_id=args.run_id,
            station=args.station,
            split_ranges=split_ranges,
        )
        if result["status"] != "audit_failed":
            run_train_stage(
                run_root=args.run_root,
                run_id=args.run_id,
                station=args.station,
                version=args.version,
                split_ranges=split_ranges,
                min_epochs=args.min_epochs,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                deterministic=args.deterministic,
                precision=args.precision,
                num_workers=args.num_workers,
            )
            result = run_package_stage(
                run_root=args.run_root,
                run_id=args.run_id,
                version=args.version,
            )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
