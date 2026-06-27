"""Package candidate models, weather data, and lab training inputs."""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.workspace import TrainingWorkspace


_CANDIDATE_FILES = (
    "model.ckpt",
    "training_dataset.pkl",
    "meta.json",
    "training_metadata.json",
    "experiment_config.json",
    "model_contract.json",
)


def package_candidate_model(
    run_dir: Path | str,
    version: str,
) -> dict[str, Any]:
    root = Path(run_dir)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_path = artifact_dir / f"{version}.tar.gz"
    version_dir = root / "model_store" / version
    for name in _CANDIDATE_FILES:
        path = version_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"candidate package missing required artifact: {path}"
            )

    candidates = (
        version_dir,
        root / "manifest.json",
        root / "audit_summary.json",
        root / "train_result.json",
    )
    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as archive:
        for candidate in candidates:
            if not candidate.exists():
                continue
            paths = (
                sorted(candidate.rglob("*"))
                if candidate.is_dir()
                else [candidate]
            )
            for path in paths:
                if not path.is_file() or "_decoder_cache" in path.parts:
                    continue
                arcname = path.relative_to(root).as_posix()
                archive.add(path, arcname=arcname)
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
    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as archive:
        for subdir in ("hrrr", "obs"):
            source = root / "data_store" / subdir
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(root).as_posix()
                    archive.add(path, arcname=arcname)
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
        archive.add(manifest_path, arcname="weather_data_manifest.json")
        members.append("weather_data_manifest.json")
    return {
        "package_path": str(package_path),
        "sha256": _sha256_file(package_path),
        "members": sorted(members),
    }


def export_lab_input_package(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    workspace = TrainingWorkspace(root)
    workspace.require_ready_audit()
    required = (
        workspace.data_store_dir,
        workspace.audit_summary_path,
        workspace.stage_state_path,
        workspace.manifest_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"lab input package missing required artifact: {path}"
            )

    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)
    package_path = (
        workspace.artifacts_dir / f"{workspace.run_id}_lab_training_input.tar.gz"
    )
    members: list[str] = []
    with tarfile.open(package_path, "w:gz") as archive:
        for candidate in required:
            paths = (
                sorted(candidate.rglob("*"))
                if candidate.is_dir()
                else [candidate]
            )
            for path in paths:
                if not path.is_file():
                    continue
                arcname = path.relative_to(root).as_posix()
                archive.add(path, arcname=arcname)
                members.append(arcname)
    return {
        "package_path": str(package_path),
        "sha256": _sha256_file(package_path),
        "members": sorted(members),
    }


def package_workspace(
    run_dir: Path | str,
    version: str,
    include_weather_data: bool = False,
) -> dict[str, Any]:
    workspace = TrainingWorkspace(run_dir)
    if not (workspace.run_dir / "train_result.json").exists():
        raise FileNotFoundError("package requires train_result.json")
    started_at = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "packaged",
        "version": version,
        "candidate_model_package": package_candidate_model(
            workspace.run_dir,
            version,
        ),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if include_weather_data:
        result["production_weather_data_package"] = (
            package_production_weather_data(workspace.run_dir)
        )
    workspace.record_stage("package", result)
    return result


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
