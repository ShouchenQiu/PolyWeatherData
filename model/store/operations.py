"""Explicit candidate import and production model promotion operations."""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.schema import DbClient
from model.store.registry import ModelRegistry


_CANDIDATE_FILES = (
    "model.ckpt",
    "training_dataset.pkl",
    "meta.json",
    "training_metadata.json",
    "experiment_config.json",
    "model_contract.json",
    "candidate_lineage.json",
)


def import_candidate_model(
    package_path: Path | str,
    model_store: Path | str,
    db_path: Path | str,
) -> dict[str, Any]:
    """Import one reviewed candidate without activating it."""
    package = Path(package_path)
    store = Path(model_store)
    with tarfile.open(package, "r:*") as archive:
        for member in archive.getmembers():
            if "_decoder_cache" in Path(member.name).parts:
                raise ValueError(
                    "candidate model package must not contain _decoder_cache; "
                    f"found {member.name}"
                )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _safe_extract(archive, temp_path)
            version_dirs = list((temp_path / "model_store").glob("*"))
            if len(version_dirs) != 1:
                raise ValueError(
                    "candidate package must contain exactly one model_store version"
                )
            source_version_dir = version_dirs[0]
            _require_candidate_files(source_version_dir)
            lineage = json.loads(
                (source_version_dir / "candidate_lineage.json").read_text(
                    encoding="ascii"
                )
            )
            if lineage.get("state") != "inactive/review_required":
                raise ValueError(
                    "candidate lineage must be inactive/review_required"
                )
            version = source_version_dir.name
            destination = store / version
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_version_dir, destination)

    meta = json.loads((destination / "meta.json").read_text(encoding="ascii"))
    metrics = meta.get("metrics", {})
    db = DbClient(str(db_path))
    try:
        db.register_model_version(
            version=version,
            checkpoint_path=str(destination / "model.ckpt"),
            train_mae=float(metrics.get("train_mae", 0.0)),
            test_mae=float(metrics.get("test_mae", 0.0)),
            created_at=str(meta.get("created_at", "")),
        )
    finally:
        db.close()
    return {
        "version": version,
        "model_dir": str(destination),
        "db_path": str(db_path),
        "active": False,
    }


def promote_model_version(
    model_store: Path | str,
    db_path: Path | str,
    version: str,
) -> dict[str, Any]:
    """Activate one locally reviewed model and append promotion evidence."""
    db = DbClient(str(db_path))
    try:
        ModelRegistry(Path(model_store), db).rollback(version)
    finally:
        db.close()
    promoted_at = datetime.now(timezone.utc).isoformat()
    audit_path = Path(model_store) / "promotion_audit.log"
    with audit_path.open("a", encoding="ascii") as handle:
        handle.write(
            json.dumps(
                {"version": version, "promoted_at": promoted_at},
                sort_keys=True,
            )
            + "\n"
        )
    return {
        "active_version": version,
        "promoted_at": promoted_at,
        "audit_log": str(audit_path),
    }


def _require_candidate_files(version_dir: Path) -> None:
    for name in _CANDIDATE_FILES:
        if not (version_dir / name).exists():
            raise FileNotFoundError(f"candidate package missing {name}")


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in archive.getmembers():
        member_path = (target / member.name).resolve()
        if target_resolved not in (member_path, *member_path.parents):
            raise ValueError(f"unsafe package member path: {member.name}")
    archive.extractall(target)
