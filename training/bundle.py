"""Build a training-only source bundle for a lower-trust lab machine."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_FILES = (
    "pyproject.toml",
    "config/__init__.py",
    "config/settings.py",
    "config/stations.py",
    "db/__init__.py",
    "db/schema.py",
    "data/__init__.py",
    "data/manager.py",
    "data/forecast/__init__.py",
    "data/forecast/base.py",
    "data/forecast/hrrr_fill.py",
    "data/storage/__init__.py",
    "data/storage/cleaner.py",
    "data/storage/parquet_store.py",
    "model/__init__.py",
    "model/features.py",
    "model/contract.py",
    "model/tft.py",
    "model/store/__init__.py",
    "model/store/registry.py",
    "training/__init__.py",
    "training/artifacts.py",
    "training/config.py",
    "training/dataset.py",
    "training/package.py",
    "training/trainer.py",
    "training/workspace.py",
    "scripts/__init__.py",
    "scripts/package_candidate_model.py",
    "scripts/train_tft.py",
)


def build_training_bundle(
    output_path: Path | str,
    project_root: Path | str = _PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        source_files: list[str] = []
        for rel in _SOURCE_FILES:
            src = root / rel
            if not src.exists():
                continue
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            source_files.append(rel)
        source_files = sorted(set(source_files))
        manifest = {
            "bundle_kind": "training_factory_lab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_output(root, ["git", "rev-parse", "HEAD"]),
            "git_status_short": _git_output(root, ["git", "status", "--short"]),
            "supported_commands": [
                "python -m scripts.train_tft",
                "python -m scripts.package_candidate_model",
            ],
            "included_roots": [
                "config",
                "data",
                "db",
                "model",
                "training",
                "scripts",
            ],
            "included_scripts": [
                rel for rel in _SOURCE_FILES if rel.startswith("scripts/")
            ],
            "excluded_roots": [
                "core",
                "loops",
                "events",
                "snipers",
                "ui",
                "backtest",
                "tests",
                "docs",
            ],
            "source_files": source_files,
            "source_file_sha256": {
                rel: _sha256_file(staging / rel) for rel in source_files
            },
        }
        (staging / "TRAINING_BUNDLE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="ascii",
        )
        with tarfile.open(output, "w:gz") as tar:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(staging).as_posix())
    return {
        "bundle_path": str(output),
        "sha256": _sha256_file(output),
        "source_file_count": len(source_files),
    }


def _git_output(root: Path, cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
