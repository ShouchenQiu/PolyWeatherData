"""Build a train-only source bundle for a lower-trust lab machine."""
from __future__ import annotations

import argparse
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
    "model/model_contract.py",
    "model/tft.py",
    "model/training/__init__.py",
    "model/training/artifacts.py",
    "model/training/config.py",
    "model/training/trainer.py",
    "model/training_config.py",
    "model/training_data.py",
    "model/store/__init__.py",
    "model/store/registry.py",
    "scripts/__init__.py",
    "scripts/run_remote_training.py",
    "scripts/train_tft.py",
)


def build_remote_training_bundle(
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
            "bundle_kind": "lab_train_only",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_output(root, ["git", "rev-parse", "HEAD"]),
            "git_status_short": _git_output(root, ["git", "status", "--short"]),
            "supported_commands": ["train", "package --model-only"],
            "included_roots": ["config", "data", "db", "model", "scripts"],
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
        (staging / "REMOTE_BUNDLE_MANIFEST.json").write_text(
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


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal remote training source bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_remote_training_bundle(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
