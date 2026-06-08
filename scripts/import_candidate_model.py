"""Import a remote candidate model package into the local model store."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from db.schema import DbClient


def import_candidate_model(
    package_path: Path | str,
    model_store: Path | str,
    db_path: Path | str,
) -> dict[str, Any]:
    package = Path(package_path)
    store = Path(model_store)
    with tarfile.open(package, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            if "_decoder_cache" in Path(member.name).parts:
                raise ValueError(
                    "candidate model package must not contain _decoder_cache; "
                    f"found {member.name}"
                )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _safe_extract(tar, tmp_path)
            version_dirs = list((tmp_path / "model_store").glob("*"))
            if len(version_dirs) != 1:
                raise ValueError("candidate package must contain exactly one model_store version")
            source_version_dir = version_dirs[0]
            version = source_version_dir.name
            _require_candidate_files(source_version_dir)
            dest = store / version
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_version_dir, dest)
            meta = json.loads((dest / "meta.json").read_text(encoding="ascii"))
            metrics = meta.get("metrics", {})
            db = DbClient(str(db_path))
            try:
                db.register_model_version(
                    version=version,
                    checkpoint_path=str(dest / "model.ckpt"),
                    train_mae=float(metrics.get("train_mae", 0.0)),
                    test_mae=float(metrics.get("test_mae", 0.0)),
                    created_at=str(meta.get("created_at", "")),
                )
            finally:
                db.close()
            return {
                "version": version,
                "model_dir": str(dest),
                "db_path": str(db_path),
                "active": False,
            }


def _require_candidate_files(version_dir: Path) -> None:
    for name in ("model.ckpt", "training_dataset.pkl", "meta.json", "training_metadata.json"):
        if not (version_dir / name).exists():
            raise FileNotFoundError(f"candidate package missing {name}")


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in tar.getmembers():
        member_path = (target / member.name).resolve()
        if target_resolved not in (member_path, *member_path.parents):
            raise ValueError(f"unsafe package member path: {member.name}")
    tar.extractall(target)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Import a candidate model package")
    parser.add_argument("--package", required=True, dest="package_path")
    parser.add_argument("--model-store", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    result = import_candidate_model(args.package_path, args.model_store, args.db)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
