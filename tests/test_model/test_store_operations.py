from __future__ import annotations

import json
import tarfile
from pathlib import Path

from db.schema import DbClient
from model.store.operations import import_candidate_model, promote_model_version


def _make_candidate_package(tmp_path: Path, version: str = "candidate_v1") -> Path:
    root = tmp_path / "src"
    version_dir = root / "model_store" / version
    version_dir.mkdir(parents=True)
    (version_dir / "model.ckpt").write_text("ckpt", encoding="ascii")
    (version_dir / "training_dataset.pkl").write_text("pkl", encoding="ascii")
    (version_dir / "training_metadata.json").write_text("{}", encoding="ascii")
    (version_dir / "experiment_config.json").write_text("{}", encoding="ascii")
    (version_dir / "model_contract.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="ascii",
    )
    (version_dir / "candidate_lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "inactive/review_required",
                "study_id": "study-1",
                "selected_trial_number": 2,
                "trial_snapshot_hash": "snapshot-hash",
                "template_hash": "template-hash",
                "data_order_hash": "data-hash",
            }
        ),
        encoding="ascii",
    )
    (version_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": version,
                "checkpoint_path": str(version_dir / "model.ckpt"),
                "metrics": {"train_mae": 0.5, "test_mae": 0.6},
                "created_at": "2026-06-08T00:00:00+00:00",
            }
        ),
        encoding="ascii",
    )
    package = tmp_path / f"{version}.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        for path in sorted(version_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(root).as_posix())
    return package


def test_import_candidate_model_copies_version_and_writes_local_db_without_active(tmp_path):
    package = _make_candidate_package(tmp_path)
    model_store = tmp_path / "model_store"
    db_path = tmp_path / "polyweather.db"

    result = import_candidate_model(package, model_store, db_path)

    assert result["version"] == "candidate_v1"
    assert (model_store / "candidate_v1" / "model.ckpt").exists()
    assert not (model_store / "active_version.txt").exists()
    db = DbClient(str(db_path))
    try:
        rows = db.query("select version, test_mae from model_versions")
    finally:
        db.close()
    assert rows[0]["version"] == "candidate_v1"
    assert rows[0]["test_mae"] == 0.6


def test_import_candidate_model_rejects_decoder_cache_member(tmp_path):
    package = tmp_path / "bad.tar.gz"
    bad_root = tmp_path / "bad"
    cache_file = bad_root / "data_store" / "_decoder_cache" / "KLGA" / "2026.parquet"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cache", encoding="ascii")
    with tarfile.open(package, "w:gz") as tar:
        tar.add(cache_file, arcname="data_store/_decoder_cache/KLGA/2026.parquet")

    try:
        import_candidate_model(package, tmp_path / "models", tmp_path / "db.sqlite")
    except ValueError as exc:
        assert "_decoder_cache" in str(exc)
    else:
        raise AssertionError("expected decoder cache package rejection")


def test_import_candidate_model_requires_contract_artifacts(tmp_path):
    root = tmp_path / "src"
    version_dir = root / "model_store" / "candidate_v1"
    version_dir.mkdir(parents=True)
    for name in ("model.ckpt", "training_dataset.pkl", "meta.json", "training_metadata.json"):
        (version_dir / name).write_text("x", encoding="ascii")
    package = tmp_path / "candidate_v1.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        for path in sorted(version_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(root).as_posix())

    try:
        import_candidate_model(package, tmp_path / "models", tmp_path / "db.sqlite")
    except FileNotFoundError as exc:
        assert "model_contract.json" in str(exc) or "experiment_config.json" in str(exc)
    else:
        raise AssertionError("expected missing contract artifacts to be rejected")


def test_promote_model_version_writes_active_and_audit_log(tmp_path):
    model_store = tmp_path / "model_store"
    (model_store / "candidate_v1").mkdir(parents=True)
    db_path = tmp_path / "polyweather.db"

    result = promote_model_version(model_store, db_path, "candidate_v1")

    assert result["active_version"] == "candidate_v1"
    assert (model_store / "active_version.txt").read_text(encoding="ascii") == "candidate_v1"
    assert (model_store / "promotion_audit.log").exists()
