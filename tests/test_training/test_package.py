import json
import tarfile

import pytest

from training.package import (
    export_lab_input_package,
    package_candidate_model,
    package_production_weather_data,
)
from training.workspace import TrainingWorkspace


def test_candidate_package_contains_model_evidence_and_excludes_decoder_cache(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    version = "candidate-v1"
    version_dir = run_dir / "model_store" / version
    version_dir.mkdir(parents=True)
    for name in (
        "model.ckpt",
        "training_dataset.pkl",
        "meta.json",
        "training_metadata.json",
        "experiment_config.json",
        "model_contract.json",
        "candidate_lineage.json",
    ):
        payload = (
            json.dumps({"schema_version": 1})
            if name == "model_contract.json"
            else name
        )
        (version_dir / name).write_text(payload, encoding="ascii")
    for name in ("manifest.json", "audit_summary.json", "train_result.json"):
        (run_dir / name).write_text("{}", encoding="ascii")
    cache = run_dir / "data_store" / "_decoder_cache" / "KLGA" / "2026-01.parquet"
    cache.parent.mkdir(parents=True)
    cache.write_text("cache", encoding="ascii")

    result = package_candidate_model(run_dir, version)

    assert result["sha256"]
    assert not any("_decoder_cache" in member for member in result["members"])
    with tarfile.open(result["package_path"], "r:gz") as archive:
        assert f"model_store/{version}/model_contract.json" in archive.getnames()
        assert f"model_store/{version}/candidate_lineage.json" in archive.getnames()


def test_v2_candidate_package_requires_mc_calibration(tmp_path) -> None:
    run_dir = tmp_path / "run"
    version_dir = run_dir / "model_store" / "candidate-v2"
    version_dir.mkdir(parents=True)
    for name in (
        "model.ckpt",
        "training_dataset.pkl",
        "meta.json",
        "training_metadata.json",
        "experiment_config.json",
        "candidate_lineage.json",
    ):
        (version_dir / name).write_text(name, encoding="ascii")
    (version_dir / "model_contract.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="ascii",
    )

    with pytest.raises(FileNotFoundError, match="mc_calibration"):
        package_candidate_model(run_dir, "candidate-v2")


def test_production_weather_package_excludes_training_decoder_cache(tmp_path) -> None:
    run_dir = tmp_path / "run"
    for relative in (
        "data_store/hrrr/KLGA/2026-Q1.parquet",
        "data_store/obs/KLGA/2026-Q1.parquet",
        "data_store/_decoder_cache/KLGA/2026-01.parquet",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="ascii")

    result = package_production_weather_data(run_dir)

    assert "data_store/hrrr/KLGA/2026-Q1.parquet" in result["members"]
    assert "data_store/obs/KLGA/2026-Q1.parquet" in result["members"]
    assert not any("_decoder_cache" in member for member in result["members"])


def test_lab_input_package_includes_decoder_cache_and_audit_evidence(tmp_path) -> None:
    workspace = TrainingWorkspace(tmp_path / "run")
    workspace.ensure()
    cache = (
        workspace.data_store_dir
        / "_decoder_cache"
        / "KLGA"
        / "2026-01.parquet"
    )
    cache.parent.mkdir(parents=True)
    cache.write_text("cache", encoding="ascii")
    workspace.write_json(workspace.audit_summary_path, {"ready": True})
    workspace.write_json(workspace.stage_state_path, {"stages": {}})
    workspace.write_json(workspace.manifest_path, {"run_id": workspace.run_id})
    audited_frame = workspace.audit_frames_dir / "train.parquet"
    audited_frame.parent.mkdir(parents=True)
    audited_frame.write_text("frame", encoding="ascii")

    result = export_lab_input_package(workspace.run_dir)

    assert "data_store/_decoder_cache/KLGA/2026-01.parquet" in result["members"]
    assert "artifacts/audit_frames/train.parquet" in result["members"]
    assert "audit_summary.json" in result["members"]
