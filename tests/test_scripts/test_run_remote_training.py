from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import scripts.run_remote_training as remote_training
from scripts.run_remote_training import (
    export_lab_input_package,
    package_candidate_model,
    package_production_weather_data,
    run_audit_stage,
    run_download_stage,
    run_export_lab_input_stage,
    run_package_stage,
    run_train_stage,
)


def test_download_stage_only_calls_download(tmp_path):
    calls: list[str] = []

    result = run_download_stage(
        run_root=tmp_path / "runs",
        run_id="r_download",
        station="KLGA",
        start="2026-01-01",
        end="2026-01-01",
        workers=2,
        download_fn=lambda **_kwargs: calls.append("download") or {"rows": 24},
    )

    run_dir = tmp_path / "runs" / "r_download"
    state = json.loads((run_dir / "stage_state.json").read_text(encoding="ascii"))
    assert calls == ["download"]
    assert result["status"] == "downloaded"
    assert (run_dir / "download_summary.json").exists()
    assert state["stages"]["download"]["status"] == "downloaded"


def test_cli_download_dispatches_only_download_stage(tmp_path, monkeypatch, capsys):
    calls: list[str] = []

    def fake_download_stage(**kwargs):
        calls.append("download")
        return {"status": "downloaded", "run_id": kwargs["run_id"]}

    monkeypatch.setattr(remote_training, "run_download_stage", fake_download_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_training",
            "download",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "r_cli",
            "--station",
            "KLGA",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    remote_training._cli()

    assert calls == ["download"]
    assert json.loads(capsys.readouterr().out)["status"] == "downloaded"


def test_cli_all_stops_when_audit_fails(tmp_path, monkeypatch, capsys):
    calls: list[str] = []

    def fake_download_stage(**_kwargs):
        calls.append("download")
        return {"status": "downloaded"}

    def fake_audit_stage(**_kwargs):
        calls.append("audit")
        return {"status": "audit_failed", "audit": {"ready": False}}

    def fake_train_stage(**_kwargs):
        calls.append("train")
        return {"status": "trained"}

    def fake_package_stage(**_kwargs):
        calls.append("package")
        return {"status": "completed"}

    monkeypatch.setattr(remote_training, "run_download_stage", fake_download_stage)
    monkeypatch.setattr(remote_training, "run_audit_stage", fake_audit_stage)
    monkeypatch.setattr(remote_training, "run_train_stage", fake_train_stage)
    monkeypatch.setattr(remote_training, "run_package_stage", fake_package_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_training",
            "all",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "r_all",
            "--station",
            "KLGA",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-01",
            "--train",
            "2026-01-01:2026-01-01",
            "--val",
            "2026-01-01:2026-01-01",
            "--test",
            "2026-01-01:2026-01-01",
            "--version",
            "candidate_v1",
        ],
    )

    remote_training._cli()

    assert calls == ["download", "audit"]
    assert json.loads(capsys.readouterr().out)["status"] == "audit_failed"


def test_cli_all_runs_ordered_stages_when_audit_passes(tmp_path, monkeypatch, capsys):
    calls: list[str] = []

    def fake_download_stage(**_kwargs):
        calls.append("download")
        return {"status": "downloaded"}

    def fake_audit_stage(**_kwargs):
        calls.append("audit")
        return {"status": "audited", "audit": {"ready": True}}

    def fake_train_stage(**_kwargs):
        calls.append("train")
        return {"status": "trained"}

    def fake_package_stage(**_kwargs):
        calls.append("package")
        return {"status": "completed"}

    monkeypatch.setattr(remote_training, "run_download_stage", fake_download_stage)
    monkeypatch.setattr(remote_training, "run_audit_stage", fake_audit_stage)
    monkeypatch.setattr(remote_training, "run_train_stage", fake_train_stage)
    monkeypatch.setattr(remote_training, "run_package_stage", fake_package_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_training",
            "all",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "r_all",
            "--station",
            "KLGA",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-01",
            "--train",
            "2026-01-01:2026-01-01",
            "--val",
            "2026-01-01:2026-01-01",
            "--test",
            "2026-01-01:2026-01-01",
            "--version",
            "candidate_v1",
        ],
    )

    remote_training._cli()

    assert calls == ["download", "audit", "train", "package"]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_package_passes_model_only_flag(tmp_path, monkeypatch, capsys):
    calls: list[dict] = []

    def fake_package_stage(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(remote_training, "run_package_stage", fake_package_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_training",
            "package",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "r_pkg",
            "--version",
            "candidate_v1",
            "--model-only",
        ],
    )

    remote_training._cli()

    assert calls[0]["model_only"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_export_lab_input_dispatches_stage(tmp_path, monkeypatch, capsys):
    calls: list[dict] = []

    def fake_export_stage(**kwargs):
        calls.append(kwargs)
        return {"status": "lab_input_exported"}

    monkeypatch.setattr(remote_training, "run_export_lab_input_stage", fake_export_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_training",
            "export-lab-input",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "r_lab",
        ],
    )

    remote_training._cli()

    assert calls == [{"run_root": str(tmp_path / "runs"), "run_id": "r_lab"}]
    assert json.loads(capsys.readouterr().out)["status"] == "lab_input_exported"


def test_audit_stage_writes_summary_and_failed_marker(tmp_path):
    run_dir = tmp_path / "runs" / "r_audit"
    (run_dir / "data_store").mkdir(parents=True)

    result = run_audit_stage(
        run_root=tmp_path / "runs",
        run_id="r_audit",
        station="KLGA",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-01", "2026-01-01"),
            "test": ("2026-01-01", "2026-01-01"),
        },
        audit_fn=lambda **_kwargs: {"ready": False, "failures": ["missing"]},
    )

    assert result["status"] == "audit_failed"
    assert (run_dir / "audit_summary.json").exists()
    assert (run_dir / "failed_audit_summary.json").exists()


def test_train_stage_requires_ready_audit(tmp_path):
    run_dir = tmp_path / "runs" / "r_train"
    run_dir.mkdir(parents=True)

    try:
        run_train_stage(
            run_root=tmp_path / "runs",
            run_id="r_train",
            station="KLGA",
            version="v1",
            split_ranges={
                "train": ("2026-01-01", "2026-01-01"),
                "val": ("2026-01-01", "2026-01-01"),
                "test": ("2026-01-01", "2026-01-01"),
            },
            train_fn=lambda **_kwargs: {"version": "v1"},
        )
    except RuntimeError as exc:
        assert "audit_summary" in str(exc)
    else:
        raise AssertionError("expected train stage to require ready audit")


def test_train_stage_reuses_existing_audit_without_download(tmp_path):
    calls: list[str] = []
    run_dir = tmp_path / "runs" / "r_train_ready"
    run_dir.mkdir(parents=True)
    (run_dir / "audit_summary.json").write_text(
        json.dumps({"ready": True}),
        encoding="ascii",
    )

    result = run_train_stage(
        run_root=tmp_path / "runs",
        run_id="r_train_ready",
        station="KLGA",
        version="try02",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-01", "2026-01-01"),
            "test": ("2026-01-01", "2026-01-01"),
        },
        train_fn=lambda **_kwargs: calls.append("train") or {"version": "try02"},
    )

    assert calls == ["train"]
    assert result["status"] == "trained"
    assert json.loads((run_dir / "train_result.json").read_text())["version"] == "try02"


def test_package_stage_requires_trained_model(tmp_path):
    run_dir = tmp_path / "runs" / "r_package"
    run_dir.mkdir(parents=True)

    try:
        run_package_stage(tmp_path / "runs", "r_package", "missing_v")
    except FileNotFoundError as exc:
        assert "train_result" in str(exc) or "model_store" in str(exc)
    else:
        raise AssertionError("expected package stage to require trained artifacts")


def test_package_stage_model_only_skips_production_weather_data(tmp_path):
    run_dir = tmp_path / "runs" / "r_model_only"
    version = "candidate_v1"
    model_dir = run_dir / "model_store" / version
    model_dir.mkdir(parents=True)
    for name in ("model.ckpt", "training_dataset.pkl", "meta.json", "training_metadata.json"):
        (model_dir / name).write_text(name, encoding="ascii")
    (run_dir / "train_result.json").write_text("{}", encoding="ascii")

    result = run_package_stage(
        tmp_path / "runs",
        "r_model_only",
        version,
        model_only=True,
    )

    assert result["status"] == "completed"
    assert "candidate_model_package" in result
    assert "production_weather_data_package" not in result
    assert not (run_dir / "artifacts" / "production_weather_data.tar.gz").exists()


def test_package_candidate_model_excludes_decoder_cache(tmp_path):
    run_dir = tmp_path / "run"
    version = "candidate_v1"
    model_dir = run_dir / "model_store" / version
    model_dir.mkdir(parents=True)
    for name in ("model.ckpt", "training_dataset.pkl", "meta.json", "training_metadata.json"):
        (model_dir / name).write_text(name, encoding="ascii")
    (run_dir / "data_store" / "_decoder_cache" / "KLGA").mkdir(parents=True)
    (run_dir / "data_store" / "_decoder_cache" / "KLGA" / "2026-01.parquet").write_text(
        "debug cache",
        encoding="ascii",
    )
    (run_dir / "audit_summary.json").write_text("{}", encoding="ascii")
    (run_dir / "train_result.json").write_text("{}", encoding="ascii")

    package = package_candidate_model(run_dir=run_dir, version=version)

    assert package["sha256"]
    assert "_decoder_cache" not in package["members"]
    assert f"model_store/{version}/training_dataset.pkl" in package["members"]


def test_package_production_weather_data_includes_only_hrrr_and_obs(tmp_path):
    run_dir = tmp_path / "run"
    hrrr_file = run_dir / "data_store" / "hrrr" / "KLGA" / "2026-Q1.parquet"
    obs_file = run_dir / "data_store" / "obs" / "KLGA" / "2026-Q1.parquet"
    cache_file = run_dir / "data_store" / "_decoder_cache" / "KLGA" / "2026-01.parquet"
    for path in (hrrr_file, obs_file, cache_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="ascii")

    package = package_production_weather_data(run_dir)

    assert "data_store/hrrr/KLGA/2026-Q1.parquet" in package["members"]
    assert "data_store/obs/KLGA/2026-Q1.parquet" in package["members"]
    assert not any("_decoder_cache" in member for member in package["members"])
    assert "weather_data_manifest.json" in package["members"]


def test_export_lab_input_requires_ready_audit(tmp_path):
    run_dir = tmp_path / "runs" / "r_lab"
    run_dir.mkdir(parents=True)

    try:
        run_export_lab_input_stage(tmp_path / "runs", "r_lab")
    except RuntimeError as exc:
        assert "audit_summary" in str(exc)
    else:
        raise AssertionError("expected lab export to require ready audit")


def test_export_lab_input_package_includes_decoder_cache_and_evidence(tmp_path):
    run_dir = tmp_path / "run"
    cache_file = run_dir / "data_store" / "_decoder_cache" / "KLGA" / "2026-01.parquet"
    hrrr_file = run_dir / "data_store" / "hrrr" / "KLGA" / "2026-Q1.parquet"
    obs_file = run_dir / "data_store" / "obs" / "KLGA" / "2026-Q1.parquet"
    for path in (cache_file, hrrr_file, obs_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="ascii")
    (run_dir / "audit_summary.json").write_text(json.dumps({"ready": True}), encoding="ascii")
    (run_dir / "stage_state.json").write_text("{}", encoding="ascii")
    (run_dir / "manifest.json").write_text("{}", encoding="ascii")

    package = export_lab_input_package(run_dir)

    assert package["sha256"]
    assert "data_store/_decoder_cache/KLGA/2026-01.parquet" in package["members"]
    assert "audit_summary.json" in package["members"]
    with tarfile.open(package["package_path"], "r:gz") as tar:
        assert "data_store/hrrr/KLGA/2026-Q1.parquet" in tar.getnames()
