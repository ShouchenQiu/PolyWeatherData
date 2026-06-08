from __future__ import annotations

import json
from pathlib import Path

from scripts.run_remote_training import (
    package_candidate_model,
    package_production_weather_data,
    run_remote_training,
)


def test_run_remote_training_stops_when_audit_fails(tmp_path):
    calls = {"train": 0}

    def audit_fn(**_kwargs):
        return {"ready": False, "failures": ["missing obs"]}

    def train_fn(**_kwargs):
        calls["train"] += 1
        return {"version": "should_not_run"}

    result = run_remote_training(
        run_root=tmp_path / "runs",
        run_id="r1",
        station="KLGA",
        start="2026-01-01",
        end="2026-01-01",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-01", "2026-01-01"),
            "test": ("2026-01-01", "2026-01-01"),
        },
        version="candidate_v1",
        audit_fn=audit_fn,
        train_fn=train_fn,
        download_fn=lambda **_kwargs: {"skipped": True},
        package_fn=lambda **_kwargs: {"package": "unused"},
    )

    assert result["status"] == "audit_failed"
    assert calls["train"] == 0
    assert (tmp_path / "runs" / "r1" / "failed_audit_summary.json").exists()


def test_run_remote_training_packages_candidate_when_audit_passes(tmp_path):
    def audit_fn(**_kwargs):
        return {"ready": True, "data_order_hash": "abc"}

    def train_fn(**kwargs):
        version = kwargs["version"]
        model_dir = Path(kwargs["model_store"]) / version
        model_dir.mkdir(parents=True)
        (model_dir / "model.ckpt").write_text("ckpt", encoding="ascii")
        (model_dir / "training_dataset.pkl").write_text("pkl", encoding="ascii")
        (model_dir / "meta.json").write_text("{}", encoding="ascii")
        (model_dir / "training_metadata.json").write_text("{}", encoding="ascii")
        return {"version": version, "test_mae": 1.0}

    result = run_remote_training(
        run_root=tmp_path / "runs",
        run_id="r2",
        station="KLGA",
        start="2026-01-01",
        end="2026-01-01",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-01", "2026-01-01"),
            "test": ("2026-01-01", "2026-01-01"),
        },
        version="candidate_v2",
        audit_fn=audit_fn,
        train_fn=train_fn,
        download_fn=lambda **_kwargs: {"skipped": True},
    )

    manifest = json.loads(
        (tmp_path / "runs" / "r2" / "manifest.json").read_text(encoding="ascii")
    )
    assert result["status"] == "completed"
    assert manifest["version"] == "candidate_v2"
    assert manifest["audit"]["ready"] is True
    assert Path(result["candidate_model_package"]["package_path"]).exists()


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
