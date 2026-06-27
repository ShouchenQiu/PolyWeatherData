from __future__ import annotations

import json
import tarfile

from training.bundle import build_training_bundle


def test_build_remote_training_bundle_includes_training_code_and_excludes_production(tmp_path):
    output = tmp_path / "training-bundle.tar.gz"

    summary = build_training_bundle(output)

    assert output.exists()
    assert summary["sha256"]
    with tarfile.open(output, "r:gz") as tar:
        members = set(tar.getnames())
        manifest = json.loads(
            tar.extractfile("TRAINING_BUNDLE_MANIFEST.json").read().decode("ascii")  # type: ignore[union-attr]
        )
    assert "scripts/train_tft.py" in members
    assert "scripts/package_candidate_model.py" in members
    assert "config/settings.py" in members
    assert "data/storage/parquet_store.py" in members
    assert "training/__init__.py" in members
    assert "training/artifacts.py" in members
    assert "training/config.py" in members
    assert "training/dataset.py" in members
    assert "training/package.py" in members
    assert "training/trainer.py" in members
    assert "training/workspace.py" in members
    assert "model/contract.py" in members
    assert "db/schema.py" in members
    assert "core/observability.py" not in members
    assert "loops/forecast_loop.py" not in members
    assert "events/discovery.py" not in members
    assert "data/clob/orderbook.py" not in members
    assert "data/forecast/hrrr_s3.py" not in members
    assert "data/forecast/hrrr_zarr.py" not in members
    assert "data/forecast/_grib_source.py" not in members
    assert "model/pipeline.py" not in members
    assert "scripts/seed_demo_data.py" not in members
    assert "scripts/batch_download_iem.py" not in members
    assert "scripts/batch_download_zarr.py" not in members
    assert "scripts/batch_prepare_training_data.py" not in members
    assert "scripts/smoke_bootstrap.py" not in members
    assert "scripts/audit_training_data.py" not in members
    assert "scripts/audit_weather_backfill.py" not in members
    assert not any(member.startswith("tests/") for member in members)
    assert not any(member.startswith("docs/") for member in members)
    assert not any(member.endswith(".env") or member == ".env" for member in members)
    assert manifest["bundle_kind"] == "training_factory_lab"
    assert manifest["supported_commands"] == [
        "python -m scripts.train_tft",
        "python -m scripts.package_candidate_model",
    ]
    assert "training/trainer.py" in manifest["source_files"]
    assert manifest["source_file_sha256"]["training/trainer.py"]
