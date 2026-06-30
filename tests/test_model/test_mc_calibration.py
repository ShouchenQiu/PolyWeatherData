from __future__ import annotations

import json

import numpy as np
import pytest

from model.mc_calibration import McCalibration


def _calibration(model_sha256: str) -> McCalibration:
    return McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        max_bias_f=0.75,
        model_sha256=model_sha256,
        fit_split="val",
        fit_start="2026-04-01",
        fit_end="2026-04-30",
        sample_count=720,
    )


def test_mc_calibration_round_trip_is_bound_to_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint-a")
    calibration = _calibration(McCalibration.sha256(checkpoint))

    calibration.save(tmp_path)
    loaded = McCalibration.load(tmp_path, checkpoint)

    np.testing.assert_allclose(loaded.correlation, np.eye(48))
    np.testing.assert_allclose(loaded.sigma_scales, calibration.sigma_scales)
    assert loaded.max_bias_f == pytest.approx(0.75)


def test_mc_calibration_rejects_checkpoint_hash_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint-a")
    _calibration(McCalibration.sha256(checkpoint)).save(tmp_path)
    checkpoint.write_bytes(b"checkpoint-b")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        McCalibration.load(tmp_path, checkpoint)


def test_mc_calibration_rejects_non_positive_definite_matrix(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calibration = _calibration(McCalibration.sha256(checkpoint))
    calibration.save(tmp_path)
    np.savez_compressed(
        tmp_path / "mc_calibration.npz",
        correlation=np.ones((48, 48)),
        sigma_scales=calibration.sigma_scales,
    )

    with pytest.raises(ValueError, match="positive-definite"):
        McCalibration.load(tmp_path, checkpoint)
