"""Tests for model/store/registry.py -- ModelRegistry max_bias and _db_latest_row."""
from __future__ import annotations

import numpy as np
import pytest
from loguru import logger

from db.schema import DbClient
from model.store.registry import ModelRegistry


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    client = DbClient(str(tmp_path / "test.db"))
    yield client
    client.close()


@pytest.fixture
def registry(db, tmp_path):
    return ModelRegistry(store_dir=tmp_path / "registry", db=db)


# -----------------------------------------------------------------------
# _db_latest_row -- uses DbClient public interface only
# -----------------------------------------------------------------------

def test_db_latest_row_returns_none_when_empty(registry):
    assert registry._db_latest_row() is None


def test_db_latest_row_returns_most_recent(registry, db):
    db.register_model_version("v1", "/ck/v1.ckpt", 0.5, 0.6, "2026-01-01T00:00:00")
    db.register_model_version("v2", "/ck/v2.ckpt", 0.4, 0.5, "2026-02-01T00:00:00")
    row = registry._db_latest_row()
    assert row["version"] == "v2"


def test_db_latest_row_does_not_access_private_conn(registry, db):
    """_db_latest_row must use db.query(), not db._conn directly."""
    import inspect
    src = inspect.getsource(registry._db_latest_row)
    assert "_conn" not in src, "_db_latest_row must not access db._conn"


# -----------------------------------------------------------------------
# store_max_bias / load_max_bias round-trip
# -----------------------------------------------------------------------

def test_load_max_bias_default_zero(registry):
    assert registry.load_max_bias("KLGA") == 0.0


def test_store_then_load_max_bias(registry):
    registry.store_max_bias("KLGA", 2.5)
    assert registry.load_max_bias("KLGA") == pytest.approx(2.5)


def test_store_max_bias_overwrites(registry):
    registry.store_max_bias("KLGA", 1.0)
    registry.store_max_bias("KLGA", 3.7)
    assert registry.load_max_bias("KLGA") == pytest.approx(3.7)


def test_store_max_bias_per_station(registry):
    registry.store_max_bias("KLGA", 1.0)
    registry.store_max_bias("KORD", 2.0)
    assert registry.load_max_bias("KLGA") == pytest.approx(1.0)
    assert registry.load_max_bias("KORD") == pytest.approx(2.0)


# -----------------------------------------------------------------------
# calibrate_max_bias
# -----------------------------------------------------------------------

def test_calibrate_max_bias_stores_result(registry):
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal((50, 48))
    bias = registry.calibrate_max_bias("KLGA", residuals)
    assert registry.load_max_bias("KLGA") == pytest.approx(bias)


def test_calibrate_max_bias_value_is_mean_of_row_maxima(registry):
    """calibrate_max_bias = mean of per-row maxima in residuals array."""
    residuals = np.array([[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]])
    # Row maxima: [3, 2, 4]; mean = 3.0
    bias = registry.calibrate_max_bias("KBOS", residuals)
    assert bias == pytest.approx(3.0)


def test_calibrate_max_bias_subsequent_load(registry):
    residuals = np.ones((10, 6)) * 2.0
    registry.calibrate_max_bias("KJFK", residuals)
    assert registry.load_max_bias("KJFK") == pytest.approx(2.0)


# -----------------------------------------------------------------------
# load_covariance -- V2V06
# -----------------------------------------------------------------------

def test_load_covariance_returns_none_when_no_active_version(registry):
    """When no checkpoint is registered, load_covariance logs WARNING and returns None."""
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        result = registry.load_covariance()
    finally:
        logger.remove(sink_id)
    assert result is None
    assert any("covariance" in m.lower() for m in messages)


def test_load_covariance_returns_none_when_file_absent(registry, db, tmp_path):
    """When covariance.npy is absent, load_covariance returns None with a WARNING."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("fake", encoding="ascii")
    registry.publish("v1", ckpt, {"train_mae": 0.5, "test_mae": 0.6})

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        result = registry.load_covariance()
    finally:
        logger.remove(sink_id)

    assert result is None
    assert any("covariance" in m.lower() for m in messages)


def test_publish_can_skip_active_version_for_candidate_model(registry, tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("fake", encoding="ascii")

    registry.publish(
        "candidate_v1",
        ckpt,
        {"train_mae": 0.5, "test_mae": 0.6},
        activate=False,
    )

    assert (registry.store_dir / "candidate_v1" / "model.ckpt").exists()
    assert not (registry.store_dir / "active_version.txt").exists()


def test_load_covariance_returns_matrix_when_present(registry, db, tmp_path):
    """When covariance.npy is present beside the checkpoint, it is returned as ndarray."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("fake", encoding="ascii")
    registry.publish("v1", ckpt, {"train_mae": 0.5, "test_mae": 0.6})

    # Write covariance.npy in the active version dir
    version_dir = registry.store_dir / "v1"
    cov = np.eye(48) * 2.0
    np.save(version_dir / "covariance.npy", cov)

    result = registry.load_covariance()

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (48, 48)
    np.testing.assert_allclose(result, cov)


def test_load_covariance_specific_version(registry, db, tmp_path):
    """load_covariance(version='v2') loads from that specific version dir."""
    for v in ("v1", "v2"):
        ckpt = tmp_path / f"{v}.ckpt"
        ckpt.write_text("fake", encoding="ascii")
        registry.publish(v, ckpt, {"train_mae": 0.5, "test_mae": 0.6})

    # Only put covariance in v2
    cov = np.eye(48) * 3.0
    np.save(registry.store_dir / "v2" / "covariance.npy", cov)

    result = registry.load_covariance(version="v2")
    assert result is not None
    assert result.shape == (48, 48)


def test_load_covariance_raises_on_wrong_shape(registry, db, tmp_path):
    """load_covariance() raises ValueError when covariance.npy is not (48, 48)."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("fake", encoding="ascii")
    registry.publish("v1", ckpt, {"train_mae": 0.5, "test_mae": 0.6})

    # Save a 24x24 matrix instead of 48x48
    bad_cov = np.eye(24)
    np.save(registry.store_dir / "v1" / "covariance.npy", bad_cov)

    with pytest.raises(ValueError, match="48"):
        registry.load_covariance()


# -----------------------------------------------------------------------
# load_latest -- ckpt_path.exists() guard (V2V06 fix 3)
# -----------------------------------------------------------------------

def test_load_latest_raises_when_ckpt_deleted_from_disk(registry, tmp_path):
    """load_latest() raises FileNotFoundError when model.ckpt was removed post-publish."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("fake", encoding="ascii")
    registry.publish("v1", ckpt, {"train_mae": 0.5, "test_mae": 0.6})

    # Simulate the checkpoint file being deleted after publication
    (registry.store_dir / "v1" / "model.ckpt").unlink()

    with pytest.raises(FileNotFoundError, match="model.ckpt"):
        registry.load_latest()
