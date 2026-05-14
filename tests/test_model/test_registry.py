"""Tests for model/store/registry.py -- ModelRegistry max_bias and _db_latest_row."""
from __future__ import annotations

import numpy as np
import pytest

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
