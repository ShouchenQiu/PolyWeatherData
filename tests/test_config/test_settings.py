"""Tests for config/settings.py -- F01 Settings Cleanup."""
import pytest

from config.settings import Settings


# ---------------------------------------------------------------------------
# AC1: Removed fields are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "klga_grid_row",
    "klga_grid_col",
    "mc_rho",
    "mc_bias_corr_c",
    "cutoff_hour_utc",
    "min_hours_before_cutoff",
    "forecast_interval_min",
    "clob_window_min",
])
def test_removed_field_absent(field):
    s = Settings()
    assert not hasattr(s, field), f"Field {field!r} should have been removed"


# ---------------------------------------------------------------------------
# AC2: mc_n_samp default changed to 10000
# ---------------------------------------------------------------------------

def test_mc_n_samp_default():
    assert Settings().mc_n_samp == 10000


# ---------------------------------------------------------------------------
# AC3: New fields have correct types and defaults
# ---------------------------------------------------------------------------

def test_trading_loop_interval_s():
    assert Settings().trading_loop_interval_s == 300


def test_staleness_threshold_min():
    assert Settings().staleness_threshold_min == 65


def test_staleness_edge_bump():
    assert abs(Settings().staleness_edge_bump - 0.02) < 1e-9


def test_hrrr_source_priority():
    assert Settings().hrrr_source_priority == ["s3", "nomads"]


def test_max_open_per_token():
    assert Settings().max_open_per_token == 1


def test_cold_start_hours():
    assert Settings().cold_start_hours == 72


def test_enable_sniper_default():
    assert Settings().enable_sniper is False


def test_train_start():
    assert Settings().train_start == "2024-10-01"


def test_train_end():
    assert Settings().train_end == "2025-12-31"


def test_val_start():
    assert Settings().val_start == "2026-01-01"


def test_val_end():
    assert Settings().val_end == "2026-02-28"


# ---------------------------------------------------------------------------
# V2F02 AC: V1 Polymarket credential fields removed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "polymarket_api_key",
    "polymarket_api_secret",
    "polymarket_passphrase",
    "polymarket_funder",
    "polymarket_proxy_address",
    "chain_id",
])
def test_v1_credential_field_absent(field):
    s = Settings()
    assert not hasattr(s, field), f"V1 credential field {field!r} must be removed"


# ---------------------------------------------------------------------------
# V2F02 AC: V2 Polymarket credential fields added
# ---------------------------------------------------------------------------

def test_polymarket_private_key_default_empty():
    import os
    os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    assert Settings().polymarket_private_key == ""


def test_polymarket_private_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xdeadbeef")
    assert Settings().polymarket_private_key == "0xdeadbeef"


def test_polymarket_wallet_address_default_empty():
    import os
    os.environ.pop("POLYMARKET_WALLET_ADDRESS", None)
    assert Settings().polymarket_wallet_address == ""


def test_polymarket_wallet_address_reads_from_env(monkeypatch):
    monkeypatch.setenv("POLYMARKET_WALLET_ADDRESS", "0xabc123")
    assert Settings().polymarket_wallet_address == "0xabc123"


def test_test_start():
    assert Settings().test_start == "2026-03-01"


def test_test_end():
    assert Settings().test_end == "2026-04-30"
