"""Tests for model/pipeline.py -- ModelPipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.stations import StationConfig
from model.mc import McSampler
from model.pipeline import ModelPipeline


# -----------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------

KLGA = StationConfig(
    station_id          = "KLGA",
    lat                 = 40.77,
    lon                 = -73.87,
    timezone            = "America/New_York",
    tz_offset           = -5,
    hrrr_grid_row_zarr  = 698,
    hrrr_grid_col_zarr  = 1556,
    iem_network         = "NY_ASOS",
    slug_pattern        = "klga-daily-high",
)

_INIT_TIME = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")

_BUCKETS = [
    (float("-inf"), 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
    (50.0, 60.0),
    (60.0, float("inf")),
]


def _make_hrrr_df(init_time: pd.Timestamp) -> pd.DataFrame:
    """48 HRRR decoder rows."""
    rows = [
        {
            "fxx":        fxx,
            "init_time":  init_time,
            "valid_time": init_time + pd.Timedelta(hours=fxx),
            "TMP_2m":     280.0 + fxx * 0.05,       # Kelvin (will be K->C in _prep)
        }
        for fxx in range(1, 49)
    ]
    return pd.DataFrame(rows)


def _make_tft_mock() -> MagicMock:
    """TftModel stub that returns fixed quantiles for fxx 1..48."""
    tft = MagicMock()
    quantiles = {
        fxx: {"p10": 4.0, "p50": 6.0, "p90": 8.0}
        for fxx in range(1, 49)
    }
    tft.predict.return_value = quantiles
    return tft


def _make_encoder_df(init_time: pd.Timestamp) -> pd.DataFrame:
    """72 merged encoder rows (HRRR f01 + obs)."""
    rows = [
        {
            "valid_time": init_time - pd.Timedelta(hours=72 - h),
            "TMP_2m":     5.0 + h * 0.1,
            "tmpc":       6.0 + h * 0.1,
        }
        for h in range(72)
    ]
    return pd.DataFrame(rows)


def _make_data_mgr(init_time: pd.Timestamp) -> MagicMock:
    """DataManager stub returning minimal HRRR and encoder DataFrames."""
    dm = MagicMock()
    dm.fetch_hrrr.return_value = _make_hrrr_df(init_time)
    dm.build_encoder_obs_window.return_value = _make_encoder_df(init_time)
    return dm


# -----------------------------------------------------------------------
# ModelPipeline.run -- return type
# -----------------------------------------------------------------------

def test_run_returns_dict():
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert isinstance(result, dict)


def test_run_result_keys():
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert "station"    in result
    assert "init_time"  in result
    assert "probs"      in result
    assert "p50_high_f" in result
    assert "quantiles"  in result
    assert "last_tick_utc" in result


def test_run_station_id():
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert result["station"] == "KLGA"


def test_run_init_time_preserved():
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert result["init_time"] == _INIT_TIME


def test_run_probs_sum_to_one():
    """Probabilities over exhaustive buckets must sum to ~1."""
    mc       = McSampler(n_samp=2000, rng_seed=42)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    total = sum(result["probs"].values())
    assert abs(total - 1.0) < 1e-9, f"probs sum = {total}"


def test_run_probs_keys_match_buckets():
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert set(result["probs"].keys()) == set([tuple(b) for b in _BUCKETS])


def test_run_p50_high_f_is_float():
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result   = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert isinstance(result["p50_high_f"], float)


def test_run_calls_tft_predict():
    """verify tft.predict is called exactly once."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    tft.predict.assert_called_once()


def test_run_calls_data_mgr():
    """fetch_hrrr, fetch_latest_iem_tick, and build_encoder_obs_window must be called."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    dm.fetch_hrrr.assert_called_once_with(_INIT_TIME, KLGA)
    dm.fetch_latest_iem_tick.assert_called_once()
    dm.build_encoder_obs_window.assert_called_once()


def test_run_does_not_repull_obs_window():
    """ModelPipeline.run() must NOT call the range-based fetch_iem during a tick."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    dm.fetch_iem.assert_not_called()


def test_run_encoder_window_is_72_rows():
    """build_encoder_obs_window must be called with n_hours=72."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    dm.build_encoder_obs_window.assert_called_once()
    _, kwargs = dm.build_encoder_obs_window.call_args
    assert kwargs.get("n_hours") == 72


def test_run_encoder_df_has_time_idx_column():
    """encoder_df passed to tft.predict must have a 'time_idx' column."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    enc_df_arg = tft.predict.call_args[0][0]
    assert "time_idx" in enc_df_arg.columns, "encoder_df must have time_idx column"


def test_run_encoder_df_has_seasonal_columns():
    """encoder_df passed to tft.predict must have month, day_of_year, hour_of_day."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    enc_df_arg = tft.predict.call_args[0][0]
    assert "month"       in enc_df_arg.columns
    assert "day_of_year" in enc_df_arg.columns
    assert "hour_of_day" in enc_df_arg.columns
    assert "sin_hour"    in enc_df_arg.columns
    assert "cos_hour"    in enc_df_arg.columns


def test_run_last_tick_utc_is_utc_timestamp():
    """result['last_tick_utc'] must be a UTC-aware datetime close to now."""
    from datetime import datetime, timezone

    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)

    before = datetime.now(timezone.utc)
    result = pipeline.run(KLGA, _INIT_TIME, _BUCKETS)
    after  = datetime.now(timezone.utc)

    ts = result["last_tick_utc"]
    assert ts.tzinfo is not None, "last_tick_utc must be timezone-aware"
    # Normalise to datetime for comparison
    ts_dt = ts if isinstance(ts, datetime) else ts.to_pydatetime()
    assert before <= ts_dt <= after, (
        f"last_tick_utc {ts_dt} outside [{before}, {after}]"
    )


def test_tft_max_encoder_length_is_72():
    from model.tft import MAX_ENCODER_LENGTH
    assert MAX_ENCODER_LENGTH == 72


def test_tft_prep_removed():
    """_prep() must no longer exist as a module-level function in model.tft."""
    import model.tft as tft_module
    assert not hasattr(tft_module, "_prep"), "_prep must be removed from model.tft"


# -----------------------------------------------------------------------
# McSampler.set_max_bias
# -----------------------------------------------------------------------

def test_mc_set_max_bias_updates_internal_bias():
    """set_max_bias() must update _max_bias used during sampling."""
    mc = McSampler(n_samp=200, rng_seed=0)
    assert mc._max_bias == 0.0
    mc.set_max_bias(5.0)
    assert mc._max_bias == 5.0


def test_mc_set_max_bias_shifts_daily_highs():
    """A positive max_bias lowers the returned daily high distribution."""
    rng = np.random.default_rng(7)
    mu    = np.full(48, 20.0)
    sigma = np.full(48, 1.0)

    mc_base = McSampler(n_samp=5000, rng_seed=7)
    mc_base.set_max_bias(0.0)
    highs_base = mc_base.sample_daily_high(mu, sigma)

    mc_biased = McSampler(n_samp=5000, rng_seed=7)
    mc_biased.set_max_bias(2.0)
    highs_biased = mc_biased.sample_daily_high(mu, sigma)

    # biased mean should be ~2 degF lower than base mean
    diff = highs_base.mean() - highs_biased.mean()
    assert abs(diff - 2.0) < 0.5, f"Expected ~2 F shift, got {diff:.3f}"


# -----------------------------------------------------------------------
# ModelPipeline -- registry wiring
# -----------------------------------------------------------------------

def test_pipeline_loads_max_bias_from_registry():
    """When registry is provided, mc.set_max_bias is called with registry value."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)

    registry = MagicMock()
    registry.load_max_bias.return_value = 3.7

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm, registry=registry)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    registry.load_max_bias.assert_called_once_with("KLGA")
    assert mc._max_bias == 3.7


def test_pipeline_without_registry_preserves_existing_max_bias():
    """When registry=None, mc._max_bias is not overwritten by the pipeline."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    mc.set_max_bias(1.5)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)  # no registry
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    assert mc._max_bias == 1.5  # unchanged

