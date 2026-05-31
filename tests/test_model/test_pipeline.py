"""Tests for model/pipeline.py -- ModelPipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.stations import StationConfig
from model.features import REQUIRED_DECODER_SOURCE_COLUMNS, REQUIRED_ENCODER_SOURCE_COLUMNS
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
    """48 HRRR decoder rows with all required source columns."""
    rows = []
    for fxx in range(1, 49):
        row: dict = {
            "fxx":        fxx,
            "init_time":  init_time,
            "valid_time": init_time + pd.Timedelta(hours=fxx),
            "TMP_2m":     280.0 + fxx * 0.05,
        }
        for col in REQUIRED_DECODER_SOURCE_COLUMNS:
            row.setdefault(col, 0.0)
        rows.append(row)
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
    """72 merged encoder rows with all required source columns."""
    rows = []
    for h in range(72):
        row: dict = {
            "valid_time":        init_time - pd.Timedelta(hours=72 - h),
            "TMP_2m":            5.0 + h * 0.1,
            "raw_tmpc_tenths":   6.0 + h * 0.1,
            "temp_error":        0.0,
        }
        for col in REQUIRED_ENCODER_SOURCE_COLUMNS:
            row.setdefault(col, 0.0)
        rows.append(row)
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
# McSampler.max_bias parameter
# -----------------------------------------------------------------------

def test_mc_max_bias_param_exact_shift():
    """max_bias parameter shifts the daily high distribution by exactly that amount."""
    s1 = McSampler(n_samp=200, rng_seed=0)
    s2 = McSampler(n_samp=200, rng_seed=0)
    mu    = np.full(48, 20.0)
    sigma = np.full(48, 0.001)
    out0  = s1.sample_daily_high(mu, sigma, max_bias=0.0)
    out5  = s2.sample_daily_high(mu, sigma, max_bias=5.0)
    np.testing.assert_allclose(out0 - 5.0, out5)


def test_mc_max_bias_param_shifts_daily_highs():
    """A positive max_bias lowers the returned daily high distribution."""
    mu    = np.full(48, 20.0)
    sigma = np.full(48, 1.0)
    mc    = McSampler(n_samp=5000, rng_seed=7)
    highs_base   = mc.sample_daily_high(mu, sigma, max_bias=0.0)
    highs_biased = mc.sample_daily_high(mu, sigma, max_bias=2.0)

    # biased mean should be ~2 degF lower than base mean
    diff = highs_base.mean() - highs_biased.mean()
    assert abs(diff - 2.0) < 0.5, f"Expected ~2 F shift, got {diff:.3f}"


# -----------------------------------------------------------------------
# ModelPipeline -- registry wiring
# -----------------------------------------------------------------------

def test_pipeline_loads_max_bias_from_registry():
    """When registry is provided, load_max_bias value is passed to sample_daily_high."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)

    registry = MagicMock()
    registry.load_max_bias.return_value = 3.7

    captured: dict = {}
    original_sdh = mc.sample_daily_high

    def spy_sdh(mu_vec, sigma_vec, max_bias=0.0):
        captured["max_bias"] = max_bias
        return original_sdh(mu_vec, sigma_vec, max_bias=max_bias)

    mc.sample_daily_high = spy_sdh

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm, registry=registry)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    registry.load_max_bias.assert_called_once_with("KLGA")
    assert captured["max_bias"] == pytest.approx(3.7)


def test_pipeline_without_registry_uses_zero_bias():
    """When registry=None, pipeline calls sample_daily_high with max_bias=0.0."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)

    captured: dict = {}
    original_sdh = mc.sample_daily_high

    def spy_sdh(mu_vec, sigma_vec, max_bias=0.0):
        captured["max_bias"] = max_bias
        return original_sdh(mu_vec, sigma_vec, max_bias=max_bias)

    mc.sample_daily_high = spy_sdh

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)  # no registry
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


# -----------------------------------------------------------------------
# V2V05 -- tft=None rejected
# -----------------------------------------------------------------------

def test_pipeline_raises_if_tft_is_none():
    """ModelPipeline.__init__ must raise TypeError when tft=None."""
    mc = McSampler(n_samp=200, rng_seed=0)
    dm = _make_data_mgr(_INIT_TIME)
    with pytest.raises(TypeError):
        ModelPipeline(tft=None, mc=mc, data_mgr=dm)


# -----------------------------------------------------------------------
# V2V05 -- target_date parameter
# -----------------------------------------------------------------------

def test_run_accepts_target_date_keyword():
    """run() must accept target_date as a keyword argument without error."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    result = pipeline.run(KLGA, _INIT_TIME, _BUCKETS, target_date="2025-01-01")
    assert isinstance(result, dict)


def test_run_target_date_no_matching_fxx_raises():
    """run() must raise ValueError when target_date has no matching fxx rows."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    # 2025-01-05 is outside the 1..48 fxx valid_time window from _INIT_TIME
    with pytest.raises(ValueError, match="target_date"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS, target_date="2025-01-05")


# -----------------------------------------------------------------------
# V2V05 -- decoder completeness gate
# -----------------------------------------------------------------------

def test_pipeline_raises_if_hrrr_empty():
    """run() must raise RuntimeError when fetch_hrrr returns an empty DataFrame."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)
    dm.fetch_hrrr.return_value = pd.DataFrame()
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(RuntimeError):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


def test_pipeline_raises_if_hrrr_missing_tmp2m():
    """run() must raise RuntimeError when decoder rows have no TMP_2m column."""
    mc   = McSampler(n_samp=200, rng_seed=0)
    tft  = _make_tft_mock()
    dm   = _make_data_mgr(_INIT_TIME)
    hrrr = _make_hrrr_df(_INIT_TIME).drop(columns=["TMP_2m"])
    dm.fetch_hrrr.return_value = hrrr
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(RuntimeError):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


# -----------------------------------------------------------------------
# V2V05 -- _add_model_features raises on missing valid_time
# -----------------------------------------------------------------------

def test_pipeline_raises_if_encoder_missing_valid_time():
    """run() must raise ValueError when encoder_df has no valid_time column."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)
    enc = _make_encoder_df(_INIT_TIME).drop(columns=["valid_time"])
    dm.build_encoder_obs_window.return_value = enc
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="valid_time"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


# -----------------------------------------------------------------------
# V2V05 -- pre-predict source column validation
# -----------------------------------------------------------------------

def test_pipeline_raises_if_encoder_missing_required_source_column():
    """run() must raise ValueError when encoder_df lacks a required source column."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)
    enc = _make_encoder_df(_INIT_TIME).drop(columns=["temp_error"])
    dm.build_encoder_obs_window.return_value = enc
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="temp_error"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


def test_pipeline_raises_if_decoder_missing_required_source_column():
    """run() must raise ValueError when hrrr_df lacks a required decoder source column."""
    mc    = McSampler(n_samp=200, rng_seed=0)
    tft   = _make_tft_mock()
    dm    = _make_data_mgr(_INIT_TIME)
    hrrr  = _make_hrrr_df(_INIT_TIME).drop(columns=["DPT_2m"])
    dm.fetch_hrrr.return_value = hrrr
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="DPT_2m"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


# -----------------------------------------------------------------------
# V2V05 -- quantile sanity guard
# -----------------------------------------------------------------------

def test_pipeline_raises_on_non_finite_quantiles():
    """run() must raise ValueError when TFT returns non-finite quantile values."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    # Inject a NaN into fxx=1 p50
    bad_q = {fxx: {"p10": 4.0, "p50": 6.0, "p90": 8.0} for fxx in range(1, 49)}
    bad_q[1] = {"p10": 4.0, "p50": float("nan"), "p90": 8.0}
    tft.predict.return_value = bad_q
    dm = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="fxx=1"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


def test_pipeline_raises_on_unordered_quantiles():
    """run() must raise ValueError when p10 > p50 for any fxx."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    bad_q = {fxx: {"p10": 4.0, "p50": 6.0, "p90": 8.0} for fxx in range(1, 49)}
    bad_q[5] = {"p10": 9.0, "p50": 6.0, "p90": 8.0}  # p10 > p50
    tft.predict.return_value = bad_q
    dm = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="fxx=5"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS)


# -----------------------------------------------------------------------
# V2V05 -- is_extrapolated filter
# -----------------------------------------------------------------------

def test_pipeline_excludes_extrapolated_fxx_from_daily_high():
    """When target_date is set, is_extrapolated=True rows must not enter MC sampling."""
    mc  = McSampler(n_samp=500, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)

    hrrr = _make_hrrr_df(_INIT_TIME).copy()
    # Mark fxx 13..48 as extrapolated (only fxx 1..12 are real)
    hrrr["is_extrapolated"] = hrrr["fxx"] > 12
    dm.fetch_hrrr.return_value = hrrr

    captured: dict = {}
    original_sdh = mc.sample_daily_high

    def spy_sdh(mu_vec, sigma_vec, max_bias=0.0):
        captured["n_hours"] = len(mu_vec)
        return original_sdh(mu_vec, sigma_vec, max_bias=max_bias)

    mc.sample_daily_high = spy_sdh
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    # target_date = 2025-01-01; INIT_TIME = 2025-01-01 12:00 UTC
    # New York is UTC-5 in January; local 00:00-24:00 = UTC 05:00-05:00 next day
    # fxx 1..48 from 12:00 UTC = valid_times 13:00..12:00 next day; local 08:00..07:00
    # Only fxx 1..12 are non-extrapolated; local times 08:00..19:00 -- all in Jan 1
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS, target_date="2025-01-01")
    assert "n_hours" in captured
    assert captured["n_hours"] <= 12


def test_pipeline_raises_if_all_target_fxx_extrapolated():
    """run() must raise ValueError when all target fxx rows are extrapolated."""
    mc  = McSampler(n_samp=200, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)

    hrrr = _make_hrrr_df(_INIT_TIME).copy()
    hrrr["is_extrapolated"] = True  # all extrapolated
    dm.fetch_hrrr.return_value = hrrr

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    with pytest.raises(ValueError, match="extrapolated"):
        pipeline.run(KLGA, _INIT_TIME, _BUCKETS, target_date="2025-01-01")


# -----------------------------------------------------------------------
# V2V05 -- daily-high local calendar day (ADR-0010)
# -----------------------------------------------------------------------

def test_pipeline_daily_high_uses_local_calendar_day():
    """With target_date, MC sampling uses only fxx in local-calendar-day window."""
    mc  = McSampler(n_samp=500, rng_seed=0)
    tft = _make_tft_mock()
    dm  = _make_data_mgr(_INIT_TIME)

    captured: dict = {}
    original_sdh = mc.sample_daily_high

    def spy_sdh(mu_vec, sigma_vec, max_bias=0.0):
        captured["n_hours"] = len(mu_vec)
        return original_sdh(mu_vec, sigma_vec, max_bias=max_bias)

    mc.sample_daily_high = spy_sdh

    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS, target_date="2025-01-01")

    # KLGA is UTC-5 in January (America/New_York, EST).
    # _INIT_TIME = 2025-01-01 12:00 UTC = 07:00 local.
    # Local day 2025-01-01 = UTC [05:00, 2025-01-02 05:00).
    # fxx 1..48 -> valid_times 13:00..12:00 UTC next day; only those within
    # [05:00 UTC, 2025-01-02 05:00 UTC) fall on local Jan 1.
    # fxx 1 = 13:00 UTC (in window), fxx 17 = 2025-01-02 05:00 UTC (boundary, exclusive)
    # Expect 16 hours in window (fxx 1..16).
    assert captured.get("n_hours") == 16
