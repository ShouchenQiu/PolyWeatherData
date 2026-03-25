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


def _make_iem_df(start: pd.Timestamp) -> pd.DataFrame:
    """24 hourly obs rows."""
    rows = [
        {
            "valid_time":      start + pd.Timedelta(hours=h),
            "raw_tmpc_tenths": 5.0 + h * 0.1,
        }
        for h in range(24)
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


def _make_data_mgr(init_time: pd.Timestamp) -> MagicMock:
    """DataManager stub returning minimal HRRR and IEM DataFrames."""
    dm          = MagicMock()
    dm.fetch_hrrr.return_value = _make_hrrr_df(init_time)
    enc_start   = init_time - pd.Timedelta(hours=23)
    enc_end     = init_time
    dm.fetch_iem.return_value  = _make_iem_df(enc_start)
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
    """Both fetch_hrrr and fetch_iem must be called."""
    mc       = McSampler(n_samp=200, rng_seed=0)
    tft      = _make_tft_mock()
    dm       = _make_data_mgr(_INIT_TIME)
    pipeline = ModelPipeline(tft=tft, mc=mc, data_mgr=dm)
    pipeline.run(KLGA, _INIT_TIME, _BUCKETS)

    dm.fetch_hrrr.assert_called_once_with(_INIT_TIME)
    dm.fetch_iem.assert_called_once()


# -----------------------------------------------------------------------
# _assemble_frames -- unit tests
# -----------------------------------------------------------------------

def test_assemble_frames_encoder_rows():
    hrrr_df  = _make_hrrr_df(_INIT_TIME)
    iem_df   = _make_iem_df(_INIT_TIME - pd.Timedelta(hours=23))
    enc, dec = ModelPipeline._assemble_frames(hrrr_df, iem_df, _INIT_TIME)

    assert len(enc) == 24
    assert enc["fxx"].min() == -23
    assert enc["fxx"].max() == 0


def test_assemble_frames_decoder_rows():
    hrrr_df  = _make_hrrr_df(_INIT_TIME)
    iem_df   = pd.DataFrame(columns=["valid_time"])
    enc, dec = ModelPipeline._assemble_frames(hrrr_df, iem_df, _INIT_TIME)

    assert len(dec) == 48
    assert dec["fxx"].min() == 1
    assert dec["fxx"].max() == 48


def test_assemble_frames_empty_iem():
    """Empty IEM DataFrame should not crash."""
    hrrr_df  = _make_hrrr_df(_INIT_TIME)
    iem_df   = pd.DataFrame(columns=["valid_time"])
    enc, dec = ModelPipeline._assemble_frames(hrrr_df, iem_df, _INIT_TIME)

    assert len(enc) == 24
