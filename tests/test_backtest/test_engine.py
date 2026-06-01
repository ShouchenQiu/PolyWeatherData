"""Tests for backtest.engine and backtest.analysis.

All tests use in-memory data -- no file I/O.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    HALFSPREAD_DRAG,
    BacktestEngine,
    _bucket_outcome,
)
from backtest.analysis import BacktestAnalysis, _pnl_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(probs_by_bucket: dict):
    """Fake ModelPipeline that returns fixed probs."""
    pipeline = types.SimpleNamespace()
    pipeline.run = lambda station, init_utc, buckets, target_date=None: {"probs": probs_by_bucket}
    return pipeline


def _make_strategy():
    return types.SimpleNamespace()


def _make_station(timezone: str = "America/New_York"):
    """Fake StationConfig with a timezone attribute."""
    return types.SimpleNamespace(timezone=timezone)


def _clob_row(
    event_date: str,
    ts_utc: str,
    slug: str,
    lo: float,
    hi: float,
    mid: float,
    spread: float = 0.04,
    is_filled: int = 0,
) -> dict:
    return {
        "timestamp_utc":    pd.Timestamp(ts_utc, tz="UTC"),
        "event_end_date":   pd.Timestamp(event_date, tz="UTC"),
        "market_slug":      slug,
        "bucket_lo_f":      lo,
        "bucket_hi_f":      hi,
        "mid_price_yes":    mid,
        "estimated_spread": spread,
        "is_filled":        is_filled,
    }


def _make_clob_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _bucket_outcome
# ---------------------------------------------------------------------------

class TestBucketOutcome:
    def test_range_inside(self):
        assert _bucket_outcome(65.0, 60.0, 70.0) == 1.0

    def test_range_on_lo(self):
        assert _bucket_outcome(60.0, 60.0, 70.0) == 1.0

    def test_range_on_hi(self):
        assert _bucket_outcome(70.0, 60.0, 70.0) == 1.0

    def test_range_outside(self):
        assert _bucket_outcome(59.9, 60.0, 70.0) == 0.0
        assert _bucket_outcome(70.1, 60.0, 70.0) == 0.0

    def test_below_inside(self):
        assert _bucket_outcome(65.0, float("-inf"), 70.0) == 1.0
        assert _bucket_outcome(70.0, float("-inf"), 70.0) == 1.0

    def test_below_outside(self):
        assert _bucket_outcome(70.1, float("-inf"), 70.0) == 0.0

    def test_higher_inside(self):
        assert _bucket_outcome(70.0, 70.0, float("inf")) == 1.0
        assert _bucket_outcome(99.0, 70.0, float("inf")) == 1.0

    def test_higher_outside(self):
        assert _bucket_outcome(69.9, 70.0, float("inf")) == 0.0

    def test_nan_actual(self):
        assert math.isnan(_bucket_outcome(float("nan"), 60.0, 70.0))


# ---------------------------------------------------------------------------
# BacktestEngine -- basic PnL
# ---------------------------------------------------------------------------

class TestBacktestEnginePnL:
    """Fixed probs + fixed CLOB -> correct PnL columns."""

    def _build_engine(self, mc_mid=0.65, clob_mid=0.55, actual=70.0, outcome=1.0):
        # Single bucket: [65, 75]
        pipeline  = _make_pipeline({(65.0, 75.0): mc_mid})
        strategy  = _make_strategy()
        actual_highs = {"2024-01-02": actual}
        clob_rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug-65-75",
                      65.0, 75.0, clob_mid)
        ]
        clob_df = _make_clob_df(clob_rows)
        engine = BacktestEngine(pipeline, strategy, clob_df, actual_highs,
                                min_edge=0.0, cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01", _make_station(), init_hours=[6])
        return df

    def test_pnl_if_yes_correct(self):
        df = self._build_engine(mc_mid=0.65, clob_mid=0.55, actual=70.0)
        assert not df.empty
        row = df.iloc[0]
        # actual inside [65,75] -> outcome=1, pnl_if_yes = 1 - 0.55 = 0.45
        assert row["pnl_if_yes"] == pytest.approx(1.0 - 0.55, abs=1e-5)

    def test_pnl_if_no_correct(self):
        df = self._build_engine(mc_mid=0.65, clob_mid=0.55, actual=70.0)
        row = df.iloc[0]
        # pnl_if_no = 0.55 - 1 = -0.45
        assert row["pnl_if_no"] == pytest.approx(0.55 - 1.0, abs=1e-5)

    def test_edge_positive_when_mc_gt_clob(self):
        df = self._build_engine(mc_mid=0.70, clob_mid=0.55)
        assert df.iloc[0]["edge"] == pytest.approx(0.70 - 0.55, abs=1e-5)

    def test_edge_negative_when_mc_lt_clob(self):
        df = self._build_engine(mc_mid=0.40, clob_mid=0.55)
        assert df.iloc[0]["edge"] == pytest.approx(0.40 - 0.55, abs=1e-5)

    def test_halfspread_is_half_spread(self):
        df = self._build_engine()
        row = df.iloc[0]
        assert row["halfspread"] == pytest.approx(row["estimated_spread"] / 2.0, abs=1e-7)

    def test_coverage_type_full_when_24h(self):
        # init at 00 UTC on day-1 -> 24 EST hours in day-2
        df = self._build_engine()
        # 6 UTC + 48h covers day+2 more, but for date 2024-01-02 from 01 UTC
        # there should be 24 EST hours (the full day)
        row = df.iloc[0]
        assert row["coverage_type"] == "full"
        assert row["n_hours"] == 24


# ---------------------------------------------------------------------------
# BacktestEngine -- CLOB window selection
# ---------------------------------------------------------------------------

class TestClobWindowSelection:

    def test_prefers_is_filled_0_over_filled_1(self):
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            # is_filled=1 row with mid=0.80 (should be ignored)
            _clob_row("2024-01-02", "2024-01-01 06:10:00", "slugA",
                      65.0, 75.0, 0.80, is_filled=1),
            # is_filled=0 row with mid=0.55 (should be chosen)
            _clob_row("2024-01-02", "2024-01-01 06:05:00", "slugA",
                      65.0, 75.0, 0.55, is_filled=0),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert not df.empty
        assert df.iloc[0]["legacy_mid_price"] == pytest.approx(0.55, abs=1e-5)
        assert df.iloc[0]["clob_quality"] == "real"

    def test_falls_back_to_filled_when_no_real(self):
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:10:00", "slugA",
                      65.0, 75.0, 0.60, is_filled=1),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert not df.empty
        assert df.iloc[0]["clob_quality"] == "filled"

    def test_row_outside_window_excluded(self):
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        # 2024-01-01 09:00 is >30 min away from init 06:00
        rows = [
            _clob_row("2024-01-02", "2024-01-01 09:00:00", "slugA",
                      65.0, 75.0, 0.60, is_filled=0),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty


# ---------------------------------------------------------------------------
# BacktestEngine -- cutoff and coverage
# ---------------------------------------------------------------------------

class TestCutoffAndCoverage:

    def test_init_at_cutoff_excluded(self):
        """init_utc at 21:00 UTC = 16:00 EST on target day -> no records."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-01": 70.0}
        rows = [
            _clob_row("2024-01-01", "2024-01-01 21:00:00", "slugA",
                      65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                cold_start_hours=0)
        # init at 21:00 UTC on same target day -> cutoff hit
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[21])
        assert df.empty

    def test_partial_coverage_detected(self):
        """init late so only partial EST hours fall on target date."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.60})
        # Target 2024-01-02. Init at 18:00 UTC -> 13:00 EST.
        # EST hours on 2024-01-02: fxx=7..18 -> roughly 12 hours => partial
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 18:10:00", "slugA",
                      65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[18])
        if not df.empty:
            row = df[df["target_date"] == "2024-01-02"]
            if not row.empty:
                assert row.iloc[0]["coverage_type"] in ("full", "partial")


# ---------------------------------------------------------------------------
# BacktestEngine -- min_edge filter
# ---------------------------------------------------------------------------

class TestMinEdge:

    def test_min_edge_filters_small_edges(self):
        # edge = 0.70 - 0.68 = 0.02; min_edge=0.05 -> should be filtered
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slugA",
                      65.0, 75.0, 0.68),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                min_edge=0.05, cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty

    def test_min_edge_keeps_large_edges(self):
        pipeline     = _make_pipeline({(65.0, 75.0): 0.80})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slugA",
                      65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), actual_highs,
                                min_edge=0.05, cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert not df.empty


# ---------------------------------------------------------------------------
# BacktestEngine -- empty CLOB
# ---------------------------------------------------------------------------

class TestEmptyClob:

    def test_empty_clob_returns_empty_dataframe(self):
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df([]), actual_highs,
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty


# ---------------------------------------------------------------------------
# BacktestEngine -- no actual_highs
# ---------------------------------------------------------------------------

class TestMissingActualHighs:

    def test_skip_when_no_actual_for_target_date(self):
        pipeline = _make_pipeline({(65.0, 75.0): 0.70})
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slugA",
                      65.0, 75.0, 0.55),
        ]
        # actual_highs is empty -> no data for 2024-01-02
        engine = BacktestEngine(pipeline, _make_strategy(),
                                _make_clob_df(rows), {},
                                cold_start_hours=0)
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty


# ---------------------------------------------------------------------------
# BacktestEngine -- cold-start guard
# ---------------------------------------------------------------------------

class TestColdStartGuard:

    def _build_clob_multi_day(self) -> pd.DataFrame:
        """CLOB with rows on day+1 through day+4 for bucket [65, 75]."""
        rows = []
        for day_offset in range(1, 5):
            event_date = f"2024-01-0{day_offset + 1}"
            ts_utc     = f"2024-01-0{day_offset} 06:00:00"
            rows.append(_clob_row(event_date, ts_utc, "slug", 65.0, 75.0, 0.55))
        return _make_clob_df(rows)

    def test_inits_inside_cold_start_skipped(self):
        """Inits within cold_start_hours of start_date must not produce records."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {
            "2024-01-02": 70.0,
            "2024-01-03": 70.0,
            "2024-01-04": 70.0,
            "2024-01-05": 70.0,
        }
        clob_df = self._build_clob_multi_day()
        # cold_start_hours=72 means inits on day 0,1,2 (first 72h) skipped
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=72,
        )
        df = engine.run("2024-01-01", "2024-01-04",
                        _make_station(), init_hours=[6])

        if not df.empty:
            # All recorded init_utc must be >= start_date + 72h
            cutoff = pd.Timestamp("2024-01-04 00:00:00", tz="UTC")  # day1 + 72h
            for row_init in df["init_utc"].unique():
                ts = pd.Timestamp(row_init, tz="UTC")
                assert ts >= cutoff, (
                    f"init {ts} is before cold-start cutoff {cutoff}"
                )

    def test_zero_cold_start_hours_allows_all_inits(self):
        """With cold_start_hours=0 no inits are skipped."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug", 65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), actual_highs,
            cold_start_hours=0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert not df.empty

    def test_default_cold_start_is_72(self):
        """Default cold_start_hours must be 72 (matching MAX_ENCODER_LENGTH)."""
        engine = BacktestEngine(
            _make_pipeline({}), _make_strategy(), _make_clob_df([]), {},
        )
        assert engine._cold_start_hours == 72

    def test_process_init_call_site_matches_pipeline_signature(self):
        """pipeline.run() receives target_date kwarg (ADR-0010)."""
        received: list = []

        def capturing_run(station, init_utc, buckets, target_date=None):
            received.append((station, init_utc, buckets, target_date))
            return {"probs": {(65.0, 75.0): 0.70}}

        pipeline      = types.SimpleNamespace()
        pipeline.run  = capturing_run
        actual_highs  = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug", 65.0, 75.0, 0.55),
        ]
        station = _make_station()
        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), actual_highs,
            cold_start_hours=0,
        )
        engine.run("2024-01-01", "2024-01-01", station, init_hours=[6])

        assert len(received) == 1
        called_station, called_init, called_buckets, called_target = received[0]
        assert called_station is station
        assert called_init == pd.Timestamp("2024-01-01 06:00:00", tz="UTC")
        assert (65.0, 75.0) in called_buckets
        assert called_target == "2024-01-02"

    def test_backtest_skips_not_ready_inits_when_data_manager_provided(self):
        """pipeline.run() is not called for inits rejected by DataManager.inference_readiness.

        Also verifies that skip reasons are recorded in engine._readiness_skips and
        that only one inference_readiness() call is made per init (no double-read).
        """
        from unittest.mock import MagicMock
        from data.manager import ReadinessStatus

        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug", 65.0, 75.0, 0.55),
        ]
        pipeline = _make_pipeline({"probs": {(65.0, 75.0): 0.70}})
        data_manager = MagicMock()
        data_manager.inference_readiness.return_value = ReadinessStatus(
            ready=False, reason="cold_start", first_eligible_init=None
        )

        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), {"2024-01-02": 70.0},
            cold_start_hours=0,
            data_manager=data_manager,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        # All inits skipped; result is empty.
        assert df.empty
        # Exactly one inference_readiness call (not two).
        data_manager.inference_readiness.assert_called()
        assert data_manager.is_inference_ready.call_count == 0, (
            "is_inference_ready() must not be called; use inference_readiness() once"
        )
        # Skip is recorded in the engine's skip list.
        assert len(engine._readiness_skips) == 1
        assert engine._readiness_skips[0]["reason"] == "cold_start"

    def test_backtest_uses_cold_start_gate_when_data_manager_none(self):
        """Old cold_start_hours gate is active when data_manager is None."""
        pipeline = _make_pipeline({"probs": {(65.0, 75.0): 0.70}})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug", 65.0, 75.0, 0.55),
        ]
        # cold_start_hours=72 means init at 2024-01-01 06:00 < cutoff (Jan 1 00:00 + 72h = Jan 4)
        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), actual_highs,
            cold_start_hours=72,
            data_manager=None,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        # Init is within cold_start window; must be skipped.
        assert df.empty


# ---------------------------------------------------------------------------
# BacktestAnalysis
# ---------------------------------------------------------------------------

def _sample_results_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic results DataFrame."""
    rng  = np.random.default_rng(seed)
    mc   = rng.uniform(0.2, 0.8, n)
    clob = rng.uniform(0.25, 0.75, n)
    out  = rng.integers(0, 2, n).astype(float)
    edge = mc - clob
    hs   = rng.uniform(0.01, 0.03, n)
    leads = rng.integers(0, 48, n).astype(float)
    inits = ["2024-01-01 06:00" for _ in range(n)]
    return pd.DataFrame({
        "target_date":        ["2024-01-02"] * n,
        "init_utc":           inits,
        "init_local":         inits,
        "n_hours":            [24] * n,
        "coverage_type":      ["full"] * n,
        "lead_hours_to_4pm":  leads,
        "bucket_lo_f":        [60.0] * n,
        "bucket_hi_f":        [70.0] * n,
        "market_slug":        ["slug60-70"] * n,
        "mc_prob":            mc,
        "clob_price":         float("nan"),
        "bid_vwap":           float("nan"),
        "legacy_mid_price":   clob,
        "price_source":       ["legacy_mid"] * n,
        "estimated_spread":   hs * 2,
        "halfspread":         hs,
        "clob_quality":       ["real"] * n,
        "edge":               edge,
        "actual_max_f":       [65.0] * n,
        "actual_outcome":     out,
        "pnl_if_yes":         out - clob,
        "pnl_if_no":          clob - out,
    })


class TestBacktestAnalysisSummary:

    def test_summary_returns_dict(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        s   = ana.summary()
        assert isinstance(s, dict)

    def test_summary_has_brier_keys(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        s   = ana.summary()
        for k in ("brier_market", "brier_tft", "bss", "max_drawdown_mid"):
            assert k in s, f"Missing key: {k}"

    def test_brier_formula(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        s   = ana.summary()
        mkt_price = df["clob_price"].fillna(df["legacy_mid_price"])
        expected_mkt = float(
            ((mkt_price - df["actual_outcome"]) ** 2).mean()
        )
        assert s["brier_market"] == pytest.approx(expected_mkt, rel=1e-4)

    def test_bss_positive_when_tft_better(self):
        """Construct a df where mc_prob == actual_outcome (perfect), CLOB is random."""
        rng  = np.random.default_rng(0)
        n    = 50
        out  = rng.integers(0, 2, n).astype(float)
        clob = rng.uniform(0.2, 0.8, n)
        mc   = out  # TFT is perfect
        edge = mc - clob
        hs   = 0.01 * np.ones(n)
        df = pd.DataFrame({
            "target_date":       ["2024-01-02"] * n,
            "init_utc":          ["2024-01-01 06:00"] * n,
            "init_local":          ["2024-01-01 01:00"] * n,
            "n_hours":           [24] * n,
            "coverage_type":     ["full"] * n,
            "lead_hours_to_4pm": [12.0] * n,
            "bucket_lo_f":       [60.0] * n,
            "bucket_hi_f":       [70.0] * n,
            "market_slug":       ["slug"] * n,
            "mc_prob":           mc,
            "clob_price":        float("nan"),
            "bid_vwap":          float("nan"),
            "legacy_mid_price":  clob,
            "price_source":      "legacy_mid",
            "estimated_spread":  hs * 2,
            "halfspread":        hs,
            "clob_quality":      ["real"] * n,
            "edge":              edge,
            "actual_max_f":      [65.0] * n,
            "actual_outcome":    out,
            "pnl_if_yes":        out - clob,
            "pnl_if_no":         clob - out,
        })
        ana = BacktestAnalysis(df)
        s   = ana.summary()
        assert s["bss"] > 0.0


class TestBacktestAnalysisByThreshold:

    def test_returns_dataframe(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        r   = ana.by_threshold([0.05, 0.10])
        assert isinstance(r, pd.DataFrame)

    def test_threshold_column_present(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        r   = ana.by_threshold([0.05])
        if not r.empty:
            assert "threshold" in r.columns

    def test_n_total_matches_counts(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        thr = 0.05
        r   = ana.by_threshold([thr])
        if not r.empty:
            row   = r.iloc[0]
            exp_y = int((df["edge"] >  thr).sum())
            exp_n = int((df["edge"] < -thr).sum())
            assert row["N_yes"]   == exp_y
            assert row["N_no"]    == exp_n
            assert row["N_total"] == exp_y + exp_n

    def test_roi_mid_formula(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        thr = 0.05
        r   = ana.by_threshold([thr])
        if not r.empty:
            row      = r.iloc[0]
            yes      = df[df["edge"] >  thr]
            no       = df[df["edge"] < -thr]
            pnl_y    = yes["pnl_if_yes"].sum()
            pnl_n    = no["pnl_if_no"].sum()
            n_tot    = len(yes) + len(no)
            expected = (pnl_y + pnl_n) / n_tot if n_tot > 0 else float("nan")
            assert row["ROI_mid"] == pytest.approx(expected, rel=1e-4)

    def test_empty_df_returns_empty(self):
        empty_df  = _sample_results_df().head(0)
        ana       = BacktestAnalysis(empty_df)
        r         = ana.by_threshold([0.05])
        assert r.empty


class TestBacktestAnalysisByLeadTime:

    def test_returns_dataframe(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        r   = ana.by_lead_time()
        assert isinstance(r, pd.DataFrame)


class TestBacktestAnalysisByInitHour:

    def test_returns_dataframe(self):
        df  = _sample_results_df()
        ana = BacktestAnalysis(df)
        r   = ana.by_init_hour()
        assert isinstance(r, pd.DataFrame)


# ---------------------------------------------------------------------------
# V2V10 -- L2 quote path and legacy mid fallback
# ---------------------------------------------------------------------------

class TestV2V10L2QuotePath:
    """When CLOB rows carry L2 bids/asks, compute_quote() drives pricing."""

    def test_l2_clob_uses_compute_quote_and_sets_clob_price(self):
        """L2 bids/asks in CLOB rows produce clob_price via compute_quote VWAP."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        # One book level at 0.55 with 1000 shares = plenty of depth
        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-l2",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,   # ignored for L2
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 1000.0}],
                "asks":             [{"price": 0.55, "size": 1000.0}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        # BUY at 0.55 with plenty of depth -> VWAP ~= 0.55
        assert row["clob_price"] == pytest.approx(0.55, abs=1e-5)
        assert row["price_source"] == "l2_vwap"
        assert math.isnan(row["legacy_mid_price"])
        # Edge uses L2 VWAP: 0.70 - 0.55 = 0.15
        assert row["edge"] == pytest.approx(0.15, abs=1e-5)
        # PnL uses L2 VWAP
        assert row["pnl_if_yes"] == pytest.approx(1.0 - 0.55, abs=1e-5)

    def test_l2_clob_incomplete_fill_drops_row(self):
        """When NEITHER BUY nor SELL has sufficient depth, row is dropped."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        # Tiny depth: only 0.1 share ~= $0.05 far below $20 quote_size_usd
        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-l2",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 0.1}],
                "asks":             [{"price": 0.55, "size": 0.1}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty

    def test_l2_clob_ask_complete_bid_incomplete_keeps_row(self):
        """ASK side has enough depth for BUY VWAP but BID side does not;
        row is still kept (clob_price set, bid_vwap NaN)."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-l2",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 0.1}],   # insufficient
                "asks":             [{"price": 0.55, "size": 1000.0}],  # sufficient
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert row["clob_price"] == pytest.approx(0.55, abs=1e-5)
        assert math.isnan(row["bid_vwap"])
        assert row["price_source"] == "l2_vwap"


class TestV2V10LegacyMidFallback:
    """When CLOB rows have only mid_price_yes, it is stored as legacy_mid_price."""

    def test_legacy_mid_sets_legacy_column_not_clob_price(self):
        """Legacy mid_price_yes -> legacy_mid_price column; clob_price is NaN."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug-legacy",
                      65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), actual_highs,
            cold_start_hours=0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert row["price_source"] == "legacy_mid"
        assert row["legacy_mid_price"] == pytest.approx(0.55, abs=1e-5)
        assert math.isnan(row["clob_price"])
        # Edge uses legacy mid: 0.70 - 0.55 = 0.15
        assert row["edge"] == pytest.approx(0.15, abs=1e-5)
        # PnL uses legacy mid
        assert row["pnl_if_yes"] == pytest.approx(1.0 - 0.55, abs=1e-5)

    def test_legacy_mid_not_passed_through_compute_quote(self):
        """Verify legacy_mid_price never goes through compute_quote path."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        # mid_price_yes=0.99 but with L2 asks at 0.01 -- if compute_quote
        # were incorrectly called it would give VWAP ~= 0.01 instead of 0.99.
        # Since there are NO L2 columns, the engine must use legacy path.
        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-mixed",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.99,
                "estimated_spread": 0.04,
                "is_filled":        0,
                # No bids/asks columns -- force legacy path
            },
        ]
        # NOTE: deliberately omit bids/asks so the engine sees legacy
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert row["price_source"] == "legacy_mid"
        # Edge = 0.70 - 0.99 = -0.29 (mid is 0.99, NOT L2 VWAP of 0.01)
        assert row["edge"] == pytest.approx(0.70 - 0.99, abs=1e-5)

    def test_readiness_skip_regression(self):
        """V2V04 readiness skip is not broken by price-source changes."""
        from unittest.mock import MagicMock
        from data.manager import ReadinessStatus

        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug", 65.0, 75.0, 0.55),
        ]
        pipeline = _make_pipeline({"probs": {(65.0, 75.0): 0.70}})
        data_manager = MagicMock()
        data_manager.inference_readiness.return_value = ReadinessStatus(
            ready=False, reason="cold_start", first_eligible_init=None
        )

        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), {"2024-01-02": 70.0},
            cold_start_hours=0,
            data_manager=data_manager,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        assert df.empty
        assert len(engine._readiness_skips) == 1


# ---------------------------------------------------------------------------
# _pnl_summary unit tests
# ---------------------------------------------------------------------------

class TestPnlSummary:

    def _df(self, edges, pnl_yes, pnl_no, hs=None):
        n = len(edges)
        if hs is None:
            hs = [0.01] * n
        return pd.DataFrame({
            "edge":       edges,
            "pnl_if_yes": pnl_yes,
            "pnl_if_no":  pnl_no,
            "halfspread": hs,
        })

    def test_none_when_no_signals(self):
        df = self._df([0.01, -0.01], [0.1, -0.1], [-0.1, 0.1])
        r  = _pnl_summary(df, thr=0.05)
        assert r is None

    def test_basic_counts(self):
        df = self._df(
            edges    = [0.10,  0.08, -0.10, -0.08,  0.01],
            pnl_yes  = [0.30,  0.20,  0.10,  0.10,  0.50],
            pnl_no   = [-0.30, -0.20, 0.40,  0.40, -0.50],
        )
        r = _pnl_summary(df, thr=0.05)
        assert r is not None
        assert r["N_yes"]   == 2
        assert r["N_no"]    == 2
        assert r["N_total"] == 4

    def test_roi_mid_correct(self):
        df = self._df(
            edges    = [0.10, -0.10],
            pnl_yes  = [0.40,  0.00],
            pnl_no   = [0.00,  0.40],
        )
        r = _pnl_summary(df, thr=0.05)
        # expected: (0.40 + 0.40) / 2 = 0.40
        assert r["ROI_mid"] == pytest.approx(0.40, abs=1e-5)

# ---------------------------------------------------------------------------
# HALFSPREAD_DRAG constant
# ---------------------------------------------------------------------------

def test_halfspread_drag_value():
    assert HALFSPREAD_DRAG == pytest.approx(-0.0127, rel=1e-6)


# ---------------------------------------------------------------------------
# V2V10 -- mixed L2/legacy routing (per-row, not window-level)
# ---------------------------------------------------------------------------

class TestV2V10MixedL2Legacy:

    def test_l2_and_legacy_rows_same_bucket_l2_priority(self):
        """When a bucket has both L2 and legacy rows in the same window,
        L2 aggregate takes priority and the legacy row does not overwrite."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        rows = [
            # Legacy row: mid=0.99, no L2 data
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-mix",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.99,
                "estimated_spread": 0.04,
                "is_filled":        0,
            },
            # L2 row: ask=0.55, same bucket
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:05:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-mix",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 1000.0}],
                "asks":             [{"price": 0.55, "size": 1000.0}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        # L2 takes priority: clob_price ~= 0.55 (ask VWAP), not 0.99
        row = df.iloc[0]
        assert row["price_source"] == "l2_vwap"
        assert row["clob_price"] == pytest.approx(0.55, abs=1e-5)
        assert math.isnan(row["legacy_mid_price"])
        # Edge = 0.70 - 0.55 = 0.15
        assert row["edge"] == pytest.approx(0.15, abs=1e-5)

    def test_legacy_only_row_survives_when_l2_columns_exist_but_empty(self):
        """A row with bids/asks columns present but empty per-row falls
        back to legacy_mid_price (not silently dropped)."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-legacy-only",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.55,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [],   # empty list = no L2 data
                "asks":             [],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert row["price_source"] == "legacy_mid"
        assert row["legacy_mid_price"] == pytest.approx(0.55, abs=1e-5)
        assert math.isnan(row["clob_price"])


# ---------------------------------------------------------------------------
# V2V10 -- bid_vwap column and side-aware pnl_if_no
# ---------------------------------------------------------------------------

class TestV2V10BidVwap:

    def test_l2_row_produces_bid_vwap_column(self):
        """L2 rows produce a bid_vwap column from SELL-side compute_quote."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-l2",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 1000.0}],
                "asks":             [{"price": 0.55, "size": 1000.0}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert "bid_vwap" in df.columns
        # BUY asks at 0.55, SELL bids at 0.53
        assert row["clob_price"] == pytest.approx(0.55, abs=1e-5)
        assert row["bid_vwap"]   == pytest.approx(0.53, abs=1e-5)

    def test_pnl_if_no_uses_bid_vwap_for_l2_rows(self):
        """For L2 rows pnl_if_no = bid_vwap - outcome, not ask_vwap - outcome."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}   # outcome = 1.0 (inside bucket)
        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-l2",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 1000.0}],
                "asks":             [{"price": 0.55, "size": 1000.0}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        # pnl_if_yes = outcome - ask_vwap = 1.0 - 0.55 = 0.45
        assert row["pnl_if_yes"] == pytest.approx(1.0 - 0.55, abs=1e-5)
        # pnl_if_no  = bid_vwap - outcome = 0.53 - 1.0 = -0.47
        assert row["pnl_if_no"] == pytest.approx(0.53 - 1.0, abs=1e-5)

    def test_legacy_row_pnl_if_no_unchanged(self):
        """Legacy rows still use mid_price for both pnl_if_yes and pnl_if_no."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}
        rows = [
            _clob_row("2024-01-02", "2024-01-01 06:00:00", "slug-legacy",
                      65.0, 75.0, 0.55),
        ]
        engine = BacktestEngine(
            pipeline, _make_strategy(), _make_clob_df(rows), actual_highs,
            cold_start_hours=0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])

        assert not df.empty
        row = df.iloc[0]
        assert row["price_source"] == "legacy_mid"
        assert math.isnan(row["bid_vwap"])
        assert row["pnl_if_yes"] == pytest.approx(1.0 - 0.55, abs=1e-5)
        assert row["pnl_if_no"]  == pytest.approx(0.55 - 1.0, abs=1e-5)

    def test_l2_incomplete_fill_does_not_fall_back_to_legacy(self):
        """Row with L2 data that fails depth check is dropped; does not
        silently use mid_price_yes as a fallback."""
        pipeline     = _make_pipeline({(65.0, 75.0): 0.70})
        actual_highs = {"2024-01-02": 70.0}

        rows = [
            {
                "timestamp_utc":    pd.Timestamp("2024-01-01 06:00:00", tz="UTC"),
                "event_end_date":   pd.Timestamp("2024-01-02", tz="UTC"),
                "market_slug":      "slug-shallow",
                "bucket_lo_f":      65.0,
                "bucket_hi_f":      75.0,
                "mid_price_yes":    0.50,
                "estimated_spread": 0.04,
                "is_filled":        0,
                "bids":             [{"price": 0.53, "size": 0.1}],
                "asks":             [{"price": 0.55, "size": 0.1}],
            },
        ]
        clob_df = _make_clob_df(rows)
        engine = BacktestEngine(
            pipeline, _make_strategy(), clob_df, actual_highs,
            cold_start_hours=0, quote_size_usd=20.0,
        )
        df = engine.run("2024-01-01", "2024-01-01",
                        _make_station(), init_hours=[6])
        # L2 depth insufficient, mid_price_yes=0.50 must not be used
        assert df.empty
