"""Tests for HrrrFill -- valid_time splice and tail-gap fallback."""
from __future__ import annotations

import pandas as pd
import pytest

from data.forecast.hrrr_fill import HrrrFill, nearest_prior_major


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_df(init_time_str: str, fxx_range: range, col_val: float = 1.0) -> pd.DataFrame:
    """Build a minimal HRRR-like DataFrame for testing."""
    init = pd.Timestamp(init_time_str, tz="UTC")
    rows = []
    for fxx in fxx_range:
        rows.append({
            "init_time": str(init),
            "fxx": fxx,
            "valid_time": init + pd.Timedelta(hours=fxx),
            "TMP_2m": col_val + fxx * 0.01,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit: _nearest_prior_major helper
# ---------------------------------------------------------------------------

def test_nearest_prior_major_on_major_hour():
    t = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    assert nearest_prior_major(t) == t


def test_nearest_prior_major_non_major():
    t = pd.Timestamp("2026-01-01 03:00", tz="UTC")
    result = nearest_prior_major(t)
    assert result == pd.Timestamp("2026-01-01 00:00", tz="UTC")


def test_nearest_prior_major_21z():
    t = pd.Timestamp("2026-01-01 21:00", tz="UTC")
    result = nearest_prior_major(t)
    assert result == pd.Timestamp("2026-01-01 18:00", tz="UTC")


# ---------------------------------------------------------------------------
# HrrrFill: splice alignment
# ---------------------------------------------------------------------------

def test_fill_returns_48_fxx_rows():
    """Output always has exactly 49 rows (fxx 0..48)."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))   # 03Z, fxx 0-18
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))   # 00Z, fxx 0-48
    result = fill.fill(non_major, nearest)
    assert len(result) == 49
    assert list(result["fxx"]) == list(range(49))


def test_fill_prefers_non_major_for_overlap():
    """valid_times in both sources: non_major data is used."""
    fill = HrrrFill()
    # non_major has TMP_2m = 100.0; nearest_major has TMP_2m = 1.0
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=100.0)
    nearest   = _make_df("2026-01-01 00:00", range(0, 49), col_val=1.0)
    result = fill.fill(non_major, nearest)
    # fxx 0 corresponds to valid_time 03Z which exists in both
    fxx0_row = result[result["fxx"] == 0].iloc[0]
    assert fxx0_row["TMP_2m"] == pytest.approx(100.0, abs=0.05)


def test_fill_uses_nearest_major_for_tail():
    """Tail valid_times (fxx 19-48 of non_major) come from nearest_major."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=100.0)
    nearest   = _make_df("2026-01-01 00:00", range(0, 49), col_val=2.0)
    result = fill.fill(non_major, nearest)
    # fxx 25 of non_major corresponds to valid_time 03Z+25h = 04Z next day
    # nearest_major 00Z+28h = same valid_time -> should be col_val 2.0
    fxx25_tmp = result[result["fxx"] == 25].iloc[0]["TMP_2m"]
    assert fxx25_tmp == pytest.approx(2.0 + 28 * 0.01, abs=0.05)


def test_fill_init_time_in_output_matches_non_major():
    """Output init_time column reflects the non-major init, not the major."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    assert "2026-01-01 03:00:00" in str(result["init_time"].iloc[0])


def test_fill_valid_time_continuous():
    """Output valid_times are contiguous hourly from non_major init to +48h."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    init = pd.Timestamp("2026-01-01 03:00", tz="UTC")
    expected_vts = [init + pd.Timedelta(hours=f) for f in range(49)]
    output_vts = pd.to_datetime(result["valid_time"], utc=True).tolist()
    assert output_vts == expected_vts


# ---------------------------------------------------------------------------
# HrrrFill: tail-gap fallback
# ---------------------------------------------------------------------------

def test_fill_uses_prior_major_for_uncovered_tail(caplog):
    """When nearest_major does not cover a tail valid_time, prior_major is used."""
    fill = HrrrFill()
    init = pd.Timestamp("2026-01-01 03:00", tz="UTC")
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=100.0)
    # nearest major only covers fxx 0-43 (missing fxx 44-48 from 00Z POV)
    nearest   = _make_df("2026-01-01 00:00", range(0, 44), col_val=2.0)
    # prior major (2025-12-31 18Z) covers all needed valid_times
    prior     = _make_df("2025-12-31 18:00", range(0, 55), col_val=3.0)
    result = fill.fill(non_major, nearest, prior)
    # fxx 48 of 03Z init = valid_time 03Z+48h = 2026-01-03 03Z
    # Not in nearest (which only goes to 2026-01-02 19Z from 00Z+43h)
    # Should come from prior_major
    assert len(result) == 49


def test_fill_tail_gap_logs_warning():
    """WARNING is logged when prior_major is used for tail gap."""
    from loguru import logger
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        fill = HrrrFill()
        non_major = _make_df("2026-01-01 03:00", range(0, 19))
        nearest   = _make_df("2026-01-01 00:00", range(0, 10))   # very short -> gaps guaranteed
        prior     = _make_df("2025-12-31 18:00", range(0, 55), col_val=3.0)
        fill.fill(non_major, nearest, prior)
    finally:
        logger.remove(sink_id)
    assert any("tail gap" in m.lower() for m in messages)


def test_fill_no_warning_when_nearest_major_covers_all():
    """No WARNING when nearest_major has all required valid_times."""
    from loguru import logger
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        fill = HrrrFill()
        non_major = _make_df("2026-01-01 03:00", range(0, 19))
        nearest   = _make_df("2026-01-01 00:00", range(0, 49))
        fill.fill(non_major, nearest)
    finally:
        logger.remove(sink_id)
    assert not any("tail gap" in m.lower() for m in messages)


def test_fill_empty_non_major_returns_empty():
    """Empty non_major_df short-circuits and returns empty."""
    fill = HrrrFill()
    result = fill.fill(pd.DataFrame(), _make_df("2026-01-01 00:00", range(0, 49)))
    assert result.empty


def test_fill_prior_major_data_appears_in_tail():
    """Values for tail valid_times genuinely come from prior_major (not NaN)."""
    fill = HrrrFill()
    # nearest only covers 00Z fxx 0-3; prior covers much more with col_val=99.0
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=1.0)
    nearest   = _make_df("2026-01-01 00:00", range(0, 4),  col_val=2.0)
    prior     = _make_df("2025-12-31 18:00", range(0, 55), col_val=99.0)
    result = fill.fill(non_major, nearest, prior)
    # fxx 19 of 03Z init = valid_time 2026-01-01 22Z
    # Not in non_major (ends at fxx 18 = 21Z), not in nearest (ends at 03Z).
    # prior (18Z + 28h = 22Z) should fill it with 99.0 + 28*0.01
    fxx19_row = result[result["fxx"] == 19].iloc[0]
    assert fxx19_row["TMP_2m"] == pytest.approx(99.0 + 28 * 0.01, abs=0.05)


# ---------------------------------------------------------------------------
# V2V02: is_extrapolated column
# ---------------------------------------------------------------------------

def test_fill_output_has_is_extrapolated_column():
    """Output always contains an is_extrapolated boolean column."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    assert "is_extrapolated" in result.columns


def test_fill_is_extrapolated_false_for_covered_rows():
    """Rows filled from nearest_major or non_major get is_extrapolated=False."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    # nearest covers all 49 rows for 03Z init (00Z+45h = 2026-01-03 00Z = 03Z+45h)
    # fxx_out 46..48 are NOT covered by 00Z fxx 0..48 -> they may be extrapolated
    # But the first 46 rows are all covered
    covered = result[result["fxx"] < 46]
    assert (covered["is_extrapolated"] == False).all()  # noqa: E712


def test_fill_tail_extrapolation_marks_is_extrapolated_true():
    """Rows beyond nearest_major coverage get is_extrapolated=True."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    # nearest = 00Z fxx 0..48; covers 03Z fxx_out 0..45 (vt up to 03Z+45h=00Z+48h)
    # fxx_out 46, 47, 48 are uncovered -> extrapolated
    nearest = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    extrap = result[result["is_extrapolated"] == True]  # noqa: E712
    assert set(extrap["fxx"]) == {46, 47, 48}


def test_fill_tail_extrapolation_raises_beyond_3h():
    """RuntimeError when uncovered tail exceeds 3 hours."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))
    # nearest = 00Z fxx 0..43; covers 03Z fxx_out 0..40 (vt 03Z+40h = 00Z+43h)
    # fxx_out 41..44 are uncovered (4 steps) -> raises at step 4
    nearest = _make_df("2026-01-01 00:00", range(0, 44))
    with pytest.raises(RuntimeError, match="tail gap exceeds 3-hour cap"):
        fill.fill(non_major, nearest)


def test_fill_tail_extrapolation_values_are_linear():
    """Extrapolated row values follow linear trend from last two aligned rows."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=1.0)
    # nearest = 00Z fxx 0..48; last two covered rows from nearest
    # fxx_out 44: vt = 03Z+44h = 00Z+47h -> nearest fxx 47: col=1.0+47*0.01=1.47
    # fxx_out 45: vt = 03Z+45h = 00Z+48h -> nearest fxx 48: col=1.0+48*0.01=1.48
    # fxx_out 46 (1st extrap): slope = (1.48 - 1.47) = 0.01 -> val ~ 1.48 + 0.01 = 1.49
    nearest = _make_df("2026-01-01 00:00", range(0, 49), col_val=1.0)
    result = fill.fill(non_major, nearest)
    fxx46 = result[result["fxx"] == 46].iloc[0]["TMP_2m"]
    assert fxx46 == pytest.approx(1.48 + 0.01, abs=0.005)


# ---------------------------------------------------------------------------
# V2V02: _vt_index duplicate handling
# ---------------------------------------------------------------------------

def test_fill_dedup_warns_on_duplicate_valid_time():
    """WARNING is emitted when non_major_df has duplicate valid_times."""
    from loguru import logger
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        fill = HrrrFill()
        base = _make_df("2026-01-01 03:00", range(0, 5))
        dup  = _make_df("2026-01-01 03:00", range(0, 3), col_val=99.0)
        non_major = pd.concat([base, dup], ignore_index=True)
        nearest   = _make_df("2026-01-01 00:00", range(0, 49))
        fill.fill(non_major, nearest)
    finally:
        logger.remove(sink_id)
    assert any("duplicate" in m.lower() for m in messages)


def test_fill_dedup_keeps_last_when_duplicate():
    """When valid_times are duplicated, the last row (by valid_time sort) is kept."""
    fill = HrrrFill()
    # Create non_major with fxx=0 appearing twice: first with col_val=1.0, then 99.0
    base = _make_df("2026-01-01 03:00", range(0, 5), col_val=1.0)
    dup  = _make_df("2026-01-01 03:00", range(0, 1), col_val=99.0)
    non_major = pd.concat([base, dup], ignore_index=True)
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    result = fill.fill(non_major, nearest)
    # fxx=0 valid_time is 03Z, appears in both base (1.0) and dup (99.0)
    # After sort ASC + keep last: the dup (99.0) is kept
    fxx0_row = result[result["fxx"] == 0].iloc[0]
    assert fxx0_row["TMP_2m"] == pytest.approx(99.0, abs=0.01)


# ---------------------------------------------------------------------------
# V2V02: sub-second valid_time rounding
# ---------------------------------------------------------------------------

def test_fill_subsecond_valid_time_matches():
    """Sub-second valid_time differences do not prevent matching after floor('s')."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=100.0)
    nearest   = _make_df("2026-01-01 00:00", range(0, 49), col_val=2.0)
    # Add tiny sub-second offset to nearest_major valid_times
    nearest["valid_time"] = pd.to_datetime(
        nearest["valid_time"], utc=True
    ) + pd.Timedelta(microseconds=1)
    result = fill.fill(non_major, nearest)
    # Should still produce a full 49-row result (sub-second offset is floored away)
    assert len(result) == 49
    # fxx 25 should be filled from nearest (not NaN) despite sub-second mismatch
    fxx25_tmp = result[result["fxx"] == 25].iloc[0]["TMP_2m"]
    assert not pd.isna(fxx25_tmp)


# ---------------------------------------------------------------------------
# V2V02: output column union
# ---------------------------------------------------------------------------

def test_fill_column_union_includes_major_only_cols():
    """Columns present only in nearest_major_df appear in fill() output."""
    fill = HrrrFill()
    non_major = _make_df("2026-01-01 03:00", range(0, 19))   # has TMP_2m only
    nearest   = _make_df("2026-01-01 00:00", range(0, 49))
    nearest["DPT_2m"] = nearest["TMP_2m"] - 5.0              # extra col in major
    result = fill.fill(non_major, nearest)
    assert "DPT_2m" in result.columns
    # Tail row fxx=25 should have non-NaN DPT_2m (filled from nearest_major)
    fxx25_dpt = result[result["fxx"] == 25].iloc[0]["DPT_2m"]
    assert not pd.isna(fxx25_dpt)


def test_fill_warning_fires_when_prior_used_not_absent():
    """WARNING fires because prior fills a valid_time, not because nothing fills it."""
    from loguru import logger
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        fill = HrrrFill()
        # nearest only covers 00Z fxx 0-3; prior covers the tail with real values
        non_major = _make_df("2026-01-01 03:00", range(0, 19), col_val=1.0)
        nearest   = _make_df("2026-01-01 00:00", range(0, 4),  col_val=2.0)
        prior     = _make_df("2025-12-31 18:00", range(0, 55), col_val=99.0)
        result = fill.fill(non_major, nearest, prior)
    finally:
        logger.remove(sink_id)
    # prior DID fill some tail rows, so warning must fire
    assert any("tail gap" in m.lower() for m in messages)
    # and those rows have actual values (not NaN)
    fxx19_val = result[result["fxx"] == 19].iloc[0]["TMP_2m"]
    assert not pd.isna(fxx19_val)
