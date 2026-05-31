"""Tests for core/probability_store.py -- ProbabilityStore V2 shape."""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone

import pytest

from core.probability_store import ProbabilitySnapshot, ProbabilityStore

_TS = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
_QUANTILES = {1: {"p10": 58.0, "p50": 63.0, "p90": 68.0}}


# ---------------------------------------------------------------------------
# ProbabilitySnapshot construction
# ---------------------------------------------------------------------------


def test_probability_snapshot_fields() -> None:
    snap = ProbabilitySnapshot(
        buckets={(60.0, 70.0): 0.65},
        quantiles_by_fxx=_QUANTILES,
        last_tick_utc=_TS,
    )
    assert snap.buckets[(60.0, 70.0)] == pytest.approx(0.65)
    assert snap.quantiles_by_fxx[1]["p50"] == 63.0
    assert snap.last_tick_utc is _TS


# ---------------------------------------------------------------------------
# ProbabilityStore write / read helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    mc_prob: float,
    *,
    quantiles_by_fxx: dict | None = None,
    ts: datetime = _TS,
    key: tuple[float, float] = (60.0, 70.0),
) -> ProbabilitySnapshot:
    return ProbabilitySnapshot(
        buckets={key: mc_prob},
        quantiles_by_fxx=quantiles_by_fxx if quantiles_by_fxx is not None else copy.deepcopy(_QUANTILES),
        last_tick_utc=ts,
    )


# ---------------------------------------------------------------------------
# ProbabilityStore write / read
# ---------------------------------------------------------------------------


def test_write_and_read_back_single_bucket() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.72))
        result = await store.read("cid-001", 0)
        assert result is not None
        assert result.buckets[(60.0, 70.0)] == pytest.approx(0.72)

    asyncio.run(run())


def test_write_and_read_multiple_day_offsets() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.60))
        await store.write("cid-001", 1, _make_snapshot(0.70))
        await store.write("cid-001", 2, _make_snapshot(0.80))

        r0 = await store.read("cid-001", 0)
        r1 = await store.read("cid-001", 1)
        r2 = await store.read("cid-001", 2)

        assert r0 is not None and r0.buckets[(60.0, 70.0)] == pytest.approx(0.60)
        assert r1 is not None and r1.buckets[(60.0, 70.0)] == pytest.approx(0.70)
        assert r2 is not None and r2.buckets[(60.0, 70.0)] == pytest.approx(0.80)

    asyncio.run(run())


def test_read_returns_none_for_missing_key() -> None:
    store = ProbabilityStore()

    async def run() -> ProbabilitySnapshot | None:
        return await store.read("cid-nonexistent", 0)

    result = asyncio.run(run())
    assert result is None


def test_write_overwrites_previous_value() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.50))
        await store.write("cid-001", 0, _make_snapshot(0.99))
        result = await store.read("cid-001", 0)
        assert result is not None
        assert result.buckets[(60.0, 70.0)] == pytest.approx(0.99)

    asyncio.run(run())


def test_write_multi_bucket_snapshot() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        snap = ProbabilitySnapshot(
            buckets={(60.0, 70.0): 0.50, (70.0, 80.0): 0.60},
            quantiles_by_fxx=_QUANTILES,
            last_tick_utc=_TS,
        )
        await store.write("cid-001", 0, snap)
        result = await store.read("cid-001", 0)
        assert result is not None
        assert len(result.buckets) == 2
        assert result.buckets[(70.0, 80.0)] == pytest.approx(0.60)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# day_offset validation
# ---------------------------------------------------------------------------


def test_day_offset_out_of_range_raises() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        with pytest.raises(ValueError, match="day_offset"):
            await store.write("cid-001", 3, _make_snapshot(0.5))

    asyncio.run(run())


def test_day_offset_negative_raises() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        with pytest.raises(ValueError, match="day_offset"):
            await store.write("cid-001", -1, _make_snapshot(0.5))

    asyncio.run(run())


def test_read_invalid_day_offset_raises() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        with pytest.raises(ValueError, match="day_offset"):
            await store.read("cid-001", 5)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Concurrent write safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_do_not_corrupt_state() -> None:
    """asyncio.gather concurrent writes for different keys are all visible."""
    store = ProbabilityStore()

    async def run() -> None:
        tasks = [
            store.write(
                f"cid-{i}", 0,
                _make_snapshot(float(i) / 100, key=(float(i), float(i + 10))),
            )
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        for i in range(10):
            result = await store.read(f"cid-{i}", 0)
            assert result is not None
            assert result.buckets[(float(i), float(i + 10))] == pytest.approx(float(i) / 100)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Immutability / isolation guarantees
# ---------------------------------------------------------------------------


def test_mutating_read_result_does_not_corrupt_store() -> None:
    """Modifying quantiles from a read() result must not affect the stored snapshot."""
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.55))
        result = await store.read("cid-001", 0)
        assert result is not None
        # Mutate the nested quantiles dict from the read result
        result.quantiles_by_fxx[1]["p50"] = 999.0

        # Re-read; store must be unaffected
        result2 = await store.read("cid-001", 0)
        assert result2 is not None
        assert result2.quantiles_by_fxx[1]["p50"] == 63.0

    asyncio.run(run())


def test_mutating_written_snapshot_does_not_corrupt_store() -> None:
    """Modifying the snapshot passed to write() must not affect the stored data."""
    store = ProbabilityStore()
    snap = _make_snapshot(0.40)

    async def run() -> None:
        await store.write("cid-001", 0, snap)
        # Mutate caller-side nested dict after the write
        snap.quantiles_by_fxx[1]["p50"] = 999.0

        result = await store.read("cid-001", 0)
        assert result is not None
        assert result.quantiles_by_fxx[1]["p50"] == 63.0

    asyncio.run(run())


# ---------------------------------------------------------------------------
# ProbabilitySnapshot interface
# ---------------------------------------------------------------------------


def test_probability_snapshot_last_tick_utc_is_top_level() -> None:
    """read() exposes last_tick_utc directly on the snapshot."""
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.55))
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert isinstance(snap, ProbabilitySnapshot)
        assert snap.last_tick_utc == _TS

    asyncio.run(run())


def test_probability_snapshot_quantiles_is_top_level() -> None:
    """read() exposes quantiles_by_fxx directly on the snapshot (not per bucket)."""
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.55))
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert snap.quantiles_by_fxx[1]["p50"] == 63.0

    asyncio.run(run())


def test_probability_snapshot_is_fresh_within_threshold() -> None:
    """is_fresh() returns True when age <= threshold_min."""
    store = ProbabilityStore()
    now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    ts  = datetime(2026, 7, 4, 11, 0, 0, tzinfo=timezone.utc)  # 60 min ago

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.55, ts=ts))
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert snap.is_fresh(threshold_min=65, now=now) is True
        assert snap.is_fresh(threshold_min=55, now=now) is False

    asyncio.run(run())


def test_probability_snapshot_age_minutes() -> None:
    """age_minutes() returns elapsed minutes since last_tick_utc."""
    store = ProbabilityStore()
    now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    ts  = datetime(2026, 7, 4, 11, 30, 0, tzinfo=timezone.utc)  # 30 min ago

    async def run() -> None:
        await store.write("cid-001", 0, _make_snapshot(0.55, ts=ts))
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert snap.age_minutes(now) == pytest.approx(30.0)

    asyncio.run(run())
