"""Tests for core/probability_store.py -- ProbabilityStore."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from core.probability_store import BucketPayload, ProbabilitySnapshot, ProbabilityStore


# ---------------------------------------------------------------------------
# BucketPayload construction
# ---------------------------------------------------------------------------


def test_bucket_payload_fields() -> None:
    now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    bp = BucketPayload(
        mc_prob=0.65,
        quantiles_by_fxx={1: {"p10": 60.0, "p50": 65.0, "p90": 70.0}},
        last_tick_utc=now,
    )
    assert bp.mc_prob == 0.65
    assert bp.quantiles_by_fxx[1]["p50"] == 65.0
    assert bp.last_tick_utc is now


# ---------------------------------------------------------------------------
# ProbabilityStore write / read
# ---------------------------------------------------------------------------


def _make_payload(mc_prob: float) -> BucketPayload:
    return BucketPayload(
        mc_prob=mc_prob,
        quantiles_by_fxx={1: {"p10": 58.0, "p50": 63.0, "p90": 68.0}},
        last_tick_utc=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_write_and_read_back_single_bucket() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        buckets = {(60.0, 70.0): _make_payload(0.72)}
        await store.write("cid-001", 0, buckets)
        result = await store.read("cid-001", 0)
        assert result is not None
        assert abs(result.buckets[(60.0, 70.0)].mc_prob - 0.72) < 1e-9

    asyncio.run(run())


def test_write_and_read_multiple_day_offsets() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): _make_payload(0.60)})
        await store.write("cid-001", 1, {(60.0, 70.0): _make_payload(0.70)})
        await store.write("cid-001", 2, {(60.0, 70.0): _make_payload(0.80)})

        r0 = await store.read("cid-001", 0)
        r1 = await store.read("cid-001", 1)
        r2 = await store.read("cid-001", 2)

        assert r0 is not None and abs(r0.buckets[(60.0, 70.0)].mc_prob - 0.60) < 1e-9
        assert r1 is not None and abs(r1.buckets[(60.0, 70.0)].mc_prob - 0.70) < 1e-9
        assert r2 is not None and abs(r2.buckets[(60.0, 70.0)].mc_prob - 0.80) < 1e-9

    asyncio.run(run())


def test_read_returns_none_for_missing_key() -> None:
    store = ProbabilityStore()

    async def run() -> BucketPayload | None:
        return await store.read("cid-nonexistent", 0)

    result = asyncio.run(run())
    assert result is None


def test_write_overwrites_previous_value() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): _make_payload(0.50)})
        await store.write("cid-001", 0, {(60.0, 70.0): _make_payload(0.99)})
        result = await store.read("cid-001", 0)
        assert result is not None
        assert abs(result.buckets[(60.0, 70.0)].mc_prob - 0.99) < 1e-9

    asyncio.run(run())


# ---------------------------------------------------------------------------
# day_offset validation
# ---------------------------------------------------------------------------


def test_day_offset_out_of_range_raises() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        with pytest.raises(ValueError, match="day_offset"):
            await store.write("cid-001", 3, {(60.0, 70.0): _make_payload(0.5)})

    asyncio.run(run())


def test_day_offset_negative_raises() -> None:
    store = ProbabilityStore()

    async def run() -> None:
        with pytest.raises(ValueError, match="day_offset"):
            await store.write("cid-001", -1, {(60.0, 70.0): _make_payload(0.5)})

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
            store.write(f"cid-{i}", 0, {(float(i), float(i + 10)): _make_payload(float(i) / 100)})
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        for i in range(10):
            result = await store.read(f"cid-{i}", 0)
            assert result is not None
            bucket = (float(i), float(i + 10))
            assert abs(result.buckets[bucket].mc_prob - float(i) / 100) < 1e-9

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Immutability / isolation guarantees
# ---------------------------------------------------------------------------


def test_mutating_read_result_does_not_corrupt_store() -> None:
    """Modifying a value from read() must not affect the stored payload."""
    store = ProbabilityStore()

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): _make_payload(0.55)})
        result = await store.read("cid-001", 0)
        assert result is not None
        # Mutate the nested dict from the read result
        result.buckets[(60.0, 70.0)].quantiles_by_fxx[1]["p50"] = 999.0

        # Re-read; store must be unaffected
        result2 = await store.read("cid-001", 0)
        assert result2 is not None
        assert result2.buckets[(60.0, 70.0)].quantiles_by_fxx[1]["p50"] == 63.0

    asyncio.run(run())


def test_mutating_written_buckets_does_not_corrupt_store() -> None:
    """Modifying the dict passed to write() must not affect the stored payload."""
    store = ProbabilityStore()
    payload = _make_payload(0.40)
    original_q = payload.quantiles_by_fxx[1]["p50"]
    buckets: dict = {(60.0, 70.0): payload}

    async def run() -> None:
        await store.write("cid-001", 0, buckets)
        # Mutate the caller-side nested dict after the write
        buckets[(60.0, 70.0)].quantiles_by_fxx[1]["p50"] = 999.0

        result = await store.read("cid-001", 0)
        assert result is not None
        assert result.buckets[(60.0, 70.0)].quantiles_by_fxx[1]["p50"] == original_q

    asyncio.run(run())


# ---------------------------------------------------------------------------
# last_tick_utc consistency
# ---------------------------------------------------------------------------


def test_write_inconsistent_last_tick_utc_raises() -> None:
    """write() rejects a bucket map with two different last_tick_utc values."""
    store = ProbabilityStore()
    ts1 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 7, 4, 13, 0, 0, tzinfo=timezone.utc)
    p1 = BucketPayload(mc_prob=0.50, quantiles_by_fxx={}, last_tick_utc=ts1)
    p2 = BucketPayload(mc_prob=0.60, quantiles_by_fxx={}, last_tick_utc=ts2)

    async def run() -> None:
        with pytest.raises(ValueError, match="last_tick_utc"):
            await store.write("cid-001", 0, {(60.0, 70.0): p1, (70.0, 80.0): p2})

    asyncio.run(run())


def test_write_consistent_last_tick_utc_accepted() -> None:
    """write() accepts a bucket map where all payloads share last_tick_utc."""
    store = ProbabilityStore()
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    p1 = BucketPayload(mc_prob=0.50, quantiles_by_fxx={}, last_tick_utc=ts)
    p2 = BucketPayload(mc_prob=0.60, quantiles_by_fxx={}, last_tick_utc=ts)

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): p1, (70.0, 80.0): p2})
        result = await store.read("cid-001", 0)
        assert result is not None
        assert len(result.buckets) == 2


# ---------------------------------------------------------------------------
# ProbabilitySnapshot interface
# ---------------------------------------------------------------------------


def test_probability_snapshot_last_tick_utc_is_top_level() -> None:
    """read() exposes last_tick_utc directly on the snapshot."""
    store = ProbabilityStore()
    ts    = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    p     = BucketPayload(mc_prob=0.55, quantiles_by_fxx={}, last_tick_utc=ts)

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): p})
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert isinstance(snap, ProbabilitySnapshot)
        assert snap.last_tick_utc == ts

    asyncio.run(run())


def test_probability_snapshot_is_fresh_within_threshold() -> None:
    """is_fresh() returns True when age <= threshold_min."""
    store = ProbabilityStore()
    now   = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    ts    = datetime(2026, 7, 4, 11, 0, 0, tzinfo=timezone.utc)  # 60 min ago
    p     = BucketPayload(mc_prob=0.55, quantiles_by_fxx={}, last_tick_utc=ts)

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): p})
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert snap.is_fresh(threshold_min=65, now=now) is True
        assert snap.is_fresh(threshold_min=55, now=now) is False

    asyncio.run(run())


def test_probability_snapshot_age_minutes() -> None:
    """age_minutes() returns elapsed minutes since last_tick_utc."""
    store = ProbabilityStore()
    now   = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    ts    = datetime(2026, 7, 4, 11, 30, 0, tzinfo=timezone.utc)  # 30 min ago
    p     = BucketPayload(mc_prob=0.55, quantiles_by_fxx={}, last_tick_utc=ts)

    async def run() -> None:
        await store.write("cid-001", 0, {(60.0, 70.0): p})
        snap = await store.read("cid-001", 0)
        assert snap is not None
        assert snap.age_minutes(now) == pytest.approx(30.0)

    asyncio.run(run())

    asyncio.run(run())
