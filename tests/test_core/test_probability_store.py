"""Tests for core/probability_store.py -- ProbabilityStore."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from core.probability_store import BucketPayload, ProbabilityStore


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
        assert abs(result[(60.0, 70.0)].mc_prob - 0.72) < 1e-9

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

        assert r0 is not None and abs(r0[(60.0, 70.0)].mc_prob - 0.60) < 1e-9
        assert r1 is not None and abs(r1[(60.0, 70.0)].mc_prob - 0.70) < 1e-9
        assert r2 is not None and abs(r2[(60.0, 70.0)].mc_prob - 0.80) < 1e-9

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
        assert abs(result[(60.0, 70.0)].mc_prob - 0.99) < 1e-9

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
            assert abs(result[bucket].mc_prob - float(i) / 100) < 1e-9

    asyncio.run(run())
