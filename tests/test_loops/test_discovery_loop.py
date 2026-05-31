"""Tests for loops/discovery_loop.py -- DiscoveryLoop (V09)."""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from loops.discovery_loop import DiscoveryLoop


# -----------------------------------------------------------------------
# Cadence
# -----------------------------------------------------------------------

def test_discovery_loop_default_interval_is_hourly():
    """Default polling interval is 3600 s (1 hour), not 86400 s (daily)."""
    loop = DiscoveryLoop(MagicMock())
    assert loop._interval_seconds == pytest.approx(3_600.0)


def test_discovery_loop_custom_interval_accepted():
    """Custom interval_seconds is stored as given."""
    loop = DiscoveryLoop(MagicMock(), interval_seconds=1800.0)
    assert loop._interval_seconds == pytest.approx(1_800.0)


# -----------------------------------------------------------------------
# No cutoff references
# -----------------------------------------------------------------------

def test_discovery_loop_has_no_cutoff_references():
    """discovery_loop module must not reference removed cutoff settings."""
    import loops.discovery_loop as mod
    src = inspect.getsource(mod)
    assert "cutoff_hour_utc" not in src
    assert "min_hours_before_cutoff" not in src


# -----------------------------------------------------------------------
# run() invokes discover()
# -----------------------------------------------------------------------

def test_discovery_loop_run_calls_discover():
    """run() must call discovery.discover() at least once."""
    discovery = MagicMock()
    loop      = DiscoveryLoop(discovery, interval_seconds=0.0)

    async def _run_one_tick():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_tick())
    discovery.discover.assert_called()


# -----------------------------------------------------------------------
# V2V01 -- lifecycle advance pass
# -----------------------------------------------------------------------

def test_discovery_loop_accepts_db_and_lifecycle():
    """DiscoveryLoop.__init__ must accept optional db and lifecycle parameters."""
    discovery = MagicMock()
    db        = MagicMock()
    lifecycle = MagicMock()
    loop = DiscoveryLoop(discovery, interval_seconds=0.0, db=db, lifecycle=lifecycle)
    assert loop._db is db
    assert loop._lifecycle is lifecycle


def test_discovery_loop_advance_pass_called_after_discover():
    """run() calls lifecycle.advance for each non-terminal event after discover()."""
    from datetime import datetime, timezone

    discovery = MagicMock()
    lifecycle = MagicMock()
    lifecycle.advance.return_value = "active"

    db = MagicMock()
    db.get_non_terminal_events.return_value = [
        {"condition_id": "cid-1", "status": "active",     "event_end_date": "2026-06-01"},
        {"condition_id": "cid-2", "status": "discovered", "event_end_date": "2025-01-01"},
    ]

    loop = DiscoveryLoop(discovery, interval_seconds=0.0, db=db, lifecycle=lifecycle)

    async def _run_one_tick():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_tick())
    assert lifecycle.advance.call_count >= 2


def test_discovery_loop_writes_status_change_to_db():
    """When lifecycle.advance returns a new status, db.update_event_status is called."""
    from events.lifecycle import EventLifecycle

    discovery = MagicMock()
    lifecycle = MagicMock()
    # advance returns 'pending_settle' for the expired event
    lifecycle.advance.return_value = "pending_settle"

    db = MagicMock()
    db.get_non_terminal_events.return_value = [
        {"condition_id": "cid-1", "status": "active", "event_end_date": "2025-01-01"},
    ]

    loop = DiscoveryLoop(discovery, interval_seconds=0.0, db=db, lifecycle=lifecycle)

    async def _run_one_tick():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_tick())
    db.update_event_status.assert_called_with("cid-1", "pending_settle")


def test_discovery_loop_no_status_change_skips_db_write():
    """When lifecycle.advance returns same status (no transition), db.update_event_status is NOT called."""
    discovery = MagicMock()
    lifecycle = MagicMock()
    lifecycle.advance.return_value = "active"  # same as current status -- no change

    db = MagicMock()
    db.get_non_terminal_events.return_value = [
        {"condition_id": "cid-1", "status": "active", "event_end_date": "2026-06-05"},
    ]

    loop = DiscoveryLoop(discovery, interval_seconds=0.0, db=db, lifecycle=lifecycle)

    async def _run_one_tick():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_tick())
    db.update_event_status.assert_not_called()


# -----------------------------------------------------------------------
# Partial-dep guard (Batch 3 arch fix)
# -----------------------------------------------------------------------

def test_discovery_loop_raises_when_only_db_provided():
    """Providing db without lifecycle must raise ValueError."""
    with pytest.raises(ValueError, match="both"):
        DiscoveryLoop(MagicMock(), db=MagicMock())


def test_discovery_loop_raises_when_only_lifecycle_provided():
    """Providing lifecycle without db must raise ValueError."""
    with pytest.raises(ValueError, match="both"):
        DiscoveryLoop(MagicMock(), lifecycle=MagicMock())


def test_discovery_loop_allows_both_none():
    """Omitting both db and lifecycle is valid (lifecycle advance pass is skipped)."""
    loop = DiscoveryLoop(MagicMock())
    assert loop._db is None
    assert loop._lifecycle is None
