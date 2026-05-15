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
