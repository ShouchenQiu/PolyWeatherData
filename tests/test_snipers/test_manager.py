"""Tests for snipers/ package: SniperManager, NoaaSniper, AwcSniper."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from snipers.awc import AwcSniper
from snipers.sniper_manager import SniperManager
from snipers.noaa import NoaaSniper


# -----------------------------------------------------------------------
# SniperManager._on_observation -- deduplication
# -----------------------------------------------------------------------

def test_on_observation_deduplicates():
    """Same METAR from two sources is enqueued only once."""
    mgr   = SniperManager([])
    metar = "KLGA 251951Z 09007KT 10SM FEW030 21/14 A2996"
    mgr._on_observation(metar, "NOAA", datetime.utcnow())
    mgr._on_observation(metar, "AWC",  datetime.utcnow())
    assert mgr.queue.qsize() == 1


def test_on_observation_enqueues_distinct():
    """Two different METARs are both enqueued."""
    mgr = SniperManager([])
    mgr._on_observation("KLGA 251951Z 09007KT 10SM FEW030 21/14 A2996",
                        "NOAA", datetime.utcnow())
    mgr._on_observation("KLGA 252051Z 08007KT 10SM FEW030 20/14 A2997",
                        "AWC",  datetime.utcnow())
    assert mgr.queue.qsize() == 2


def test_on_observation_strips_whitespace():
    """Whitespace padding should not defeat deduplication."""
    mgr   = SniperManager([])
    metar = "KLGA 251951Z 09007KT"
    mgr._on_observation(metar + "  ",  "NOAA", datetime.utcnow())
    mgr._on_observation("  " + metar,  "AWC",  datetime.utcnow())
    assert mgr.queue.qsize() == 1


def test_queue_item_contains_source_and_timestamp():
    """Queued item is a (metar, source, timestamp) tuple."""
    mgr = SniperManager([])
    ts  = datetime(2026, 3, 25, 20, 0, 0)
    mgr._on_observation("KLGA obs", "NOAA", ts)
    metar_out, source_out, ts_out = mgr.queue.get_nowait()
    assert source_out == "NOAA"
    assert ts_out     == ts


def test_seen_obs_tracks_unique_texts():
    """seen_obs grows with each unique METAR."""
    mgr = SniperManager([])
    mgr._on_observation("METAR A", "NOAA", datetime.utcnow())
    mgr._on_observation("METAR B", "NOAA", datetime.utcnow())
    mgr._on_observation("METAR A", "AWC",  datetime.utcnow())
    assert len(mgr.seen_obs) == 2


# -----------------------------------------------------------------------
# SniperManager.start_all -- async, two sources fire the same METAR
# -----------------------------------------------------------------------

def test_start_all_two_sources_same_metar():
    """Both sources fire the same METAR; queue receives it only once."""
    metar = "KLGA 252051Z 09009KT 10SM BKN040 20/14 A2997"
    ts    = datetime.utcnow()

    async def _fake_start_noaa(on_obs):
        on_obs(metar, "NOAA", ts)

    async def _fake_start_awc(on_obs):
        on_obs(metar, "AWC", ts)

    src1 = MagicMock()
    src1.start = AsyncMock(side_effect=_fake_start_noaa)
    src2 = MagicMock()
    src2.start = AsyncMock(side_effect=_fake_start_awc)

    mgr = SniperManager([src1, src2])
    asyncio.run(mgr.start_all())
    assert mgr.queue.qsize() == 1


def test_start_all_two_sources_different_metars():
    """Both sources fire different METARs; both are enqueued."""
    async def _start1(on_obs):
        on_obs("KLGA obs1", "NOAA", datetime.utcnow())

    async def _start2(on_obs):
        on_obs("KLGA obs2", "AWC", datetime.utcnow())

    src1 = MagicMock()
    src1.start = AsyncMock(side_effect=_start1)
    src2 = MagicMock()
    src2.start = AsyncMock(side_effect=_start2)

    mgr = SniperManager([src1, src2])
    asyncio.run(mgr.start_all())
    assert mgr.queue.qsize() == 2


# -----------------------------------------------------------------------
# NoaaSniper._sniper_window -- window boundary arithmetic
# -----------------------------------------------------------------------

def test_noaa_window_before_start():
    """Before the sniper window, ws is in the future."""
    sniper = NoaaSniper()
    now    = datetime(2026, 3, 25, 14, 30, 0)
    ws, we = sniper._sniper_window(now)
    assert ws > now
    assert ws.minute == 50
    assert ws.second == 45


def test_noaa_window_inside():
    """During the window, now is between ws (past) and we (future)."""
    sniper = NoaaSniper()
    now    = datetime(2026, 3, 25, 14, 53, 0)
    ws, we = sniper._sniper_window(now)
    assert ws <= now
    assert we > now


def test_noaa_window_after_end_rolls_to_next_hour():
    """After xx:56:00, the window rolls forward to the next hour."""
    sniper = NoaaSniper()
    now    = datetime(2026, 3, 25, 14, 57, 0)
    ws, we = sniper._sniper_window(now)
    assert ws.hour   == 15
    assert ws.minute == 50


def test_noaa_window_exactly_at_end_rolls():
    """At exactly xx:56:00 (window end), it rolls to the next hour."""
    sniper = NoaaSniper()
    now    = datetime(2026, 3, 25, 14, 56, 0)
    ws, we = sniper._sniper_window(now)
    assert ws.hour == 15


# -----------------------------------------------------------------------
# AwcSniper._in_sniper_window -- tuple comparison boundary checks
# -----------------------------------------------------------------------

def test_awc_in_window_inside():
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 52, 30)) is True


def test_awc_in_window_before_start_second():
    """xx:51:00 -- minute matches but second is before 30."""
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 51, 0)) is False


def test_awc_in_window_after_end():
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 54, 0)) is False


def test_awc_in_window_at_start_boundary():
    """Exactly at (51, 30) -- inclusive start."""
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 51, 30)) is True


def test_awc_in_window_at_end_boundary():
    """Exactly at (53, 59) -- inclusive end."""
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 53, 59)) is True


def test_awc_not_in_window_outside_hour():
    """xx:00:00 is completely outside the window."""
    assert AwcSniper()._in_sniper_window(datetime(2026, 3, 25, 14, 0, 0)) is False
