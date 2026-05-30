"""Tests for snipers/ package: SniperManager, NoaaSniper, AwcSniper."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
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


class _NoaaResp:
    def __init__(
        self,
        status_code: int,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class _NoaaSession:
    def __init__(
        self,
        head_response: _NoaaResp | Exception | None = None,
        get_response: _NoaaResp | Exception | None = None,
    ) -> None:
        self.head_response = head_response or _NoaaResp(200)
        self.get_response = get_response or _NoaaResp(200)

    def head(self, *args, **kwargs):
        if isinstance(self.head_response, Exception):
            raise self.head_response
        return self.head_response

    def get(self, *args, **kwargs):
        if isinstance(self.get_response, Exception):
            raise self.get_response
        return self.get_response


def test_noaa_head_returns_last_modified():
    session = _NoaaSession(
        head_response=_NoaaResp(200, headers={"Last-Modified": "mod-1"})
    )

    assert NoaaSniper()._head(session) == "mod-1"


def test_noaa_head_failure_returns_empty():
    assert NoaaSniper()._head(_NoaaSession(head_response=_NoaaResp(503))) == ""
    assert NoaaSniper()._head(_NoaaSession(head_response=RuntimeError("net"))) == ""


def test_noaa_get_normalizes_metar_line():
    session = _NoaaSession(
        get_response=_NoaaResp(200, "2026/03/25 19:51\n KLGA   251951Z   20/14 ")
    )

    assert NoaaSniper()._get(session) == "KLGA 251951Z 20/14"


def test_noaa_get_failure_returns_none():
    assert NoaaSniper()._get(_NoaaSession(get_response=_NoaaResp(500))) is None
    assert NoaaSniper()._get(_NoaaSession(get_response=RuntimeError("net"))) is None


def test_noaa_start_retries_idle_callback_failure_and_uses_aware_utc(monkeypatch):
    """last_mod should advance only after the callback succeeds."""
    sniper = NoaaSniper()
    sniper._head = MagicMock(side_effect=["old", "new", "new"])
    sniper._get = MagicMock(return_value="KLGA 251951Z 20/14")
    sniper._sniper_window = MagicMock(
        side_effect=lambda now: (
            now + timedelta(seconds=60),
            now + timedelta(seconds=360),
        )
    )
    sniper.IDLE_INTERVAL = 0
    sessions: list[MagicMock] = []

    def _session_factory():
        session = MagicMock()
        sessions.append(session)
        return session

    monkeypatch.setattr("snipers.noaa.requests.Session", _session_factory)
    sleep_calls = 0
    seen: list[tuple[str, str, datetime]] = []

    def _callback(raw, source, timestamp):
        seen.append((raw, source, timestamp))
        if len(seen) == 1:
            raise RuntimeError("callback failed")

    async def _sleep(_secs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sniper.start(_callback))

    assert [item[0] for item in seen] == [
        "KLGA 251951Z 20/14",
        "KLGA 251951Z 20/14",
    ]
    assert seen[-1][1] == "NOAA"
    assert seen[-1][2].tzinfo is timezone.utc
    assert sessions[0].close.called


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


# -----------------------------------------------------------------------
# AwcSniper._get / start -- response normalization and callback behavior
# -----------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _Session:
    def __init__(self, response: _Resp | Exception) -> None:
        self.response = response

    def get(self, *args, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_awc_get_normalizes_first_metar():
    """AWC responses may contain '$'-separated rows; only newest row is used."""
    sniper = AwcSniper()
    session = _Session(_Resp(200, " KLGA   251951Z   20/14  $ KLGA old "))

    assert sniper._get(session) == "KLGA 251951Z 20/14"


def test_awc_get_non_200_returns_none():
    assert AwcSniper()._get(_Session(_Resp(503, "unavailable"))) is None


def test_awc_get_exception_returns_none():
    assert AwcSniper()._get(_Session(RuntimeError("network"))) is None


def test_awc_start_warmup_does_not_emit_and_new_obs_uses_aware_utc(monkeypatch):
    """Warmup seeds seen_obs; the next new METAR is emitted with aware UTC time."""
    sniper = AwcSniper()
    sniper._get = MagicMock(side_effect=["KLGA warmup", "KLGA new 20/14"])
    sniper.NORMAL_INTERVAL = 0
    sniper.SNIPER_INTERVAL = 0
    seen: list[tuple[str, str, datetime]] = []

    async def _stop_after_first_sleep(_secs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _stop_after_first_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sniper.start(lambda *args: seen.append(args)))

    assert len(seen) == 1
    assert seen[0][0] == "KLGA new 20/14"
    assert seen[0][1] == "AWC"
    assert seen[0][2].tzinfo is timezone.utc


def test_awc_start_retries_when_callback_fails(monkeypatch):
    """A callback failure must not permanently mark the METAR as seen."""
    sniper = AwcSniper()
    sniper._get = MagicMock(side_effect=[None, "KLGA new 20/14", "KLGA new 20/14"])
    sniper.NORMAL_INTERVAL = 0
    sniper.SNIPER_INTERVAL = 0
    calls: list[str] = []
    sleep_calls = 0

    def _callback(raw, _source, _timestamp):
        calls.append(raw)
        if len(calls) == 1:
            raise RuntimeError("callback failed")

    async def _sleep(_secs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sniper.start(_callback))

    assert calls == ["KLGA new 20/14", "KLGA new 20/14"]
