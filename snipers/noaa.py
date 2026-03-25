"""NOAA tgftp METAR sniper source.

Ported from Herbie-main/combined_sniper.py (noaa_task, lines 67-175).

Polling strategy:
  - Idle mode  : HEAD every IDLE_INTERVAL seconds, watch for Last-Modified change.
  - Sniper mode: HEAD every SNIPER_INTERVAL seconds during xx:50:45 - xx:56:00.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Callable

import requests

from snipers.base import BaseSniperSource

_HEADERS = {
    "User-Agent": "PolyWeatherBot/1.0",
    "Connection": "keep-alive",
}

_REQUEST_TIMEOUT = 4


class NoaaSniper(BaseSniperSource):
    """Polls NOAA tgftp for METAR updates via HTTP HEAD + GET.

    Parameters
    ----------
    station:
        ICAO station identifier, e.g. ``"KLGA"``.
    """

    IDLE_INTERVAL   = 3.0
    SNIPER_INTERVAL = 0.2
    SNIPER_START    = (50, 45)   # (minute, second)
    SNIPER_END      = (56, 0)    # (minute, second) ¡ª exclusive end-minute

    def __init__(self, station: str = "KLGA") -> None:
        self._station = station
        self._url     = (
            "https://tgftp.nws.noaa.gov/data/observations/metar"
            f"/stations/{station}.TXT"
        )
        self._running = False

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def _sniper_window(self, now: datetime):
        """Return (window_start, window_end) datetimes for the current hour."""
        sm, ss = self.SNIPER_START
        em, _  = self.SNIPER_END
        ws = now.replace(minute=sm, second=ss, microsecond=0)
        we = now.replace(minute=em, second=0,  microsecond=0)
        if now >= we:
            ws += timedelta(hours=1)
            we += timedelta(hours=1)
        return ws, we

    # ------------------------------------------------------------------
    # HTTP helpers (run in thread pool to avoid blocking the event loop)
    # ------------------------------------------------------------------

    def _head(self, session: requests.Session) -> str:
        r = session.head(self._url, timeout=_REQUEST_TIMEOUT)
        return r.headers.get("Last-Modified", "")

    def _get(self, session: requests.Session) -> str:
        r = session.get(self._url, timeout=_REQUEST_TIMEOUT)
        return r.text

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self, on_observation: Callable) -> None:
        """Run the NOAA polling loop until stop() is called."""
        self._running = True
        idle_session  = requests.Session()
        idle_session.headers.update(_HEADERS)

        try:
            last_mod = await asyncio.to_thread(self._head, idle_session)
        except Exception:
            last_mod = ""

        while self._running:
            now      = datetime.now()
            ws, we   = self._sniper_window(now)

            if now < ws:
                # ---- Idle mode: low-frequency polling ----
                try:
                    cur_mod = await asyncio.to_thread(self._head, idle_session)
                    if cur_mod and cur_mod != last_mod:
                        content  = await asyncio.to_thread(self._get, idle_session)
                        last_mod = cur_mod
                        on_observation(content, "NOAA", datetime.utcnow())
                except Exception:
                    pass
                secs_to_window = (ws - datetime.now()).total_seconds()
                await asyncio.sleep(min(secs_to_window, self.IDLE_INTERVAL))
                continue

            # ---- Sniper window ----
            session = requests.Session()
            session.headers.update(_HEADERS)
            try:
                # TCP warm-up HEAD before tight polling loop
                try:
                    warm_mod = await asyncio.to_thread(self._head, session)
                    if warm_mod and warm_mod != last_mod:
                        content  = await asyncio.to_thread(self._get, session)
                        last_mod = warm_mod
                        on_observation(content, "NOAA", datetime.utcnow())
                        retreat = (we - datetime.now()).total_seconds() + 60
                        await asyncio.sleep(max(retreat, 1))
                        continue
                except Exception:
                    await asyncio.sleep(1)
                    continue

                captured = False
                while self._running and datetime.now() < we and not captured:
                    try:
                        cur_mod = await asyncio.to_thread(self._head, session)
                        if cur_mod and cur_mod != last_mod:
                            content  = await asyncio.to_thread(self._get, session)
                            last_mod = cur_mod
                            on_observation(content, "NOAA", datetime.utcnow())
                            captured = True
                            retreat  = (we - datetime.now()).total_seconds() + 60
                            await asyncio.sleep(max(retreat, 1))
                    except Exception:
                        pass
                    if not captured:
                        await asyncio.sleep(self.SNIPER_INTERVAL)
            finally:
                session.close()

            if not captured:
                next_ws, _ = self._sniper_window(datetime.now())
                await asyncio.sleep(
                    min((next_ws - datetime.now()).total_seconds(), self.IDLE_INTERVAL)
                )

    async def stop(self) -> None:
        """Signal the polling loop to exit on the next iteration."""
        self._running = False
