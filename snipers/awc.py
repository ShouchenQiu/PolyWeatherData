"""AWC (Aviation Weather Center) METAR sniper source.

Ported from Herbie-main/combined_sniper.py (awc_task, lines 185-270).

Polling strategy:
  - Normal mode: GET every NORMAL_INTERVAL seconds.
  - Sniper mode: GET every SNIPER_INTERVAL seconds during xx:51:30 - xx:53:59.
  - Change detection: new METAR text differs from all previously-seen observations.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import requests
from loguru import logger

from snipers.base import BaseSniperSource

_HEADERS = {
    "User-Agent": "PolyWeatherBot/1.0",
    "Connection": "keep-alive",
    "Accept":     "text/plain",
}

_REQUEST_TIMEOUT = 4
_SEEN_OBS_MAX = 256


class AwcSniper(BaseSniperSource):
    """Polls AWC raw METAR API for new observations.

    Parameters
    ----------
    station:
        ICAO station identifier, e.g. ``"KLGA"``.
    """

    NORMAL_INTERVAL = 2.0
    SNIPER_INTERVAL = 0.65
    SNIPER_START    = (51, 30)   # (minute, second)
    SNIPER_END      = (53, 59)   # (minute, second) -- inclusive

    def __init__(self, station: str = "KLGA") -> None:
        self._station = station
        self._url     = (
            "https://aviationweather.gov/api/data/metar"
            f"?ids={station}&format=raw&taf=false&hours=1"
        )
        self._running = False

    # ------------------------------------------------------------------
    # Window helper
    # ------------------------------------------------------------------

    def _in_sniper_window(self, now: datetime) -> bool:
        """Return True when the current time falls inside the sniper window."""
        return self.SNIPER_START <= (now.minute, now.second) <= self.SNIPER_END

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get(self, session: requests.Session) -> str | None:
        """Fetch current METAR text; returns None on error or empty response."""
        try:
            r = session.get(self._url, timeout=_REQUEST_TIMEOUT)
            if r.status_code != 200:
                logger.debug("AwcSniper: AWC returned HTTP {}", r.status_code)
                return None
            # API returns newest-first entries separated by '$'; take only the first
            first = r.text.split("$")[0].strip()
            return " ".join(first.split()) or None
        except Exception as exc:
            logger.debug("AwcSniper: AWC request failed: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self, on_observation: Callable) -> None:
        """Run the AWC polling loop until stop() is called."""
        self._running = True
        session       = requests.Session()
        session.headers.update(_HEADERS)
        executor = ThreadPoolExecutor(max_workers=1)
        seen_obs: set[str] = set()
        seen_order: deque[str] = deque()

        async def _blocking(function, *args):
            if self.NORMAL_INTERVAL <= 0 or self.SNIPER_INTERVAL <= 0:
                return function(*args)
            return await asyncio.get_running_loop().run_in_executor(
                executor,
                function,
                *args,
            )

        def _mark_seen(obs: str) -> None:
            if obs in seen_obs:
                return
            seen_obs.add(obs)
            seen_order.append(obs)
            while len(seen_order) > _SEEN_OBS_MAX:
                seen_obs.discard(seen_order.popleft())

        # Baseline: pre-seed seen_obs so the current observation is not
        # forwarded as a new capture on the first poll.
        try:
            warmup = await _blocking(self._get, session)
            if warmup:
                _mark_seen(warmup)
        except Exception as exc:
            logger.debug("AwcSniper: warmup fetch failed: {}", exc)

        try:
            while self._running:
                now = datetime.now(timezone.utc)
                sniper = self._in_sniper_window(now)
                try:
                    raw = await _blocking(self._get, session)
                    if raw and raw not in seen_obs:
                        on_observation(raw, "AWC", datetime.now(timezone.utc))
                        _mark_seen(raw)
                except Exception as exc:
                    logger.warning("AwcSniper: polling iteration failed: {}", exc)
                await asyncio.sleep(
                    self.SNIPER_INTERVAL if sniper else self.NORMAL_INTERVAL
                )
        except asyncio.CancelledError:
            raise
        finally:
            session.close()
            executor.shutdown(wait=True, cancel_futures=True)

    async def stop(self) -> None:
        """Signal the polling loop to exit on the next iteration."""
        self._running = False
