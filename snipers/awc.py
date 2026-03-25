"""AWC (Aviation Weather Center) METAR sniper source.

Ported from Herbie-main/combined_sniper.py (awc_task, lines 185-270).

Polling strategy:
  - Normal mode: GET every NORMAL_INTERVAL seconds.
  - Sniper mode: GET every SNIPER_INTERVAL seconds during xx:51:30 - xx:53:59.
  - Change detection: new METAR text differs from all previously-seen observations.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable

import requests

from snipers.base import BaseSniperSource

_HEADERS = {
    "User-Agent": "PolyWeatherBot/1.0",
    "Connection": "keep-alive",
    "Accept":     "text/plain",
}

_REQUEST_TIMEOUT = 4


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
    SNIPER_END      = (53, 59)   # (minute, second) ¡ª inclusive

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
                return None
            # API returns newest-first entries separated by '$'; take only the first
            first = r.text.split("$")[0].strip()
            return " ".join(first.split()) or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self, on_observation: Callable) -> None:
        """Run the AWC polling loop until stop() is called."""
        self._running = True
        session       = requests.Session()
        session.headers.update(_HEADERS)
        seen_obs: set[str] = set()

        # Baseline: pre-seed seen_obs so the current observation is not
        # forwarded as a new capture on the first poll.
        try:
            warmup = await asyncio.to_thread(self._get, session)
            if warmup:
                seen_obs.add(warmup)
        except Exception:
            pass

        try:
            while self._running:
                sniper = self._in_sniper_window(datetime.now())
                try:
                    raw = await asyncio.to_thread(self._get, session)
                    if raw and raw not in seen_obs:
                        seen_obs.add(raw)
                        on_observation(raw, "AWC", datetime.utcnow())
                except Exception:
                    pass
                await asyncio.sleep(
                    self.SNIPER_INTERVAL if sniper else self.NORMAL_INTERVAL
                )
        except asyncio.CancelledError:
            pass
        finally:
            session.close()

    async def stop(self) -> None:
        """Signal the polling loop to exit on the next iteration."""
        self._running = False
