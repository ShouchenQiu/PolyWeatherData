"""Polymarket event discovery via the Gamma API.

Fetches active weather-temperature events, parses their bucket slugs into
(lo_f, hi_f) pairs, and upserts them into the local database.

Reference: py-clob-client-main/list_feb20_markets.py
           Herbie-main/mc_helpers.py  (parse_slug)
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np
import requests
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from db.schema import DbClient


# ---------------------------------------------------------------------------
# Slug parser  ->  (lo_f, hi_f) in degrees F
# ---------------------------------------------------------------------------

def parse_slug(slug: str) -> tuple[float, float]:
    """Parse a Polymarket market slug into a temperature bucket (lo_f, hi_f).

    Patterns handled (case-insensitive):
    - ``70f-or-higher``  / ``70forhigher``  -> (70, +inf)
    - ``be-70f-or-higher``                  -> (70, +inf)
    - ``60forbelow``                        -> (-inf, 60)
    - ``60-70f``  (range)                   -> (60, 70)

    Returns (nan, nan) if no pattern matches.
    """
    s = slug.lower()

    # "X-or-higher" / "Xforhigher"
    m = re.search(r"(\d+)f(?:orhigher|or-higher)", s)
    if not m:
        m = re.search(r"(\d+)f-or-higher", s)
    if not m:
        m = re.search(r"be-(\d+)f-or-higher", s)
    if m:
        return (float(m.group(1)), float("inf"))

    # "Xforbelow"
    m = re.search(r"(\d+)forbelow", s)
    if m:
        return (float("-inf"), float(m.group(1)))

    # "X-Yf"  (range)
    m = re.search(r"(\d+)-(\d+)f", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# EventDiscovery
# ---------------------------------------------------------------------------

class EventDiscovery:
    """Discover weather temperature events from the Polymarket Gamma API.

    Workflow
    --------
    1. GET ``<gamma_api_url>/events`` with optional ``slug_pattern`` filter.
    2. For each event whose slug matches the configured pattern, iterate its
       markets and parse each market slug into (lo_f, hi_f).
    3. Upsert events + markets into the local DB.
    4. Return the list of newly-discovered event dicts.

    Parameters
    ----------
    db:
        DbClient instance used for persistence.
    settings:
        Global Settings (provides gamma_api_url and station configs).
    slug_pattern:
        Substring that event slugs must contain to be considered weather
        temperature events.  Defaults to ``"temperature"`` when None is
        passed.
    """

    DEFAULT_SLUG_PATTERN = "temperature"
    _EVENTS_ENDPOINT     = "/events"

    def __init__(
        self,
        db: "DbClient",
        settings: "Settings",
        slug_pattern: str | None = None,
    ) -> None:
        self._db             = db
        self._settings       = settings
        self._slug_pattern   = (slug_pattern or self.DEFAULT_SLUG_PATTERN).lower()
        self._gamma_api_url  = settings.gamma_api_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self, params: dict | None = None) -> list[dict]:
        """Fetch events from Gamma API, upsert into DB, and return results.

        Parameters
        ----------
        params:
            Optional extra query parameters passed to the Gamma API.
            Typically ``{"active": "true"}`` to limit to live events.

        Returns
        -------
        List of event dicts that were discovered (new or updated).
        """
        events = self._fetch_events(params or {})
        discovered: list[dict] = []
        logger.debug("EventDiscovery: {} events from Gamma API", len(events))

        for ev in events:
            slug = (ev.get("slug") or "").lower()
            if self._slug_pattern not in slug:
                continue

            condition_id   = ev.get("conditionId") or ev.get("id") or slug
            question       = ev.get("title") or ev.get("question") or slug
            event_slug     = ev.get("slug") or ""
            event_end_date = ev.get("endDate") or ev.get("end_date_iso") or ""
            station_id     = self._infer_station(slug)

            self._db.upsert_event(
                condition_id   = condition_id,
                question       = question,
                event_slug     = event_slug,
                event_end_date = event_end_date,
                station_id     = station_id,
                status         = "discovered",
            )

            markets_parsed = self._process_markets(
                ev.get("markets", []), condition_id
            )
            logger.info(
                "Discovered event: {} end={} station={} markets={}",
                condition_id, event_end_date, station_id, len(markets_parsed),
            )

            discovered.append({
                "condition_id":   condition_id,
                "event_slug":     event_slug,
                "event_end_date": event_end_date,
                "station_id":     station_id,
                "markets":        markets_parsed,
            })

        return discovered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_events(self, extra_params: dict) -> list[dict]:
        url = self._gamma_api_url + self._EVENTS_ENDPOINT
        try:
            resp = requests.get(url, params=extra_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Gamma API request failed: {exc}"
            ) from exc

    def _process_markets(
        self, markets: list[dict], condition_id: str
    ) -> list[dict]:
        """Parse each market's slug and upsert into DB."""
        parsed: list[dict] = []
        for mkt in markets:
            market_slug = mkt.get("slug") or mkt.get("market_slug") or ""
            lo_f, hi_f  = parse_slug(market_slug)

            # clobTokenIds is typically a JSON-encoded list: '["tok1","tok2"]'
            token_ids_raw = mkt.get("clobTokenIds") or "[]"
            if isinstance(token_ids_raw, str):
                try:
                    token_ids = json.loads(token_ids_raw)
                except json.JSONDecodeError:
                    token_ids = []
            else:
                token_ids = list(token_ids_raw)

            outcomes_raw = mkt.get("outcomes") or '["Yes","No"]'
            if isinstance(outcomes_raw, str):
                try:
                    outcomes = json.loads(outcomes_raw)
                except json.JSONDecodeError:
                    outcomes = ["Yes", "No"]
            else:
                outcomes = list(outcomes_raw)

            for tok, outcome in zip(token_ids, outcomes):
                self._db.upsert_market(
                    token_id     = str(tok),
                    condition_id = condition_id,
                    market_slug  = market_slug,
                    bucket_lo_f  = None if (lo_f != lo_f) else lo_f,
                    bucket_hi_f  = None if (hi_f != hi_f) else hi_f,
                    outcome      = str(outcome),
                )

            parsed.append({
                "market_slug": market_slug,
                "lo_f":        lo_f,
                "hi_f":        hi_f,
                "token_ids":   token_ids,
                "outcomes":    outcomes,
            })

        return parsed

    def _infer_station(self, slug: str) -> str:
        """Attempt to extract a station ID from the event slug.

        Falls back to the first configured station if no match found.
        """
        from config.stations import STATIONS

        slug_up = slug.upper()
        for station_id in STATIONS:
            if station_id.upper() in slug_up:
                return station_id

        # Default to first station (KLGA)
        if STATIONS:
            return next(iter(STATIONS))
        return "KLGA"
