"""Polymarket event discovery via the Gamma API (V2V01).

Fetches active weather-temperature events using the PublicClient SDK
paginator (ADR-0009), parses their bucket slugs into (lo_f, hi_f) pairs,
and upserts them into the local database.

Slug semantics (V2V01):
    parse_slug() requires BOTH a station token AND a temperature-range
    token in the slug.  A slug matching only one component is treated as
    unrecognized and persisted with discovery_status='unrecognized_slug'.

Station inference (V2V01):
    _infer_station() uses the slug_aliases list on each StationConfig.
    Raises ValueError if no alias matches.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from core.alerts import AlertBus
    from db.schema import DbClient


# ---------------------------------------------------------------------------
# Slug parser registry  ->  (lo_f, hi_f) in degrees F
# ---------------------------------------------------------------------------

def _hi_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float("inf"))


def _lo_bound(m: re.Match) -> tuple[float, float]:
    return (float("-inf"), float(m.group(1)))


def _range_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float(m.group(2)))


# Each entry: (compiled pattern, handler).  First match wins.
# To add a new slug format, append one tuple here -- no other change needed.
_SLUG_REGISTRY = [
    # Xf-or-higher / Xforhigher / be-Xf-or-higher / Xf-or-above / be-Xf-or-above
    (
        re.compile(r"(?:be-)?(?:\d+-)?(\d+)f(?:-or-higher|-or-above|orhigher|orabove)"),
        _hi_bound,
    ),
    # Xf-or-below / Xforbelow / be-Xf-or-below
    (
        re.compile(r"(?:be-)?(\d+)f(?:-or-below|orbelow)"),
        _lo_bound,
    ),
    # X-Yf  (range)
    (
        re.compile(r"(\d+)-(\d+)f"),
        _range_bound,
    ),
]


def parse_slug(slug: str) -> tuple[float, float] | None:
    """Parse a Polymarket market slug into a temperature bucket (lo_f, hi_f).

    Patterns handled (case-insensitive):
    - ``70f-or-higher`` / ``70forhigher`` / ``be-70f-or-higher`` -> (70, +inf)
    - ``70f-or-above``  / ``70forabove``  / ``be-70f-or-above``  -> (70, +inf)
    - ``55f-or-below``  / ``55forbelow``  / ``be-55f-or-below``  -> (-inf, 55)
    - ``60-70f``  (range)                                        -> (60, 70)

    V2V01: returns None unless the slug also contains a recognizable station
    token (ICAO identifier or alias from config.stations.STATIONS).  Slugs
    matching only the temperature pattern are treated as unrecognized.

    Returns None and logs a WARNING if no pattern matches or no station found.
    """
    from config.stations import STATIONS

    s = slug.lower()

    # Check for at least one station alias in the slug
    has_station = any(
        alias in s
        for st in STATIONS.values()
        for alias in (st.slug_aliases or [st.station_id.lower()])
    )
    if not has_station:
        logger.warning("parse_slug: no station token found in slug {!r}", slug)
        return None

    for pattern, handler in _SLUG_REGISTRY:
        m = pattern.search(s)
        if m:
            return handler(m)
    logger.warning("parse_slug: no temperature pattern matched slug {!r}", slug)
    return None


# ---------------------------------------------------------------------------
# EventDiscovery
# ---------------------------------------------------------------------------

class EventDiscovery:
    """Discover weather temperature events from the Polymarket Gamma API.

    Workflow (V2V01)
    ----------------
    1. Call ``PublicClient.list_events(live=True, tag_slug="weather")`` via the
       SDK paginator (ADR-0009).  Falls back to raw ``requests.get`` when the
       SDK is not available (tests / offline mode).
    2. For each event whose slug matches the configured pattern, iterate its
       markets and parse each market slug into (lo_f, hi_f).
    3. Markets with unparseable slugs are upserted with
       ``discovery_status="unrecognized_slug"`` and a ``discovery_warn`` alert
       is emitted.
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
    alert_bus:
        Optional AlertBus for emitting ``discovery_warn`` events.
    """

    DEFAULT_SLUG_PATTERN = "temperature"
    _EVENTS_ENDPOINT     = "/events"

    def __init__(
        self,
        db: "DbClient",
        settings: "Settings",
        slug_pattern: str | None = None,
        alert_bus: "AlertBus | None" = None,
    ) -> None:
        self._db             = db
        self._settings       = settings
        self._slug_pattern   = (slug_pattern or self.DEFAULT_SLUG_PATTERN).lower()
        self._gamma_api_url  = settings.gamma_api_url.rstrip("/")
        self._alert_bus      = alert_bus

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
        List of event dicts that were newly discovered (not previously stored).
        """
        events = self._fetch_events(params or {})
        discovered: list[dict] = []
        n_updated    = 0
        n_skipped    = 0
        n_unrecog    = 0
        logger.debug("EventDiscovery: {} events from Gamma API", len(events))

        for ev in events:
            slug = (ev.get("slug") or "").lower()
            if self._slug_pattern not in slug:
                continue

            condition_id = ev.get("conditionId") or ev.get("id") or slug

            # Dedup: event metadata already persisted -- only sync missing markets.
            if self._db.event_exists(condition_id):
                added, unrecog_new = self._sync_missing_markets(
                    ev.get("markets", []), condition_id
                )
                n_unrecog += unrecog_new
                if added:
                    n_updated += 1
                    logger.info(
                        "EventDiscovery: synced {} new market(s) for existing event {}",
                        len(added), condition_id,
                    )
                else:
                    n_skipped += 1
                    logger.debug(
                        "EventDiscovery: no new markets for existing event {}",
                        condition_id,
                    )
                continue

            question       = ev.get("title") or ev.get("question") or slug
            event_slug     = ev.get("slug") or ""
            event_end_date = ev.get("endDate") or ev.get("end_date_iso") or ""
            try:
                station_id = self._infer_station(slug)
            except ValueError as exc:
                logger.error("EventDiscovery: {}", exc)
                n_skipped += 1
                continue

            self._db.upsert_event(
                condition_id   = condition_id,
                question       = question,
                event_slug     = event_slug,
                event_end_date = event_end_date,
                station_id     = station_id,
                status         = "discovered",
            )

            markets_parsed, unrecog_count = self._process_markets(
                ev.get("markets", []), condition_id
            )
            n_unrecog += unrecog_count
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

        logger.info(
            "EventDiscovery poll complete: new={} updated={} skipped={} unrecognized_slug={}",
            len(discovered), n_updated, n_skipped, n_unrecog,
        )
        return discovered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_events(self, extra_params: dict) -> list[dict]:
        """Fetch events from the Gamma API.

        Uses ``PublicClient.list_events(live=True, tag_slug="weather")`` when
        the SDK is available (ADR-0009).  Falls back to a raw ``requests.get``
        call when the SDK is not installed (offline / test mode).
        """
        try:
            from polymarket_client import PublicClient  # type: ignore[import-untyped]
            client = PublicClient()
            return list(client.list_events(live=True, tag_slug="weather"))
        except ImportError:
            pass

        import requests
        url = self._gamma_api_url + self._EVENTS_ENDPOINT
        try:
            resp = requests.get(url, params=extra_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Gamma API request failed: {exc}"
            ) from exc

    def _sync_missing_markets(
        self, markets: list[dict], condition_id: str
    ) -> tuple[list[dict], int]:
        """Upsert markets whose token_id is not yet in the DB.

        Called when the parent event already exists.  Only adds genuinely
        new tokens; existing token rows are not touched.

        Applies the same validation as _process_markets (V2V01):
        - token/outcome length mismatch -> log ERROR + skip
        - unrecognized slug -> upsert with discovery_status='unrecognized_slug'
          + emit discovery_warn alert

        Returns
        -------
        (added, unrecog_count)
        """
        added: list[dict] = []
        unrecog_count = 0

        for mkt in markets:
            market_slug = mkt.get("slug") or mkt.get("market_slug") or ""

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

            # V2V01: length parity check -- same rule as _process_markets
            if len(token_ids) != len(outcomes):
                logger.error(
                    "EventDiscovery: token/outcome length mismatch for market {} "
                    "(tokens={}, outcomes={}); skipping",
                    market_slug, len(token_ids), len(outcomes),
                )
                continue

            result           = parse_slug(market_slug)
            lo_f, hi_f       = result if result is not None else (None, None)
            discovery_status = "parsed" if result is not None else "unrecognized_slug"

            if result is None:
                unrecog_count += 1
                self._emit_discovery_warn(market_slug, condition_id)

            for token_id, outcome in zip(token_ids, outcomes):
                if not token_id:
                    continue
                if self._db.market_exists(token_id):
                    continue
                self._db.upsert_market(
                    token_id         = str(token_id),
                    condition_id     = condition_id,
                    market_slug      = market_slug,
                    bucket_lo_f      = lo_f,
                    bucket_hi_f      = hi_f,
                    outcome          = str(outcome),
                    discovery_status = discovery_status,
                )
                added.append({"token_id": str(token_id), "outcome": str(outcome)})

        return added, unrecog_count

    def _process_markets(
        self, markets: list[dict], condition_id: str
    ) -> tuple[list[dict], int]:
        """Parse each market slug and upsert into DB.

        Returns
        -------
        (parsed_list, unrecog_count) where unrecog_count is the number of
        markets that had unparseable slugs.
        """
        parsed: list[dict] = []
        unrecog_count = 0

        for mkt in markets:
            market_slug = mkt.get("slug") or mkt.get("market_slug") or ""

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

            # V2V01: assert length parity; skip without truncation on mismatch
            if len(token_ids) != len(outcomes):
                logger.error(
                    "EventDiscovery: token/outcome length mismatch for market {} "
                    "(tokens={}, outcomes={}); skipping",
                    market_slug, len(token_ids), len(outcomes),
                )
                continue

            result     = parse_slug(market_slug)
            lo_f, hi_f = result if result is not None else (None, None)

            if result is None:
                # Unrecognized slug: persist with discovery_status flag + alert
                discovery_status = "unrecognized_slug"
                unrecog_count += 1
                self._emit_discovery_warn(market_slug, condition_id)
            else:
                discovery_status = "parsed"

            for tok, outcome in zip(token_ids, outcomes):
                self._db.upsert_market(
                    token_id         = str(tok),
                    condition_id     = condition_id,
                    market_slug      = market_slug,
                    bucket_lo_f      = lo_f,
                    bucket_hi_f      = hi_f,
                    outcome          = str(outcome),
                    discovery_status = discovery_status,
                )

            parsed.append({
                "market_slug":       market_slug,
                "lo_f":              lo_f,
                "hi_f":              hi_f,
                "token_ids":         token_ids,
                "outcomes":          outcomes,
                "discovery_status":  discovery_status,
            })

        return parsed, unrecog_count

    def _emit_discovery_warn(self, market_slug: str, condition_id: str) -> None:
        """Emit a discovery_warn AlertEvent when a slug is unrecognized."""
        if self._alert_bus is None:
            return
        try:
            from core.alerts import AlertEvent
            self._alert_bus.emit_nowait(AlertEvent(
                alert_type  = "discovery_warn",
                station     = None,
                message     = f"Unrecognized slug: {market_slug!r} (condition: {condition_id})",
                occurred_at = datetime.now(timezone.utc),
                context     = {"slug": market_slug, "condition_id": condition_id},
            ))
        except Exception:
            logger.exception("EventDiscovery: failed to emit discovery_warn alert")

    def _infer_station(self, slug: str) -> str:
        """Return the station_id that owns *slug* based on slug_aliases.

        Raises ValueError if no station alias matches the slug.
        """
        from config.stations import STATIONS
        s = slug.lower()
        for st in STATIONS.values():
            aliases = st.slug_aliases or [st.station_id.lower()]
            for alias in aliases:
                if alias.lower() in s:
                    return st.station_id
        raise ValueError(
            f"no station alias matched slug {slug!r}"
        )
