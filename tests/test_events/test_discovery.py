"""Tests for events/discovery.py -- EventDiscovery and parse_slug."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from db.schema import DbClient
from events.discovery import EventDiscovery, parse_slug


# -----------------------------------------------------------------------
# parse_slug
# -----------------------------------------------------------------------

@pytest.mark.parametrize("slug, expected", [
    # open high end -- or-higher variants
    ("highest-temperature-in-nyc-70f-or-higher", (70.0, float("inf"))),
    ("klga-high-75forhigher", (75.0, float("inf"))),
    ("be-80f-or-higher-daily-high", (80.0, float("inf"))),
    # open high end -- or-above variants
    ("klga-high-70f-or-above", (70.0, float("inf"))),
    ("klga-high-70forabove", (70.0, float("inf"))),
    ("be-70f-or-above-daily-high", (70.0, float("inf"))),
    # open low end -- or-below variants
    ("klga-high-55forbelow", (float("-inf"), 55.0)),
    ("klga-high-55f-or-below", (float("-inf"), 55.0)),
    ("be-55f-or-below", (float("-inf"), 55.0)),
    # range
    ("high-temperature-60-70f-nyc", (60.0, 70.0)),
    ("klga-daily-high-46-47f", (46.0, 47.0)),
])
def test_parse_slug(slug, expected):
    result = parse_slug(slug)
    assert result is not None
    lo, hi = result
    exp_lo, exp_hi = expected
    assert lo == exp_lo
    assert hi == exp_hi


def test_parse_slug_no_match_returns_none():
    assert parse_slug("something-unrelated") is None


def test_parse_slug_no_match_logs_warning(capfd):
    import io
    from loguru import logger
    sink = io.StringIO()
    sink_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        parse_slug("something-unrelated")
    finally:
        logger.remove(sink_id)
    assert "parse_slug" in sink.getvalue() or "no pattern" in sink.getvalue().lower()


def test_parse_slug_case_insensitive():
    result = parse_slug("KLGA-HIGH-50-60F")
    assert result is not None
    lo, hi = result
    assert lo == 50.0
    assert hi == 60.0


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def mem_db():
    db = DbClient(":memory:")
    # Pre-seed station so FK on events.station_id passes
    db.upsert_station("KLGA", 40.7769, -73.874, "America/New_York")
    yield db
    db.close()


def _make_settings(gamma_url: str = "https://gamma-api.polymarket.com"):
    settings             = MagicMock()
    settings.gamma_api_url = gamma_url
    return settings


def _gamma_response(slug_pattern: str = "highest-temperature") -> list[dict]:
    """Minimal Gamma API event list response."""
    return [
        {
            "id":            "cond-001",
            "conditionId":   "cond-001",
            "slug":          f"highest-temperature-in-nyc-2025-03-20",
            "title":         "Highest Temperature in NYC on 2025-03-20",
            "endDate":       "2025-03-20",
            "markets": [
                {
                    "slug":          "highest-temperature-in-nyc-2025-03-20-60-70f",
                    "conditionId":   "cond-001",
                    "clobTokenIds":  json.dumps(["tok-yes-1", "tok-no-1"]),
                    "outcomes":      json.dumps(["Yes", "No"]),
                },
                {
                    "slug":          "highest-temperature-in-nyc-2025-03-20-70f-or-higher",
                    "conditionId":   "cond-001",
                    "clobTokenIds":  json.dumps(["tok-yes-2", "tok-no-2"]),
                    "outcomes":      json.dumps(["Yes", "No"]),
                },
            ],
        },
    ]


# -----------------------------------------------------------------------
# EventDiscovery.discover
# -----------------------------------------------------------------------

def test_discover_returns_list(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    assert isinstance(result, list)
    assert len(result) == 1


def test_discover_upserts_event(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()

    events = mem_db.get_active_events()
    assert len(events) == 1
    assert events[0]["condition_id"] == "cond-001"


def test_discover_upserts_markets(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()

    cur  = mem_db._conn.execute("SELECT * FROM markets").fetchall()
    # 2 markets x 2 tokens each = 4 rows
    assert len(cur) == 4


def test_discover_parses_buckets(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    markets = result[0]["markets"]
    slugs_to_buckets = {m["market_slug"]: (m["lo_f"], m["hi_f"]) for m in markets}

    key60 = "highest-temperature-in-nyc-2025-03-20-60-70f"
    key70 = "highest-temperature-in-nyc-2025-03-20-70f-or-higher"

    assert key60 in slugs_to_buckets
    assert slugs_to_buckets[key60] == (60.0, 70.0)

    assert key70 in slugs_to_buckets
    assert slugs_to_buckets[key70] == (70.0, float("inf"))


def test_discover_skips_non_matching_event(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    # Event slug does NOT contain "temperature"
    response = [{"slug": "will-it-rain-in-nyc", "id": "x", "markets": []}]

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    assert result == []


def test_discover_empty_response(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)

    with patch.object(disc, "_fetch_events", return_value=[]):
        result = disc.discover()

    assert result == []


def test_discover_infers_station_klga(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    # Slug does not contain "KLGA" so it defaults to first station
    assert result[0]["station_id"] == "KLGA"


def test_discover_fetch_events_raises_on_http_error(mem_db):
    import requests

    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)

    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(
                side_effect=requests.HTTPError("503")
            )
        )
        with pytest.raises(RuntimeError, match="Gamma API"):
            disc.discover()


def test_discover_idempotent(mem_db):
    """Calling discover twice should not create duplicate event rows."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()
        disc.discover()

    events = mem_db.get_active_events()
    assert len(events) == 1


def test_discover_result_has_required_keys(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    ev = result[0]
    for key in ("condition_id", "event_slug", "event_end_date",
                "station_id", "markets"):
        assert key in ev, f"Missing key: {key}"
