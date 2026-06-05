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
    ("klga-highest-temperature-70f-or-higher", (70.0, float("inf"))),
    ("klga-high-75forhigher", (75.0, float("inf"))),
    ("klga-be-80f-or-higher-daily-high", (80.0, float("inf"))),
    # open high end -- or-above variants
    ("klga-high-70f-or-above", (70.0, float("inf"))),
    ("klga-high-70forabove", (70.0, float("inf"))),
    ("klga-be-70f-or-above-daily-high", (70.0, float("inf"))),
    # open low end -- or-below variants
    ("klga-high-55forbelow", (float("-inf"), 55.0)),
    ("klga-high-55f-or-below", (float("-inf"), 55.0)),
    ("klga-be-55f-or-below", (float("-inf"), 55.0)),
    # range
    ("klga-high-temperature-60-70f-nyc", (60.0, 70.0)),
    ("klga-daily-high-46-47f", (46.0, 47.0)),
    # Current NYC weather event format resolves against KLGA.
    ("highest-temperature-in-nyc-on-june-5-2026-81forbelow", (float("-inf"), 81.0)),
    ("highest-temperature-in-nyc-on-june-5-2026-82-83f", (82.0, 83.0)),
    ("highest-temperature-in-nyc-on-june-5-2026-100forhigher", (100.0, float("inf"))),
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


def test_parse_slug_no_station_returns_none():
    """Slug without a station alias is treated as unrecognized per V2V01."""
    assert parse_slug("highest-temperature-70f-or-higher") is None


def test_parse_slug_no_temp_pattern_returns_none():
    """Slug with only station but no temperature range is unrecognized per V2V01."""
    assert parse_slug("klga-daily-weather-observation") is None


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
            "slug":          "highest-temperature-klga-nyc-2025-03-20",
            "title":         "Highest Temperature at KLGA on 2025-03-20",
            "endDate":       "2025-03-20",
            "markets": [
                {
                    "id":            "mkt-60-70",
                    "slug":          "klga-highest-temperature-60-70f-2025-03-20",
                    "conditionId":   "cond-001",
                    "clobTokenIds":  json.dumps(["tok-yes-1", "tok-no-1"]),
                    "outcomes":      json.dumps(["Yes", "No"]),
                },
                {
                    "id":            "mkt-70-plus",
                    "slug":          "klga-highest-temperature-70f-or-higher-2025-03-20",
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

    # Use event_exists -- discover() inserts with status='discovered', not 'active'
    assert mem_db.event_exists("cond-001")


def test_discover_upserts_markets(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()

    cur  = mem_db._conn.execute("SELECT * FROM markets").fetchall()
    # 2 markets x 2 tokens each = 4 rows
    assert len(cur) == 4


def test_discover_persists_market_id(mem_db):
    """Market id is retained for SDK redeem_positions(market_id=...)."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)

    with patch.object(disc, "_fetch_events", return_value=_gamma_response()):
        disc.discover()

    row = mem_db._conn.execute(
        "SELECT market_id FROM markets WHERE token_id='tok-yes-1'"
    ).fetchone()
    assert row is not None
    assert row["market_id"] == "mkt-60-70"


def test_discover_parses_buckets(mem_db):
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover()

    markets = result[0]["markets"]
    slugs_to_buckets = {m["market_slug"]: (m["lo_f"], m["hi_f"]) for m in markets}

    key60 = "klga-highest-temperature-60-70f-2025-03-20"
    key70 = "klga-highest-temperature-70f-or-higher-2025-03-20"

    assert key60 in slugs_to_buckets
    assert slugs_to_buckets[key60] == (60.0, 70.0)

    assert key70 in slugs_to_buckets
    assert slugs_to_buckets[key70] == (70.0, float("inf"))


def test_discover_parses_current_nyc_temperature_event(mem_db):
    """Polymarket NYC temperature events resolve against KLGA."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = [
        {
            "id":          "553899",
            "conditionId": "553899",
            "slug":        "highest-temperature-in-nyc-on-june-5-2026",
            "title":       "Highest temperature in NYC on June 5?",
            "endDate":     "2026-06-05T12:00:00Z",
            "markets": [
                {
                    "id":           "mkt-81",
                    "slug":         "highest-temperature-in-nyc-on-june-5-2026-81forbelow",
                    "conditionId":  "cond-81",
                    "clobTokenIds": json.dumps(["tok-81-yes", "tok-81-no"]),
                    "outcomes":     json.dumps(["Yes", "No"]),
                },
                {
                    "id":           "mkt-100",
                    "slug":         "highest-temperature-in-nyc-on-june-5-2026-100forhigher",
                    "conditionId":  "cond-100",
                    "clobTokenIds": json.dumps(["tok-100-yes", "tok-100-no"]),
                    "outcomes":     json.dumps(["Yes", "No"]),
                },
            ],
        }
    ]

    with patch.object(disc, "_fetch_events", return_value=response):
        result = disc.discover({"active": "true"})

    assert result[0]["station_id"] == "KLGA"
    slugs_to_buckets = {
        m["market_slug"]: (m["lo_f"], m["hi_f"])
        for m in result[0]["markets"]
    }
    assert slugs_to_buckets[
        "highest-temperature-in-nyc-on-june-5-2026-81forbelow"
    ] == (float("-inf"), 81.0)
    assert slugs_to_buckets[
        "highest-temperature-in-nyc-on-june-5-2026-100forhigher"
    ] == (100.0, float("inf"))


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


def test_sdk_list_events_params_passes_slug_and_page_size():
    params = EventDiscovery._sdk_list_events_params({
        "slug": "highest-temperature-in-nyc-on-june-5-2026",
        "page_size": "20",
    })

    assert params == {
        "slug": "highest-temperature-in-nyc-on-june-5-2026",
        "page_size": 20,
    }


def test_sdk_list_events_params_maps_active_to_closed_false():
    params = EventDiscovery._sdk_list_events_params({"active": "true"})

    assert params["closed"] is False
    assert params["page_size"] == 100


def test_discover_idempotent(mem_db):
    """Calling discover twice should not create duplicate event rows."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()
        disc.discover()

    # Exactly one event row regardless of how many times discover() ran
    count = mem_db._conn.execute(
        "SELECT COUNT(*) FROM events WHERE condition_id='cond-001'"
    ).fetchone()[0]
    assert count == 1


def test_discover_dedup_second_call_returns_empty(mem_db):
    """Second discover() with the same event_id returns [] -- no re-discovery."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    with patch.object(disc, "_fetch_events", return_value=response):
        first  = disc.discover()
        second = disc.discover()

    assert len(first) == 1
    assert second == []


def test_discover_dedup_skips_upsert_on_second_call(mem_db):
    """DB write count for upsert_event must be exactly 1 across two calls."""
    from unittest.mock import call
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    response = _gamma_response()

    original_upsert = mem_db.upsert_event
    upsert_calls: list = []

    def _counting_upsert(*args, **kwargs):
        upsert_calls.append((args, kwargs))
        return original_upsert(*args, **kwargs)

    mem_db.upsert_event = _counting_upsert

    with patch.object(disc, "_fetch_events", return_value=response):
        disc.discover()
        disc.discover()

    # upsert_event must have been called exactly once (first call only)
    assert len(upsert_calls) == 1


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


# ---------------------------------------------------------------------------
# V2V01 -- _infer_station: alias lookup
# ---------------------------------------------------------------------------

def _make_settings_with_stations(gamma_url: str = "https://gamma-api.polymarket.com"):
    """Settings mock that forwards real STATIONS for alias tests."""
    settings = MagicMock()
    settings.gamma_api_url = gamma_url
    return settings


def test_infer_station_matches_alias(mem_db):
    """Slug containing an alias returns the matching station_id."""
    disc = EventDiscovery(db=mem_db, settings=_make_settings())
    # KLGA has alias 'klga'
    station = disc._infer_station("will-the-high-temp-at-klga-exceed-75f")
    assert station == "KLGA"


def test_infer_station_matches_alias_case_insensitive(mem_db):
    """Alias matching is case-insensitive."""
    disc = EventDiscovery(db=mem_db, settings=_make_settings())
    station = disc._infer_station("KLGA-DAILY-HIGH-75-80F")
    assert station == "KLGA"


def test_infer_station_matches_laguardia_alias(mem_db):
    """'laguardia' alias maps to KLGA."""
    disc = EventDiscovery(db=mem_db, settings=_make_settings())
    station = disc._infer_station("highest-temp-at-laguardia-2026-06-01")
    assert station == "KLGA"


def test_infer_station_matches_nyc_alias(mem_db):
    """Polymarket uses NYC in event slugs for KLGA weather markets."""
    disc = EventDiscovery(db=mem_db, settings=_make_settings())
    station = disc._infer_station("highest-temperature-in-nyc-on-june-5-2026")
    assert station == "KLGA"


def test_infer_station_raises_on_no_match(mem_db):
    """ValueError raised when no alias matches the slug."""
    disc = EventDiscovery(db=mem_db, settings=_make_settings())
    with pytest.raises(ValueError, match="station"):
        disc._infer_station("highest-temp-at-chicago-ohare-2026-06-01")


# ---------------------------------------------------------------------------
# V2V01 -- _process_markets: token/outcome length check
# ---------------------------------------------------------------------------

def test_process_markets_skips_mismatched_length(mem_db):
    """Market with len(tokens) != len(outcomes) is skipped without DB write."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)

    bad_market = {
        "slug":         "will-the-high-temp-at-klga-60-70f",
        "conditionId":  "cid-bad",
        "clobTokenIds": json.dumps(["tok-yes-only"]),  # 1 token
        "outcomes":     json.dumps(["Yes", "No"]),     # 2 outcomes
    }
    result, unrecog = disc._process_markets([bad_market], "cid-bad")
    assert result == []  # market skipped
    assert unrecog == 0
    rows = mem_db._conn.execute("SELECT * FROM markets").fetchall()
    assert len(rows) == 0


def test_process_markets_upserts_matched_market(mem_db):
    """Market where len(tokens) == len(outcomes) is upserted normally."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    mem_db.upsert_event("cid-ok", "q", "klga-slug", "2026-06-01", "KLGA")

    good_market = {
        "slug":         "will-the-high-temp-at-klga-60-70f",
        "conditionId":  "cid-ok",
        "clobTokenIds": json.dumps(["tok-yes", "tok-no"]),
        "outcomes":     json.dumps(["Yes", "No"]),
    }
    result, unrecog = disc._process_markets([good_market], "cid-ok")
    assert len(result) == 1
    rows = mem_db._conn.execute("SELECT * FROM markets").fetchall()
    assert len(rows) == 2


def test_event_to_dict_maps_sdk_models_to_gamma_shape():
    """Typed SDK Event/Market models map to downstream Gamma-like dicts."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    ev = SimpleNamespace(
        id="event-1",
        slug="highest-temperature-klga-2026-06-01",
        title="Highest Temperature at KLGA",
        created_at=datetime(2026, 5, 31, tzinfo=timezone.utc),
        updated_at=None,
        schedule=SimpleNamespace(end_date=datetime(2026, 6, 1, tzinfo=timezone.utc)),
        markets=(
            SimpleNamespace(
                id="market-1",
                slug="klga-highest-temperature-60-70f-2026-06-01",
                condition_id="condition-1",
                question="Will KLGA be 60-70F?",
                outcomes=SimpleNamespace(
                    yes=SimpleNamespace(token_id="yes-token"),
                    no=SimpleNamespace(token_id="no-token"),
                ),
            ),
        ),
    )

    out = EventDiscovery._event_to_dict(ev)

    assert out["id"] == "event-1"
    assert out["endDate"].startswith("2026-06-01")
    assert out["markets"][0]["market_id"] == "market-1"
    assert out["markets"][0]["conditionId"] == "condition-1"
    assert json.loads(out["markets"][0]["clobTokenIds"]) == ["yes-token", "no-token"]


# ---------------------------------------------------------------------------
# V2V01 -- unrecognized slug: upsert with discovery_status + discovery_warn
# ---------------------------------------------------------------------------

def test_discover_unrecognized_slug_upserts_market(mem_db):
    """Market with unparseable slug is upserted with discovery_status='unrecognized_slug'."""
    settings = _make_settings()
    disc     = EventDiscovery(db=mem_db, settings=settings)
    disc._alert_bus = MagicMock()

    bad_slug_event = {
        "id":          "cid-unrecog",
        "conditionId": "cid-unrecog",
        "slug":        "highest-temperature-in-nyc-klga-2026-06-01",
        "title":       "Some temperature event",
        "endDate":     "2026-06-01",
        "markets": [
            {
                "slug":         "this-is-not-a-temperature-bucket",
                "conditionId":  "cid-unrecog",
                "clobTokenIds": json.dumps(["tok-a", "tok-b"]),
                "outcomes":     json.dumps(["Yes", "No"]),
            }
        ],
    }
    with patch.object(disc, "_fetch_events", return_value=[bad_slug_event]):
        disc.discover()

    row = mem_db._conn.execute(
        "SELECT discovery_status FROM markets WHERE token_id='tok-a'"
    ).fetchone()
    assert row is not None
    assert row["discovery_status"] == "unrecognized_slug"


def test_discover_unrecognized_slug_emits_alert(mem_db):
    """AlertBus receives a discovery_warn event for unrecognized slugs."""
    from core.alerts import AlertBus, AlertEvent

    settings  = _make_settings()
    disc      = EventDiscovery(db=mem_db, settings=settings)
    bus       = AlertBus()
    received: list[AlertEvent] = []

    import asyncio

    async def handler(e: AlertEvent) -> None:
        received.append(e)

    bus.subscribe(handler)
    disc._alert_bus = bus

    bad_slug_event = {
        "id":          "cid-warn",
        "conditionId": "cid-warn",
        "slug":        "highest-temperature-in-nyc-klga-2026-06-01",
        "title":       "Some temperature event",
        "endDate":     "2026-06-01",
        "markets": [
            {
                "slug":         "completely-unrelated-slug",
                "conditionId":  "cid-warn",
                "clobTokenIds": json.dumps(["tok-x", "tok-y"]),
                "outcomes":     json.dumps(["Yes", "No"]),
            }
        ],
    }
    with patch.object(disc, "_fetch_events", return_value=[bad_slug_event]):
        disc.discover()

    asyncio.run(asyncio.sleep(0))  # flush async handlers
    alert_types = [e.alert_type for e in received]
    assert "discovery_warn" in alert_types


# ---------------------------------------------------------------------------
# V2V01 -- discovery_status hard gate (tradeable/active queries)
# ---------------------------------------------------------------------------

def test_get_tradeable_events_excludes_unrecognized_slug(mem_db):
    """get_tradeable_events returns only markets with discovery_status='parsed'."""
    # Insert an event with an unrecognized-slug market
    mem_db.upsert_event("cid-u", "q", "slug", "2026-06-05", "KLGA", status="active")
    mem_db._conn.execute(
        "INSERT INTO markets (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome, discovery_status)"
        " VALUES ('tok-u', 'cid-u', 'some-slug', NULL, NULL, 'YES', 'unrecognized_slug')"
    )
    mem_db._conn.commit()

    events = mem_db.get_tradeable_events("2026-06-05")
    cids = [e["condition_id"] for e in events]
    assert "cid-u" not in cids


def test_get_markets_for_event_excludes_null_bucket_markets(mem_db):
    """get_markets_for_event only returns parsed markets with non-null buckets."""
    mem_db.upsert_event("cid-m", "q", "slug", "2026-06-05", "KLGA", status="active")
    # good market
    mem_db.upsert_market("tok-good", "cid-m", "60-70f-slug", 60.0, 70.0, "YES")
    # null-bucket market
    mem_db._conn.execute(
        "INSERT INTO markets (token_id, condition_id, market_slug, bucket_lo_f, bucket_hi_f, outcome, discovery_status)"
        " VALUES ('tok-null', 'cid-m', 'unrecog', NULL, NULL, 'YES', 'unrecognized_slug')"
    )
    mem_db._conn.commit()

    markets = mem_db.get_markets_for_event("cid-m")
    token_ids = [m["token_id"] for m in markets]
    assert "tok-good" in token_ids
    assert "tok-null" not in token_ids
