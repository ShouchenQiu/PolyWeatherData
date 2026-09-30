"""Polymarket market-slug parser: slug -> temperature bucket (lo, hi).

Extracted from the original ``events/discovery.py`` (only the parsing logic
that the data-acquisition path needs).  The returned bounds are in the unit
used by the slug (Fahrenheit or Celsius) and the upper bound is exclusive.

Slug semantics:
    parse_slug() requires BOTH a station token AND a temperature-range token.
    A slug matching only one component is treated as unrecognized and returns
    ``None``.
"""
from __future__ import annotations

import re

from loguru import logger


# ---------------------------------------------------------------------------
# Pattern handlers -> (lo, hi) in the slug's own unit (whole degrees)
# ---------------------------------------------------------------------------

def _hi_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float("inf"))


def _lo_bound(m: re.Match) -> tuple[float, float]:
    return (float("-inf"), float(m.group(1)) + 1.0)


def _range_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float(m.group(2)) + 1.0)


def _c_hi_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float("inf"))


def _c_lo_bound(m: re.Match) -> tuple[float, float]:
    return (float("-inf"), float(m.group(1)) + 1.0)


def _c_single_bound(m: re.Match) -> tuple[float, float]:
    return (float(m.group(1)), float(m.group(1)) + 1.0)


# Each entry: (compiled pattern, handler).  First match wins.
_SLUG_REGISTRY = [
    # Xf-or-higher / Xforhigher / be-Xf-or-higher / Xf-or-above
    (
        re.compile(r"(?:be-)?(?:\d+-)?(\d+)f(?:-or-higher|-or-above|orhigher|orabove)"),
        _hi_bound,
    ),
    # Xf-or-below / Xforbelow / Xf-or-lower
    (
        re.compile(r"(?:be-)?(\d+)f(?:-or-below|orbelow|-or-lower|orlower)"),
        _lo_bound,
    ),
    # Legacy NYC compact range: between-6263f
    (
        re.compile(r"between-(\d{2})(\d{2})f"),
        _range_bound,
    ),
    # Compact two-digit endpoints without prefix: be-6667f
    (
        re.compile(r"(?:be-)(\d{2})(\d{2})f"),
        _range_bound,
    ),
    # Seattle/Atlanta compact range: between-4849-f
    (
        re.compile(r"between-(\d{2})(\d{2})-f"),
        _range_bound,
    ),
    # Seattle/Atlanta hyphenated single: be-47-f-or-below / be-58-f-or-higher
    (
        re.compile(r"(?:be-)?(\d+)-f(?:-or-higher|-or-above|-or-below|-or-lower)"),
        lambda m: (
            _hi_bound(m)
            if "-or-higher" in m.group(0) or "-or-above" in m.group(0)
            else _lo_bound(m)
        ),
    ),
    # Celsius: Xc / Xcorbelow / Xcorhigher
    (
        re.compile(r"(?:be-)?(\d+)c(?:-or-higher|-or-above|orhigher|orabove)"),
        _c_hi_bound,
    ),
    (
        re.compile(r"(?:be-)?(\d+)c(?:-or-below|orbelow|-or-lower|orlower)"),
        _c_lo_bound,
    ),
    (
        re.compile(r"(?:be-)?(\d+)c\b"),
        _c_single_bound,
    ),
    # Hyphenated Celsius (Seoul/Toronto): be-3-c-or-below / be-4-c
    (
        re.compile(r"(?:be-)?(\d+)-c(?:-or-higher|-or-above)"),
        _c_hi_bound,
    ),
    (
        re.compile(r"(?:be-)?(\d+)-c(?:-or-below|-or-lower)"),
        _c_lo_bound,
    ),
    (
        re.compile(r"(?:be-)?(\d+)-c\b"),
        _c_single_bound,
    ),
    # X-Yf (range)
    (
        re.compile(r"(\d+)-(\d+)f"),
        _range_bound,
    ),
    # between-Xf-and-Yf (London Jan-26 variant)
    (
        re.compile(r"between-(\d+)f-and-(\d+)f"),
        _range_bound,
    ),
]


def parse_slug(
    slug: str, *, require_station: bool = True
) -> tuple[float, float] | None:
    """Parse a Polymarket market slug into a temperature bucket ``(lo, hi)``.

    Patterns handled (case-insensitive):

    - ``70f-or-higher`` / ``70forhigher`` / ``be-70f-or-higher`` -> (70, +inf)
    - ``70f-or-above``  / ``70forabove``  / ``be-70f-or-above``  -> (70, +inf)
    - ``55f-or-below``  / ``55forbelow``  / ``be-55f-or-below``  -> (-inf, 56)
    - ``60-70f``  (inclusive whole-degree range)                 -> (60, 71)

    The returned upper bound is exclusive.  Polymarket resolves these markets
    from whole-degree observations, so inclusive labels gain one degree at the
    upper edge.

    Returns ``None`` unless the slug also contains a recognizable station
    token (ICAO identifier or alias from ``config.stations.STATIONS``) when
    ``require_station`` is True.  Pass ``require_station=False`` for
    multi-city ingestion where the station is not registered locally.
    """
    from config.stations import STATIONS

    s = slug.lower()

    if require_station:
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
