"""Tests for events/lifecycle.py -- EventLifecycle state machine."""
from __future__ import annotations

import pytest
import pandas as pd

from events.lifecycle import (
    EventLifecycle,
    DISCOVERED,
    ACTIVE,
    CUTOFF,
    PENDING_SETTLE,
    SETTLED,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _event(status: str, end_date: str = "2025-03-20") -> dict:
    return {"status": status, "event_end_date": end_date}


def _ts(date: str, hour: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"{date}T{hour:02d}:00:00", tz="UTC")


lc = EventLifecycle()


# -----------------------------------------------------------------------
# discovered -> active
# -----------------------------------------------------------------------

def test_discovered_before_cutoff_becomes_active():
    ev  = _event(DISCOVERED, "2025-03-20")
    now = _ts("2025-03-20", hour=10)   # 10:00 UTC < 21:00 UTC cutoff
    assert lc.advance(ev, now) == ACTIVE


def test_active_before_cutoff_stays_active():
    ev  = _event(ACTIVE, "2025-03-20")
    now = _ts("2025-03-20", hour=15)
    assert lc.advance(ev, now) == ACTIVE


# -----------------------------------------------------------------------
# no cutoff gating -- ACTIVE until midnight
# -----------------------------------------------------------------------

def test_advance_active_at_21_utc():
    """21:00 UTC is no longer a cutoff; event stays ACTIVE."""
    ev  = _event(ACTIVE, "2025-03-20")
    now = _ts("2025-03-20", hour=21)
    assert lc.advance(ev, now) == ACTIVE


def test_advance_active_at_23_utc():
    ev  = _event(ACTIVE, "2025-03-20")
    now = _ts("2025-03-20", hour=23)
    assert lc.advance(ev, now) == ACTIVE


def test_discovered_at_22_utc_still_active():
    ev  = _event(DISCOVERED, "2025-03-20")
    now = _ts("2025-03-20", hour=22)
    assert lc.advance(ev, now) == ACTIVE


# -----------------------------------------------------------------------
# -> pending_settle
# -----------------------------------------------------------------------

def test_advance_to_pending_settle_next_day():
    ev  = _event(ACTIVE, "2025-03-20")
    now = _ts("2025-03-21", hour=0)    # midnight next day
    assert lc.advance(ev, now) == PENDING_SETTLE


def test_advance_to_pending_settle_well_after_event():
    ev  = _event(CUTOFF, "2025-03-20")
    now = _ts("2025-03-22", hour=12)
    assert lc.advance(ev, now) == PENDING_SETTLE


def test_pending_settle_stays_pending():
    ev  = _event(PENDING_SETTLE, "2025-03-20")
    now = _ts("2025-03-25", hour=12)
    assert lc.advance(ev, now) == PENDING_SETTLE


# -----------------------------------------------------------------------
# settled is terminal
# -----------------------------------------------------------------------

def test_settled_is_terminal():
    ev  = _event(SETTLED, "2025-03-20")
    now = _ts("2025-03-25", hour=12)
    assert lc.advance(ev, now) == SETTLED


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

def test_no_end_date_returns_current_status():
    ev  = {"status": DISCOVERED, "event_end_date": ""}
    now = _ts("2025-03-20", hour=10)
    assert lc.advance(ev, now) == DISCOVERED


def test_invalid_end_date_returns_current_status():
    ev  = {"status": ACTIVE, "event_end_date": "not-a-date"}
    now = _ts("2025-03-20", hour=10)
    assert lc.advance(ev, now) == ACTIVE


def test_naive_timestamp_works():
    """advance() should accept a timezone-naive pd.Timestamp."""
    ev  = _event(DISCOVERED, "2025-03-20")
    now = pd.Timestamp("2025-03-20T10:00:00")  # no tz
    assert lc.advance(ev, now) == ACTIVE


def test_full_end_iso_string():
    ev  = _event(ACTIVE, "2025-03-20T00:00:00Z")
    now = _ts("2025-03-20", hour=10)
    assert lc.advance(ev, now) == ACTIVE



# -----------------------------------------------------------------------
# is_tradeable convenience wrapper
# -----------------------------------------------------------------------

def test_is_tradeable_true_when_active():
    ev  = _event(DISCOVERED, "2025-03-20")
    now = _ts("2025-03-20", hour=10)
    assert lc.is_tradeable(ev, now) is True


def test_is_tradeable_true_at_21_utc():
    """21:00 UTC is now ACTIVE (cutoff gating removed); event is tradeable."""
    ev  = _event(ACTIVE, "2025-03-20")
    now = _ts("2025-03-20", hour=21)
    assert lc.is_tradeable(ev, now) is True


def test_is_tradeable_false_when_settled():
    ev  = _event(SETTLED, "2025-03-20")
    now = _ts("2025-03-20", hour=10)
    assert lc.is_tradeable(ev, now) is False


# -----------------------------------------------------------------------
# State transition matrix (exhaustive)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("status,hour,expected", [
    (DISCOVERED,     10, ACTIVE),
    (DISCOVERED,     21, ACTIVE),   # no cutoff hour gating
    (ACTIVE,         10, ACTIVE),
    (ACTIVE,         21, ACTIVE),   # no cutoff hour gating
    (CUTOFF,         10, ACTIVE),   # DB-legacy CUTOFF rows heal to ACTIVE
    (CUTOFF,         22, ACTIVE),   # DB-legacy CUTOFF rows heal to ACTIVE
    (PENDING_SETTLE, 10, PENDING_SETTLE),
    (SETTLED,        10, SETTLED),
])
def test_state_transition_matrix(status, hour, expected):
    ev  = _event(status, "2025-03-20")
    now = _ts("2025-03-20", hour=hour)
    assert lc.advance(ev, now) == expected
