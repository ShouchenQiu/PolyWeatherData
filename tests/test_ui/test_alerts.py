"""Tests for ui/alerts.py -- StalenessState."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ui.alerts import StalenessState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# StalenessState
# ---------------------------------------------------------------------------


def test_staleness_state_initially_not_stale() -> None:
    state = StalenessState()
    assert state.is_stale is False
    assert state.stale_since is None
    assert state.station is None


def test_staleness_state_mark_stale() -> None:
    state = StalenessState()
    state.mark_stale(_NOW, "KLGA")
    assert state.is_stale is True
    assert state.stale_since == _NOW
    assert state.station == "KLGA"


def test_staleness_state_mark_stale_station_none() -> None:
    state = StalenessState()
    state.mark_stale(_NOW, None)
    assert state.is_stale is True
    assert state.station is None
