"""Tests for core/risk/ -- FixedFraction risk model."""
from __future__ import annotations

import pytest

from core.risk.fixed_fraction import FixedFraction
from core.types import Signal


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _signal(mc_prob: float = 0.6, clob_price: float = 0.4) -> Signal:
    return Signal(
        token_id   = "tok1",
        side       = "BUY",
        mc_prob    = mc_prob,
        clob_price = clob_price,
        edge       = mc_prob - clob_price,
    )


# -----------------------------------------------------------------------
# compute_size
# -----------------------------------------------------------------------

def test_compute_size_returns_position_usd_when_headroom_sufficient():
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    size = risk.compute_size(_signal(), current_exposure=0.0)
    assert size == 20.0


def test_compute_size_clips_to_remaining_headroom():
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    # Only 15 USDC of headroom left
    size = risk.compute_size(_signal(), current_exposure=85.0)
    assert size == 15.0


def test_compute_size_zero_when_fully_at_cap():
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    size = risk.compute_size(_signal(), current_exposure=100.0)
    assert size == 0.0


def test_compute_size_clips_to_max_position():
    risk = FixedFraction(position_usd=200.0, max_position_usd=50.0,
                         max_total_usd=500.0)
    size = risk.compute_size(_signal(), current_exposure=0.0)
    assert size == 50.0


# -----------------------------------------------------------------------
# approve_signal
# -----------------------------------------------------------------------

def test_approve_signal_true_within_limits():
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    assert risk.approve_signal(_signal(), current_exposure=0.0) is True


def test_approve_signal_false_over_total_cap():
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    # Exposure 95 + 20 = 115 > 100
    assert risk.approve_signal(_signal(), current_exposure=95.0) is False


def test_approve_signal_false_exactly_at_cap():
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    assert risk.approve_signal(_signal(), current_exposure=100.0) is False


def test_approve_signal_true_at_boundary():
    """80 + 20 = 100 == max_total_usd: still within cap (not exceeded)."""
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    assert risk.approve_signal(_signal(), current_exposure=80.0) is True


# -----------------------------------------------------------------------
# from_settings
# -----------------------------------------------------------------------

def test_from_settings():
    class FakeSettings:
        max_position_usd = 25.0
        max_total_usd    = 300.0

    risk = FixedFraction.from_settings(FakeSettings())
    assert risk.position_usd     == 25.0
    assert risk.max_position_usd == 25.0
    assert risk.max_total_usd    == 300.0


# -----------------------------------------------------------------------
# Edge: zero position_usd
# -----------------------------------------------------------------------

def test_zero_position_usd_never_approved():
    risk = FixedFraction(position_usd=0.0, max_total_usd=500.0)
    assert risk.approve_signal(_signal(), current_exposure=0.0) is False
