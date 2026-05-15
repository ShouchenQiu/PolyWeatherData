"""Tests for core/risk/ -- FixedFraction, KellyCriterion, RiskPipeline."""
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


# -----------------------------------------------------------------------
# FixedFraction -- duplicate-position prevention via open_positions
# -----------------------------------------------------------------------

def test_approve_signal_false_when_at_token_limit():
    """Reject when open_positions[token_id] >= max_open_per_token."""
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_open_per_token=1)
    sig = _signal()
    sig = Signal(token_id="tok1", side="BUY", mc_prob=0.6,
                 clob_price=0.4, edge=0.2)
    open_pos = {"tok1": 1}   # already at limit
    assert risk.approve_signal(sig, current_exposure=0.0,
                               open_positions=open_pos) is False


def test_approve_signal_true_when_below_token_limit():
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_open_per_token=2)
    sig = Signal(token_id="tok1", side="BUY", mc_prob=0.6,
                 clob_price=0.4, edge=0.2)
    open_pos = {"tok1": 1}   # 1 open, limit is 2
    assert risk.approve_signal(sig, current_exposure=0.0,
                               open_positions=open_pos) is True


def test_approve_signal_true_when_token_not_in_open_positions():
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_open_per_token=1)
    sig = Signal(token_id="tok-new", side="BUY", mc_prob=0.6,
                 clob_price=0.4, edge=0.2)
    open_pos = {"tok-other": 1}   # different token
    assert risk.approve_signal(sig, current_exposure=0.0,
                               open_positions=open_pos) is True


def test_approve_signal_no_open_positions_still_passes():
    """open_positions=None skips the per-token check (backward compat)."""
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_open_per_token=1)
    sig = Signal(token_id="tok1", side="BUY", mc_prob=0.6,
                 clob_price=0.4, edge=0.2)
    assert risk.approve_signal(sig, current_exposure=0.0,
                               open_positions=None) is True


# -----------------------------------------------------------------------
# KellyCriterion
# -----------------------------------------------------------------------

def test_kelly_criterion_approve_raises():
    from core.risk.kelly import KellyCriterion
    k = KellyCriterion()
    with pytest.raises(NotImplementedError):
        k.approve_signal(_signal(), current_exposure=0.0)


def test_kelly_criterion_compute_size_raises():
    from core.risk.kelly import KellyCriterion
    k = KellyCriterion()
    with pytest.raises(NotImplementedError):
        k.compute_size(_signal(), current_exposure=0.0)


# -----------------------------------------------------------------------
# RiskPipeline
# -----------------------------------------------------------------------

def test_risk_pipeline_import():
    from core.risk.pipeline import RiskPipeline  # noqa: F401


def test_risk_pipeline_empty_list_raises():
    from core.risk.pipeline import RiskPipeline
    with pytest.raises(ValueError, match="at least one"):
        RiskPipeline([])


def test_risk_pipeline_approves_when_all_approve():
    from core.risk.pipeline import RiskPipeline
    r1 = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    r2 = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    rp = RiskPipeline([r1, r2])
    assert rp.approve_signal(_signal(), current_exposure=0.0) is True


def test_risk_pipeline_rejects_when_one_rejects():
    from core.risk.pipeline import RiskPipeline
    r_ok     = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    r_reject = FixedFraction(position_usd=20.0, max_total_usd=10.0)
    rp = RiskPipeline([r_ok, r_reject])
    # current_exposure=9 + 20 = 29 > 10, so r_reject rejects
    assert rp.approve_signal(_signal(), current_exposure=9.0) is False


def test_risk_pipeline_compute_size_not_called_on_rejected():
    """compute_size is never invoked when approve_signal returns False."""
    from core.risk.pipeline import RiskPipeline

    called: list[str] = []

    class SpyRisk(FixedFraction):
        def approve_signal(self, signal, current_exposure, open_positions=None):
            return False

        def compute_size(self, signal, current_exposure, open_positions=None):
            called.append("compute_size")
            return 20.0

    rp = RiskPipeline([SpyRisk()])
    rp.approve_signal(_signal(), current_exposure=0.0)
    assert called == []


def test_risk_pipeline_compute_size_returns_minimum():
    """compute_size returns the minimum positive value across all sub-models."""
    from core.risk.pipeline import RiskPipeline
    r_big   = FixedFraction(position_usd=50.0, max_total_usd=500.0)
    r_small = FixedFraction(position_usd=10.0, max_total_usd=500.0)
    rp = RiskPipeline([r_big, r_small])
    size = rp.compute_size(_signal(), current_exposure=0.0)
    assert size == pytest.approx(10.0)


def test_risk_pipeline_passes_open_positions_through():
    """open_positions kwarg is forwarded to each sub-model."""
    from core.risk.pipeline import RiskPipeline

    received: list[dict | None] = []

    class SpyRisk(FixedFraction):
        def approve_signal(self, signal, current_exposure, open_positions=None):
            received.append(open_positions)
            return True

    rp = RiskPipeline([SpyRisk(position_usd=20.0, max_total_usd=500.0)])
    rp.approve_signal(_signal(), current_exposure=0.0,
                      open_positions={"tok1": 1})
    assert received == [{"tok1": 1}]
