"""Tests for core/risk/ -- FixedFraction, KellyCriterion, RiskPipeline."""
from __future__ import annotations

import pytest

from core.risk.fixed_fraction import FixedFraction
from core.types import OrderIntent


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _signal(executable_price: float = 0.4) -> OrderIntent:
    return OrderIntent(
        token_id         = "tok1",
        side             = "BUY",
        size_usd         = None,
        executable_price = executable_price,
        edge             = 0.2,
    )


@pytest.fixture()
def db_with_orders():
    """In-memory DB with a single filled BUY: 50 shares of tok1 at 0.40."""
    from db.schema import DbClient
    from core.positions import PositionTracker
    db = DbClient(":memory:")
    db.upsert_station("K", 40.0, -70.0, "UTC")
    db.upsert_event("cid-1", "Q", "slug", "2026-07-04", "K")
    db.upsert_market("tok1", "cid-1", "ms", 60.0, 70.0, "YES")
    tracker = PositionTracker(db)
    tracker.record_order("o1", "tok1", "cid-1", "BUY", 20.0, 0.40)
    tracker.mark_filled("o1", 0.40)
    return db


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


def test_approve_signal_sell_bypasses_exposure_gate():
    """SELL signals are approved even when BUY headroom is exhausted."""
    risk = FixedFraction(position_usd=20.0, max_total_usd=100.0)
    sell_sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                           executable_price=0.60, edge=0.2)
    assert risk.approve_signal(sell_sig, current_exposure=100.0) is True


def test_approve_signal_sell_bypasses_zero_position_usd_gate():
    """position_usd=0 blocks BUY but must not block SELL."""
    risk = FixedFraction(position_usd=0.0, max_total_usd=500.0)
    sell_sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                           executable_price=0.60, edge=0.2)
    assert risk.approve_signal(sell_sig, current_exposure=0.0) is True


# -----------------------------------------------------------------------
# FixedFraction -- SELL sizing via positions (V2F05)
# -----------------------------------------------------------------------

def test_compute_size_sell_uses_position_shares(db_with_orders):
    """SELL size = max_exit_fraction * net_shares * clob_price."""
    from core.positions import PositionTracker
    tracker = PositionTracker(db_with_orders)
    # 50 shares open from BUY at 0.40
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_exit_fraction=1.0)
    sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                      executable_price=0.60, edge=0.2)
    size = risk.compute_size(sig, current_exposure=0.0, positions=tracker)
    # 1.0 * 50 * 0.60 = 30
    assert size == pytest.approx(30.0)


def test_compute_size_sell_partial_exit(db_with_orders):
    """max_exit_fraction < 1 yields a partial exit."""
    from core.positions import PositionTracker
    tracker = PositionTracker(db_with_orders)
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0,
                         max_exit_fraction=0.5)
    sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                      executable_price=0.60, edge=0.2)
    size = risk.compute_size(sig, current_exposure=0.0, positions=tracker)
    # 0.5 * 50 * 0.60 = 15
    assert size == pytest.approx(15.0)


def test_compute_size_sell_uses_executable_price_when_set(db_with_orders):
    """SELL size uses executable_price for sizing."""
    from core.positions import PositionTracker
    tracker = PositionTracker(db_with_orders)
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                      executable_price=0.65, edge=0.2)
    size = risk.compute_size(sig, current_exposure=0.0, positions=tracker)
    # 1.0 * 50 * 0.65 = 32.5
    assert size == pytest.approx(32.5)


def test_compute_size_sell_zero_when_no_positions():
    """SELL size is 0 when no positions tracker is provided."""
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    sig = OrderIntent(token_id="tok1", side="SELL", size_usd=None,
                      executable_price=0.60, edge=0.2)
    assert risk.compute_size(sig, current_exposure=0.0) == pytest.approx(0.0)


def test_compute_size_sell_zero_when_no_net_shares(db_with_orders):
    """SELL size is 0 when net shares for the token are 0 or negative."""
    from core.positions import PositionTracker
    tracker = PositionTracker(db_with_orders)
    risk = FixedFraction(position_usd=20.0, max_total_usd=500.0)
    sig = OrderIntent(token_id="tok-unknown", side="SELL", size_usd=None,
                      executable_price=0.60, edge=0.2)
    assert risk.compute_size(sig, current_exposure=0.0, positions=tracker) == pytest.approx(0.0)


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
        def approve_signal(self, signal, current_exposure):
            return False

        def compute_size(self, signal, current_exposure, positions=None):
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
