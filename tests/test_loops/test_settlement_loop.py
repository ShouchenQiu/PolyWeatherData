"""Tests for V12 -- Event Lifecycle + Settlement Loop."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.exchange.settlement import SettlementRelayer
from core.positions import PositionTracker
from data.storage.parquet_store import ParquetStore
from db.schema import DbClient
from events.lifecycle import ACTIVE, DISCOVERED, PENDING_SETTLE, SETTLED, EventLifecycle
from loops.settlement_loop import SettlementLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> DbClient:
    d = DbClient(":memory:")
    d.upsert_station("KLGA", 40.78, -73.88, "America/New_York")
    d.upsert_event(
        "cid1", "Will it be 60-70F?", "temp-60-70",
        "2025-03-20", "KLGA", status="pending_settle",
    )
    d.upsert_market("tok-yes", "cid1", "slug-60-70", 60.0, 70.0, "YES")
    d.execute(
        "INSERT INTO orders "
        "(order_id, token_id, condition_id, side, size_usd, price, status, created_at)"
        " VALUES ('ord-yes', 'tok-yes', 'cid1', 'BUY', 100.0, 0.5, 'filled', '2025-01-01')"
    )
    return d


@pytest.fixture
def obs_store(tmp_path: Path) -> ParquetStore:
    store = ParquetStore(tmp_path / "obs")
    df = pd.DataFrame({
        "valid_time": [pd.Timestamp("2025-03-20 12:00", tz="UTC"),
                       pd.Timestamp("2025-03-20 15:00", tz="UTC")],
        "tmpf":       [63.5, 65.1],   # max = 65.1 F, in bucket 60-70
    })
    store.append("KLGA", df)
    return store


@pytest.fixture
def lifecycle() -> EventLifecycle:
    return EventLifecycle()


@pytest.fixture
def settlement_loop(db: DbClient, obs_store: ParquetStore, lifecycle: EventLifecycle) -> SettlementLoop:
    return SettlementLoop(
        lifecycle  = lifecycle,
        positions  = PositionTracker(db),
        obs_store  = obs_store,
        db         = db,
    )


def _event_row(db: DbClient, condition_id: str) -> dict:
    rows = db.query("SELECT * FROM events WHERE condition_id=?", (condition_id,))
    return dict(rows[0])


def _order_row(db: DbClient, order_id: str) -> dict:
    rows = db.query("SELECT * FROM orders WHERE order_id=?", (order_id,))
    return dict(rows[0])


# ---------------------------------------------------------------------------
# SettlementRelayer stub
# ---------------------------------------------------------------------------


def test_settlement_relayer_raises_not_implemented():
    """resolve_on_chain raises NotImplementedError referencing ADR-0007."""
    relayer = SettlementRelayer()
    with pytest.raises(NotImplementedError, match="ADR-0007"):
        relayer.resolve_on_chain("cid1")


# ---------------------------------------------------------------------------
# EventLifecycle.set_settled
# ---------------------------------------------------------------------------


def test_set_settled_writes_settled_to_db(db: DbClient, lifecycle: EventLifecycle):
    lifecycle.set_settled("cid1", db)
    assert _event_row(db, "cid1")["status"] == "settled"


def test_set_settled_not_called_by_advance(lifecycle: EventLifecycle):
    """`advance()` never transitions to 'settled' on its own."""
    ev = {"status": PENDING_SETTLE, "event_end_date": "2025-03-20"}
    now = pd.Timestamp("2025-03-25T12:00:00", tz="UTC")
    assert lifecycle.advance(ev, now) == PENDING_SETTLE  # stays pending, not settled


# ---------------------------------------------------------------------------
# EventLifecycle.advance -- V12 midnight boundary, no cutoff
# ---------------------------------------------------------------------------


def test_advance_pending_settle_at_midnight_of_next_day():
    """One second past midnight of the day after event_end_date -> PENDING_SETTLE."""
    lc  = EventLifecycle()
    ev  = {"status": ACTIVE, "event_end_date": "2025-03-20"}
    now = pd.Timestamp("2025-03-21T00:00:01", tz="UTC")
    assert lc.advance(ev, now) == PENDING_SETTLE


def test_advance_active_on_event_end_day():
    """On the event_end_date itself (before midnight), still ACTIVE."""
    lc  = EventLifecycle()
    ev  = {"status": ACTIVE, "event_end_date": "2025-03-20"}
    now = pd.Timestamp("2025-03-20T21:00:00", tz="UTC")  # 21:00 UTC, no cutoff
    assert lc.advance(ev, now) == ACTIVE


def test_advance_no_cutoff_hour_at_21_utc():
    """21:00 UTC on event date is now ACTIVE (cutoff gating removed)."""
    lc  = EventLifecycle()
    ev  = {"status": DISCOVERED, "event_end_date": "2025-03-20"}
    # Under old code this would be CUTOFF; under V12 it's ACTIVE
    assert lc.advance(ev, pd.Timestamp("2025-03-20T21:00:00", tz="UTC")) == ACTIVE


# ---------------------------------------------------------------------------
# confirm_settlement -- winning bucket
# ---------------------------------------------------------------------------


def test_confirm_settlement_winning_bucket_payout(
    settlement_loop: SettlementLoop,
    db: DbClient,
):
    """Order in the winning bucket receives payout == size_usd."""
    event = _event_row(db, "cid1")
    settlement_loop.confirm_settlement(event, "KLGA")
    order = _order_row(db, "ord-yes")
    assert order["status"] == "settled"
    assert order["settled_payout"] == pytest.approx(100.0)


def test_confirm_settlement_losing_bucket_payout_zero(
    db: DbClient,
    obs_store: ParquetStore,
    lifecycle: EventLifecycle,
):
    """Order in a losing bucket receives payout == 0.0."""
    # Add a second market and order in the losing bucket (70-80F)
    db.upsert_market("tok-lose", "cid1", "slug-70-80", 70.0, 80.0, "YES")
    db.execute(
        "INSERT INTO orders "
        "(order_id, token_id, condition_id, side, size_usd, price, status, created_at)"
        " VALUES ('ord-lose', 'tok-lose', 'cid1', 'BUY', 50.0, 0.5, 'filled', '2025-01-01')"
    )
    loop = SettlementLoop(
        lifecycle  = lifecycle,
        positions  = PositionTracker(db),
        obs_store  = obs_store,
        db         = db,
    )
    event = _event_row(db, "cid1")
    loop.confirm_settlement(event, "KLGA")
    # Winning order paid out
    assert _order_row(db, "ord-yes")["settled_payout"] == pytest.approx(100.0)
    # Losing order gets zero
    assert _order_row(db, "ord-lose")["settled_payout"] == pytest.approx(0.0)


def test_confirm_settlement_calls_set_settled_once(
    settlement_loop: SettlementLoop,
    db: DbClient,
):
    """After confirm_settlement, event status is 'settled' exactly once."""
    event = _event_row(db, "cid1")
    settlement_loop.confirm_settlement(event, "KLGA")
    assert _event_row(db, "cid1")["status"] == "settled"


def test_confirm_settlement_no_obs_data_skips_without_error(
    db: DbClient,
    tmp_path: Path,
    lifecycle: EventLifecycle,
):
    """When the obs store has no data for the station+date, settlement is skipped gracefully."""
    empty_store = ParquetStore(tmp_path / "empty")
    loop = SettlementLoop(
        lifecycle  = lifecycle,
        positions  = PositionTracker(db),
        obs_store  = empty_store,
        db         = db,
    )
    event = _event_row(db, "cid1")
    loop.confirm_settlement(event, "KLGA")  # must not raise
    # Event stays pending_settle (not settled) since we had no data
    assert _event_row(db, "cid1")["status"] == "pending_settle"


def test_confirm_settlement_no_station_skips(
    settlement_loop: SettlementLoop,
    db: DbClient,
):
    """When station is None, confirm_settlement skips gracefully."""
    event = _event_row(db, "cid1")
    settlement_loop.confirm_settlement(event, None)  # must not raise
    assert _event_row(db, "cid1")["status"] == "pending_settle"


# ---------------------------------------------------------------------------
# SettlementLoop.run -- uses get_pending_settlement_events
# ---------------------------------------------------------------------------


def test_run_processes_pending_settle_events(
    settlement_loop: SettlementLoop,
    db: DbClient,
):
    """run() tick settles all pending_settle events and cancels cleanly."""
    async def run_once():
        task = asyncio.create_task(settlement_loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_once())
    assert _event_row(db, "cid1")["status"] == "settled"
