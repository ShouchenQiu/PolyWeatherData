"""Tests for RedeemLoop -- poll and process redeem tasks."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from db.schema import DbClient
from loops.redeem_loop import RedeemLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    client = DbClient(":memory:")
    yield client
    client.close()


def _settings(trading_mode: str = "live", interval_s: float = 1.0):
    s = MagicMock()
    s.trading_mode             = trading_mode
    s.redeem_loop_interval_s   = interval_s
    return s


# ---------------------------------------------------------------------------
# RedeemLoop.run -- success path
# ---------------------------------------------------------------------------

def test_run_processes_pending_task_and_marks_redeemed(db):
    """RedeemLoop calls the SDK once per pending task and marks it redeemed."""
    db.enqueue_redeem_task("cid-1", status="pending")

    secure = MagicMock()
    secure.ctf_redeem_positions_call.return_value = None  # no error

    loop = RedeemLoop(db, secure, _settings())

    async def _run_one_cycle():
        # Let run() do one poll cycle then cancel
        task = asyncio.create_task(loop.run())
        # Sleep enough for one poll (interval=1s, but first poll is immediate)
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_cycle())

    secure.ctf_redeem_positions_call.assert_called_once_with("cid-1")
    # After success, no pending/failed tasks remain
    remaining = db.get_pending_redeem_tasks()
    assert len(remaining) == 0

    # Direct check: status is 'redeemed'
    rows = db.query("SELECT status FROM redeem_tasks WHERE condition_id=?", ("cid-1",))
    assert rows
    assert rows[0]["status"] == "redeemed"


# ---------------------------------------------------------------------------
# RedeemLoop -- failure / retry path
# ---------------------------------------------------------------------------

def test_run_failure_increments_attempt_and_records_error(db):
    """On SDK failure, status='failed' and attempt_count increments."""
    db.enqueue_redeem_task("cid-2", status="pending")

    secure = MagicMock()
    secure.ctf_redeem_positions_call.side_effect = RuntimeError("rpc timeout")

    loop = RedeemLoop(db, secure, _settings())

    async def _run_one_cycle():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_one_cycle())

    tasks = db.get_pending_redeem_tasks()
    assert len(tasks) == 1
    assert tasks[0]["condition_id"] == "cid-2"
    assert tasks[0]["status"] == "failed"
    assert tasks[0]["attempt_count"] >= 1
    assert "RuntimeError" in (tasks[0]["last_error"] or "")


# ---------------------------------------------------------------------------
# RedeemLoop -- paper mode guard
# ---------------------------------------------------------------------------

def test_paper_mode_marks_pending_tasks_skipped_paper(db):
    """Paper-mode RedeemLoop marks pending/failed tasks as skipped_paper."""
    db.enqueue_redeem_task("cid-3", status="pending")
    # Create a failed row via raw SQL for full coverage
    db._conn.execute(
        "INSERT INTO redeem_tasks (condition_id, status, attempt_count,"
        " created_at, updated_at) VALUES ('cid-4-fail', 'failed', 1, '', '')"
    )
    db._conn.commit()

    secure = MagicMock()
    loop = RedeemLoop(db, secure, _settings(trading_mode="paper"))

    async def _run():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    # SDK must not have been called
    secure.ctf_redeem_positions_call.assert_not_called()

    # Both tasks should be skipped_paper
    for cid in ("cid-3", "cid-4-fail"):
        rows = db.query(
            "SELECT status FROM redeem_tasks WHERE condition_id=?",
            (cid,),
        )
        assert rows
        assert rows[0]["status"] == "skipped_paper"


# ---------------------------------------------------------------------------
# RedeemLoop -- empty queue
# ---------------------------------------------------------------------------

def test_run_no_pending_tasks_does_not_call_sdk(db):
    """When no pending/failed tasks exist, the SDK is never called."""
    secure = MagicMock()
    loop = RedeemLoop(db, secure, _settings())

    async def _run():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    secure.ctf_redeem_positions_call.assert_not_called()
