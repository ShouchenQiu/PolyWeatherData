"""RedeemLoop: poll redeem_tasks and call CTF redeem on-chain via the SDK."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from db.schema import DbClient


class RedeemLoop:
    """Process pending and failed redeem tasks.

    Polls ``redeem_tasks`` for rows with status ``pending`` or ``failed``.
    Calls ``SecureClient.ctf_redeem_positions_call(condition_id)`` once per
    condition.  On success writes ``redeemed``; on failure writes ``failed``,
    increments ``attempt_count``, records ``last_error``, and retries on the
    next poll cycle.

    Parameters
    ----------
    db:
        DbClient for querying/updating redeem_tasks.
    secure_client:
        ``SecureClient`` instance from polymarket_client.  Must expose
        ``ctf_redeem_positions_call(condition_id)``.
    settings:
        Settings instance providing ``redeem_loop_interval_s``.
    """

    def __init__(
        self,
        db:            "DbClient",
        secure_client: Any,
        settings:      "Settings",
    ) -> None:
        self._db               = db
        self._secure           = secure_client
        self._interval_seconds = settings.redeem_loop_interval_s
        self._is_paper         = (settings.trading_mode == "paper")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll and process redeem tasks until cancelled."""
        if self._is_paper:
            logger.info(
                "RedeemLoop: paper mode; marking pending tasks skipped_paper"
            )
            self._skip_paper_tasks()
            logger.info(
                "RedeemLoop: paper-mode guard active; not polling"
            )
            # Sleep forever in paper mode: no real polling needed.
            while True:
                await asyncio.sleep(self._interval_seconds)

        logger.info(
            "RedeemLoop started (interval={}s)",
            self._interval_seconds,
        )
        while True:
            try:
                tasks = self._db.get_pending_redeem_tasks()
                for task in tasks:
                    cid = task["condition_id"]
                    try:
                        await asyncio.to_thread(self._process_one, cid)
                    except Exception:
                        logger.exception(
                            "RedeemLoop: process_one failed for {}", cid
                        )
            except Exception:
                logger.exception("RedeemLoop: unhandled error in run()")
            await asyncio.sleep(self._interval_seconds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_one(self, condition_id: str) -> None:
        """Call the SDK redeem and record success or failure.

        Runs in a thread via ``asyncio.to_thread`` to avoid blocking the
        bot's event loop on slow RPC calls.
        """
        logger.info("RedeemLoop: processing {}", condition_id)
        try:
            self._secure.ctf_redeem_positions_call(condition_id)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "RedeemLoop: redeem failed for {}: {}", condition_id, err_msg
            )
            self._db.mark_redeem_failed(condition_id, err_msg)
            return

        self._db.mark_redeemed(condition_id)
        logger.info("RedeemLoop: {} redeemed", condition_id)

    def _skip_paper_tasks(self) -> None:
        """Mark any pending/failed tasks as skipped_paper in paper mode."""
        tasks = self._db.get_pending_redeem_tasks()
        for task in tasks:
            cid = task["condition_id"]
            self._db.enqueue_redeem_task(cid, status="skipped_paper")
            logger.info(
                "RedeemLoop: paper mode -- redeem skipped for condition_id={}",
                cid,
            )
