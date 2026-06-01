"""SettlementLoop: confirm temperature-bucket settlements for expired events."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from core.settlement_resolver import SettlementResolver

if TYPE_CHECKING:
    from core.exchange.settlement import SettlementRelayer
    from core.positions import PositionTracker
    from data.storage.parquet_store import ParquetStore
    from db.schema import DbClient
    from events.lifecycle import EventLifecycle


class SettlementLoop:
    """Confirm settlement for events in ``pending_settle`` status.

    For each pending event:
    1. Reads IEM observed max temperature from the ParquetStore.
    2. Identifies the winning bucket (bucket_lo_f <= max_temp < bucket_hi_f).
    3. Marks each filled/closed order settled with YES/NO-correct payout.
    4. Enqueues a redeem task for live-mode winning conditions (when is_live=True).
    5. Calls EventLifecycle.set_settled() to finalise the event.

    When IEM data is unavailable, falls back to ``relayer.resolve_on_chain``
    if a relayer is wired.  If neither source is available, the event stays
    in ``pending_settle`` for the next attempt.

    Parameters
    ----------
    lifecycle:
        EventLifecycle for the set_settled() call.
    positions:
        PositionTracker used to write the per-order settled_payout.
    obs_store:
        ParquetStore holding IEM observed temperature data.
    db:
        DbClient for querying events, markets, and orders.
    is_live:
        True when running in live trading mode (triggers redeem task
        enqueue with status='pending'); False for paper mode (enqueues
        with status='skipped_paper').
    relayer:
        Optional SettlementRelayer for on-chain resolution when IEM data is
        unavailable.  Defaults to None (deferred -- ADR-0007).
    interval_seconds:
        How often to run the settlement check.  Default 3600 (1 h).
    """

    def __init__(
        self,
        lifecycle:        "EventLifecycle",
        positions:        "PositionTracker",
        obs_store:        "ParquetStore",
        db:               "DbClient",
        is_live:          bool = False,
        relayer:          "SettlementRelayer | None" = None,
        interval_seconds: float = 3_600.0,
    ) -> None:
        self._lifecycle        = lifecycle
        self._positions        = positions
        self._obs_store        = obs_store
        self._db               = db
        self._is_live          = is_live
        self._relayer          = relayer
        self._interval_seconds = interval_seconds
        self._resolver         = SettlementResolver(db)

    # ------------------------------------------------------------------
    # Settlement logic
    # ------------------------------------------------------------------

    def confirm_settlement(self, event: dict, station: str | None) -> None:
        """Settle all filled/closed orders for *event* and mark it settled.

        Payout rules:
          - YES token in the winning bucket  -> payout = size_usd
          - YES token in a losing bucket     -> payout = 0.0
          - NO token in the winning bucket   -> payout = 0.0
          - NO token in a losing bucket      -> payout = size_usd

        If IEM obs data is unavailable and a relayer is wired, delegates to
        ``relayer.resolve_on_chain``.  Otherwise the event stays in
        ``pending_settle`` for the next check.

        The observation window uses the station's local calendar day
        (midnight to midnight in station timezone), not UTC day boundaries.
        This matches the daily-high target semantics from ADR-0010.

        Parameters
        ----------
        event:
            Row dict from the events table (must contain ``condition_id`` and
            ``event_end_date``).
        station:
            IEM station identifier (e.g. ``"KLGA"``).
        """
        from zoneinfo import ZoneInfo

        from config.stations import STATIONS

        condition_id = event["condition_id"]
        end_date_str = event.get("event_end_date") or ""

        if not station:
            logger.warning(
                "SettlementLoop: no station for event {}; skipping", condition_id
            )
            return

        if not end_date_str:
            logger.warning(
                "SettlementLoop: no event_end_date for event {}; skipping",
                condition_id,
            )
            return

        # Resolve the station's local calendar day boundaries.
        station_cfg = STATIONS.get(station)
        tz = ZoneInfo(station_cfg.timezone) if station_cfg else ZoneInfo("UTC")
        try:
            local_midnight = pd.Timestamp(end_date_str).tz_localize(tz)
        except (ValueError, TypeError):
            logger.warning(
                "SettlementLoop: unparseable event_end_date '{}' for {}; skipping",
                end_date_str,
                condition_id,
            )
            return

        local_day_end = local_midnight + pd.Timedelta(days=1)
        # Convert to UTC for ParquetStore range query.
        day_start_utc = local_midnight.tz_convert("UTC")
        day_end_utc   = local_day_end.tz_convert("UTC")
        obs_df        = self._obs_store.read(station, day_start_utc, day_end_utc)

        # Historical column name is misleading: current IEM/NOAA sources store
        # this as degrees Celsius, preserving tenths precision as a float.
        if obs_df.empty or "raw_tmpc_tenths" not in obs_df.columns:
            logger.warning(
                "SettlementLoop: no raw_tmpc_tenths obs for station={} date={}; event={}",
                station,
                end_date_str,
                condition_id,
            )
            self._try_relayer_fallback(condition_id)
            return

        tmp_c = obs_df["raw_tmpc_tenths"].max()
        actual_max_temp: float = tmp_c * 9.0 / 5.0 + 32.0
        self._settle_from_obs(condition_id, actual_max_temp)

    def _settle_from_obs(self, condition_id: str, actual_max_temp: float) -> None:
        """Apply YES/NO payout logic, enqueue redeem if winner, and mark settled."""
        settlements = self._resolver.resolve_from_obs(condition_id, actual_max_temp)
        for s in settlements:
            self._positions.mark_settled(s.order_id, s.payout)
        self._enqueue_redeem_if_winner(condition_id, settlements)
        self._lifecycle.set_settled(condition_id, self._db)
        logger.info(
            "SettlementLoop: event {} settled via IEM (max_temp={:.1f}F, orders={})",
            condition_id,
            actual_max_temp,
            len(settlements),
        )

    def _enqueue_redeem_if_winner(
        self, condition_id: str, settlements: list,
    ) -> None:
        """Upsert a redeem task when any winning token has net-long position.

        Only tokens with at least one winning settlement (payout > 0) are
        considered.  A losing BUY order (signed_shares > 0, payout = 0) must
        NOT trigger a redeem task.

        Net signed_shares per winning token must be > 0.  A SELL-only token
        or a BUY+SELL net-zero pair does not trigger a redeem task.

        Live mode: status='pending'.  Paper mode: status='skipped_paper'.
        """
        from collections import defaultdict

        # Only tokens that actually won (at least one settlement payout > 0)
        winning_tokens: set[str] = {
            s.token_id for s in settlements if s.payout > 0.0
        }
        if not winning_tokens:
            return

        net_by_token: dict[str, float] = defaultdict(float)
        for s in settlements:
            if s.token_id in winning_tokens:
                net_by_token[s.token_id] += s.signed_shares

        has_net_long = any(v > 0.0 for v in net_by_token.values())
        if not has_net_long:
            return
        market_id = self._market_id_for_redeem(winning_tokens)
        status = "pending" if self._is_live else "skipped_paper"
        suffix = "" if self._is_live else " (paper mode)"
        self._db.enqueue_redeem_task(
            condition_id,
            status=status,
            market_id=market_id,
        )
        logger.info(
            "SettlementLoop: redeem task enqueued for {} market_id={} (status={}{})",
            condition_id, market_id or "<legacy>", status, suffix,
        )

    def _market_id_for_redeem(self, winning_tokens: set[str]) -> str | None:
        """Return the SDK market_id for one winning token, when available."""
        placeholders = ",".join("?" for _ in winning_tokens)
        if not placeholders:
            return None
        try:
            rows = self._db.query(
                f"""
                SELECT DISTINCT market_id FROM markets
                WHERE token_id IN ({placeholders})
                  AND market_id IS NOT NULL
                  AND market_id != ''
                ORDER BY market_id
                """,
                tuple(winning_tokens),
            )
        except Exception:
            logger.exception("SettlementLoop: failed to resolve redeem market_id")
            return None
        return str(rows[0]["market_id"]) if rows else None

    def _try_relayer_fallback(self, condition_id: str) -> None:
        """Attempt on-chain resolution via SettlementRelayer when IEM data is absent."""
        if self._relayer is None:
            logger.info(
                "SettlementLoop: no relayer wired for {}; staying pending_settle",
                condition_id,
            )
            return
        try:
            resolved = self._relayer.resolve_on_chain(condition_id)
            if resolved:
                self._lifecycle.set_settled(condition_id, self._db)
                logger.info(
                    "SettlementLoop: event {} settled via SettlementRelayer",
                    condition_id,
                )
            else:
                logger.warning(
                    "SettlementLoop: relayer returned False for {}; staying pending_settle",
                    condition_id,
                )
        except NotImplementedError:
            logger.warning(
                "SettlementLoop: SettlementRelayer.resolve_on_chain not yet "
                "implemented for {}; staying pending_settle (ADR-0007)",
                condition_id,
            )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Periodically confirm settlement for all pending_settle events."""
        logger.info("SettlementLoop started (interval={}s)", self._interval_seconds)
        while True:
            try:
                events = self._db.get_pending_settlement_events()
                for event in events:
                    cid        = event["condition_id"]
                    station_id = event.get("station_id")
                    try:
                        await asyncio.to_thread(
                            self.confirm_settlement, event, station_id
                        )
                    except Exception:
                        logger.exception(
                            "SettlementLoop: confirm_settlement failed for {}", cid
                        )
            except Exception:
                logger.exception("SettlementLoop: unhandled error in run()")
            await asyncio.sleep(self._interval_seconds)
