"""Event lifecycle state machine.

States (in order):
  discovered  -> active -> cutoff -> pending_settle -> settled

Transitions are driven by the current UTC time relative to the
event's end date.
"""
from __future__ import annotations

import pandas as pd


# State constants (kept as plain strings so they match the DB column value)
DISCOVERED      = "discovered"
ACTIVE          = "active"
CUTOFF          = "cutoff"
PENDING_SETTLE  = "pending_settle"
SETTLED         = "settled"

# 16:00 EST = 21:00 UTC
_CUTOFF_HOUR_UTC = 21


class EventLifecycle:
    """Advance an event through its lifecycle based on the current time.

    State machine
    -------------
    discovered
        Initial state.  Event has been found in the Gamma API but trading
        has not started.

    active
        Markets are open and the HRRR forecast + TFT pipeline can generate
        signals.  Entered once the event end date is in the future and the
        cutoff hour has not yet passed.

    cutoff
        16:00 EST (21:00 UTC) on the event end date day.  No new orders
        should be placed; existing positions are held.

    pending_settle
        The event end date has passed (midnight UTC).  Awaiting settlement
        data (actual observed temperature).

    settled
        Settlement is complete (set externally by the Settler component).
        This state is NOT set by advance() -- it is set by marking status
        "settled" in the DB.

    Parameters
    ----------
    cutoff_hour_utc:
        The UTC hour at which trading is cut off on the event date.
        Defaults to 21 (16:00 EST).
    """

    def __init__(self, cutoff_hour_utc: int = _CUTOFF_HOUR_UTC) -> None:
        self.cutoff_hour_utc = cutoff_hour_utc

    def advance(self, event: dict, now_utc: pd.Timestamp) -> str:
        """Return the next state for *event* given current UTC time.

        Parameters
        ----------
        event:
            Dict with at minimum keys ``"status"`` and ``"event_end_date"``.
            ``event_end_date`` should be an ISO-8601 date string such as
            ``"2025-03-20"`` or ``"2025-03-20T00:00:00Z"``.
        now_utc:
            Current time in UTC (timezone-aware preferred).

        Returns
        -------
        New status string.  Returns current status unchanged when the
        current state is terminal (``settled``) or no transition applies.
        """
        current = event.get("status", DISCOVERED)

        if current == SETTLED:
            return SETTLED

        end_date_str = event.get("event_end_date") or ""
        if not end_date_str:
            # Cannot determine transitions without an end date
            return current

        try:
            event_end = pd.Timestamp(end_date_str, tz="UTC")
        except (ValueError, TypeError):
            return current

        # Normalise now_utc to UTC aware
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")

        # Cutoff boundary: event_end_date at cutoff_hour_utc
        cutoff_ts = event_end.normalize() + pd.Timedelta(hours=self.cutoff_hour_utc)

        # --- State transitions ---
        if current == PENDING_SETTLE:
            # No further automatic advance beyond pending_settle
            return PENDING_SETTLE

        if now_utc >= event_end + pd.Timedelta(days=1):
            # Day after the event date -> pending settlement
            return PENDING_SETTLE

        if now_utc >= cutoff_ts:
            # Past cutoff on the event date
            return CUTOFF

        # Before cutoff: always ACTIVE (time-driven, overrides stale status)
        return ACTIVE

    def is_tradeable(self, event: dict, now_utc: pd.Timestamp) -> bool:
        """Return True if signals may be generated for this event.

        An event is tradeable when it is in the ``active`` state after
        advancing.  This is a convenience wrapper around ``advance()``.
        """
        return self.advance(event, now_utc) == ACTIVE
