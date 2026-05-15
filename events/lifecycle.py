"""Event lifecycle state machine.

States:
  discovered -> active -> pending_settle -> settled

Transitions are driven by the current UTC time relative to the
event's end date.  The CUTOFF constant is preserved for backward
compatibility with existing DB rows but is no longer a transition
target; rows with status 'cutoff' heal to 'active' on the next
advance() call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from db.schema import DbClient


# State constants (kept as plain strings so they match the DB column value)
DISCOVERED      = "discovered"
ACTIVE          = "active"
CUTOFF          = "cutoff"          # legacy -- no longer a transition target
PENDING_SETTLE  = "pending_settle"
SETTLED         = "settled"


class EventLifecycle:
    """Advance an event through its lifecycle based on the current time.

    State machine
    -------------
    discovered
        Initial state.  Event has been found in the Gamma API but trading
        has not started.

    active
        Markets are open and the HRRR forecast + TFT pipeline can generate
        signals.  Entered whenever the event end date has not yet passed
        (midnight UTC boundary).

    pending_settle
        The day after the event end date (midnight UTC).  Awaiting settlement
        data (actual observed temperature).

    settled
        Terminal state.  Set explicitly by calling set_settled(); advance()
        never transitions to this state automatically.
    """

    def __init__(self) -> None:
        pass

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
        current state is terminal (``settled``) or no end date is available.
        """
        current = event.get("status", DISCOVERED)

        if current == SETTLED:
            return SETTLED

        end_date_str = event.get("event_end_date") or ""
        if not end_date_str:
            return current

        try:
            event_end = pd.Timestamp(end_date_str, tz="UTC")
        except (ValueError, TypeError):
            return current

        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")

        if current == PENDING_SETTLE:
            return PENDING_SETTLE

        if now_utc >= event_end + pd.Timedelta(days=1):
            return PENDING_SETTLE

        return ACTIVE

    def set_settled(self, event_id: str, db: "DbClient") -> None:
        """Write the 'settled' terminal state for *event_id* to the DB.

        Must be called explicitly after all orders have been marked;
        advance() never transitions to 'settled' automatically.
        """
        db.update_event_status(event_id, SETTLED)

    def is_tradeable(self, event: dict, now_utc: pd.Timestamp) -> bool:
        """Return True if signals may be generated for this event.

        An event is tradeable when it is in the ``active`` state after
        advancing.  This is a convenience wrapper around ``advance()``.
        """
        return self.advance(event, now_utc) == ACTIVE
