"""Fixed-fraction position sizing model.

Allocates a fixed USDC amount per signal, subject to two hard caps:
  - max_position_usd   : maximum USDC per individual position.
  - max_total_usd      : maximum total open USDC exposure (portfolio cap).
  - max_open_per_token : maximum concurrent open orders per token_id.

Signals are rejected when any cap would be exceeded or the per-token
open-position limit is reached.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.risk.base import BaseRiskModel

if TYPE_CHECKING:
    from core.types import Signal


class FixedFraction(BaseRiskModel):
    """Size each trade at a fixed USDC amount with hard portfolio caps.

    Parameters
    ----------
    position_usd:
        Target USDC size per signal.
    max_position_usd:
        Hard cap on any single position (defaults to position_usd).
    max_total_usd:
        Portfolio cap: total open exposure must not exceed this value.
    max_open_per_token:
        Maximum concurrent open orders (pending|filled) for a single
        token_id.  Signals are rejected when
        ``open_positions.get(token_id, 0) >= max_open_per_token``.
        Defaults to 1.
    """

    def __init__(
        self,
        position_usd:      float = 20.0,
        max_position_usd:  float | None = None,
        max_total_usd:     float = 500.0,
        max_open_per_token: int  = 1,
    ) -> None:
        self.position_usd      = position_usd
        self.max_position_usd  = max_position_usd if max_position_usd is not None else position_usd
        self.max_total_usd     = max_total_usd
        self.max_open_per_token = max_open_per_token

    @classmethod
    def from_settings(cls, settings: object) -> "FixedFraction":
        return cls(
            position_usd       = getattr(settings, "max_position_usd",    20.0),
            max_position_usd   = getattr(settings, "max_position_usd",    20.0),
            max_total_usd      = getattr(settings, "max_total_usd",      500.0),
            max_open_per_token = getattr(settings, "max_open_per_token",      1),
        )

    def approve_signal(
        self,
        signal:           "Signal",
        current_exposure: float,
        open_positions:   dict[str, int] | None = None,
    ) -> bool:
        """Return False unless there is headroom for a full position_usd trade.

        Rejects when:
        - position_usd <= 0
        - current_exposure + position_usd > max_total_usd
        - open_positions is provided and open_positions.get(token_id, 0)
          >= max_open_per_token
        """
        if self.position_usd <= 0:
            return False
        if current_exposure + self.position_usd > self.max_total_usd:
            return False
        if open_positions is not None:
            if open_positions.get(signal.token_id, 0) >= self.max_open_per_token:
                return False
        return True

    def compute_size(
        self,
        signal:           "Signal",
        current_exposure: float,
        open_positions:   dict[str, int] | None = None,
    ) -> float:
        """Return the actual USDC amount to allocate.

        Clips to the per-position cap and the remaining headroom under
        the portfolio cap.
        """
        headroom = max(0.0, self.max_total_usd - current_exposure)
        size     = min(self.position_usd, self.max_position_usd, headroom)
        return size
