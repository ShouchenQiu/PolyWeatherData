"""Fixed-fraction position sizing model.

Allocates a fixed USDC amount per signal, subject to two hard caps:
  - max_position_usd: maximum USDC per individual position.
  - max_total_usd   : maximum total open USDC exposure (portfolio cap).

Signals are rejected when either cap would be exceeded.
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
    """

    def __init__(
        self,
        position_usd:     float = 20.0,
        max_position_usd: float | None = None,
        max_total_usd:    float = 500.0,
    ) -> None:
        self.position_usd     = position_usd
        self.max_position_usd = max_position_usd if max_position_usd is not None else position_usd
        self.max_total_usd    = max_total_usd

    @classmethod
    def from_settings(cls, settings: object) -> "FixedFraction":
        return cls(
            position_usd     = getattr(settings, "max_position_usd", 20.0),
            max_position_usd = getattr(settings, "max_position_usd", 20.0),
            max_total_usd    = getattr(settings, "max_total_usd",    500.0),
        )

    def approve_signal(
        self,
        signal:           "Signal",
        current_exposure: float,
    ) -> bool:
        """Return False unless there is headroom for a full position_usd trade."""
        if self.position_usd <= 0:
            return False
        if current_exposure + self.position_usd > self.max_total_usd:
            return False
        return True

    def compute_size(
        self,
        signal:           "Signal",
        current_exposure: float,
    ) -> float:
        """Return the actual USDC amount to allocate.

        Clips to the per-position cap and the remaining headroom under
        the portfolio cap.
        """
        headroom = max(0.0, self.max_total_usd - current_exposure)
        size     = min(self.position_usd, self.max_position_usd, headroom)
        return size
