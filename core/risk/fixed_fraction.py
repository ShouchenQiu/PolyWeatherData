"""Fixed-fraction position sizing model.

Allocates a fixed USDC amount per BUY signal, subject to two hard caps:
  - max_position_usd : maximum USDC per individual position.
  - max_total_usd    : maximum total open USDC exposure (portfolio cap).

For SELL signals the size is derived from the net open shares held, scaled
by max_exit_fraction (defaults to 1.0 = full exit).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.risk.base import BaseRiskModel

if TYPE_CHECKING:
    from core.types import Signal


class FixedFraction(BaseRiskModel):
    """Size each BUY at a fixed USDC amount; size each SELL from held shares.

    Parameters
    ----------
    position_usd:
        Target USDC size per BUY signal.
    max_position_usd:
        Hard cap on any single BUY position (defaults to position_usd).
    max_total_usd:
        Portfolio cap: total open exposure must not exceed this value.
    max_exit_fraction:
        Fraction of held shares to sell on a SELL signal (0..1].
        Defaults to 1.0 (full exit).
    """

    def __init__(
        self,
        position_usd:     float = 20.0,
        max_position_usd: float | None = None,
        max_total_usd:    float = 500.0,
        max_exit_fraction: float = 1.0,
    ) -> None:
        self.position_usd      = position_usd
        self.max_position_usd  = max_position_usd if max_position_usd is not None else position_usd
        self.max_total_usd     = max_total_usd
        self.max_exit_fraction = max_exit_fraction

    @classmethod
    def from_settings(cls, settings: object) -> "FixedFraction":
        return cls(
            position_usd      = getattr(settings, "max_position_usd",   20.0),
            max_position_usd  = getattr(settings, "max_position_usd",   20.0),
            max_total_usd     = getattr(settings, "max_total_usd",     500.0),
            max_exit_fraction = getattr(settings, "max_exit_fraction",   1.0),
        )

    def approve_signal(
        self,
        signal:           "Signal",
        current_exposure: float,
    ) -> bool:
        """Return False unless the signal passes position-level checks.

        BUY signals are rejected when:
        - position_usd <= 0
        - current_exposure + position_usd > max_total_usd

        SELL signals bypass the exposure gate entirely; they are always
        approved here.  compute_size() returns 0.0 if there are no net
        shares to sell, which prevents a zero-size order downstream.
        """
        if signal.side == "SELL":
            return True
        if self.position_usd <= 0:
            return False
        if current_exposure + self.position_usd > self.max_total_usd:
            return False
        return True

    def compute_size(
        self,
        signal:           "Signal",
        current_exposure: float,
        positions:        object | None = None,
    ) -> float:
        """Return the USDC amount to allocate.

        For BUY signals: clips to the per-position cap and the remaining
        headroom under the portfolio cap.

        For SELL signals: returns
            max_exit_fraction * net_shares * price
        where net_shares comes from ``positions.position_shares()`` when
        *positions* is provided.  Returns 0.0 if net_shares <= 0 or
        *positions* is not provided.
        """
        if signal.side == "SELL":
            if positions is None:
                return 0.0
            net_shares = positions.position_shares(signal.token_id)
            if net_shares <= 0:
                return 0.0
            price = signal.executable_price if signal.executable_price is not None else signal.clob_price
            return self.max_exit_fraction * net_shares * price

        headroom = max(0.0, self.max_total_usd - current_exposure)
        size     = min(self.position_usd, self.max_position_usd, headroom)
        return size
