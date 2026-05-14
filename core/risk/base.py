"""BaseRiskModel ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Signal


class BaseRiskModel(ABC):
    """Abstract base class for position-sizing and signal approval."""

    @abstractmethod
    def approve_signal(
        self,
        signal:           "Signal",
        current_exposure: float,
        open_positions:   dict[str, int] | None = None,
    ) -> bool:
        """Return True if the signal should be sent to the exchange.

        Parameters
        ----------
        signal:
            Signal to evaluate.
        current_exposure:
            Total open USDC exposure across all positions.
        open_positions:
            Mapping of token_id -> open-order count (pending|filled).
            When provided, implementations may reject signals for tokens
            already at their position limit.  Defaults to None (no check).
        """

    @abstractmethod
    def compute_size(
        self,
        signal:           "Signal",
        current_exposure: float,
        open_positions:   dict[str, int] | None = None,
    ) -> float:
        """Return the USDC size to allocate for this signal.

        Must be called only after ``approve_signal()`` returns True.
        """
