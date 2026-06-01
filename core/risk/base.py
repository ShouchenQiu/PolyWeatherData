"""BaseRiskModel ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import OrderIntent


class BaseRiskModel(ABC):
    """Abstract base class for position-sizing and signal approval."""

    @abstractmethod
    def approve_signal(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
    ) -> bool:
        """Return True if the signal should be sent to the exchange.

        Parameters
        ----------
        signal:
            OrderIntent to evaluate.
        current_exposure:
            Total open USDC exposure across all positions.
        """

    @abstractmethod
    def compute_size(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
        positions:        object | None = None,
    ) -> float:
        """Return the USDC size to allocate for this signal.

        Must be called only after ``approve_signal()`` returns True.

        Parameters
        ----------
        positions:
            Optional PositionTracker used for SELL exit sizing.
            When provided, SELL sizes are derived from
            ``positions.position_shares()`` rather than a fixed USD amount.
        """
