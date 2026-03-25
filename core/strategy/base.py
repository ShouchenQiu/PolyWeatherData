"""BaseStrategy ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def compute_signals(
        self,
        probs:   dict[tuple[float, float], float],
        books:   dict[str, dict],
        markets: list[dict],
    ) -> list["Signal"]:
        """Compute trading signals for the current market state.

        Parameters
        ----------
        probs:
            ``{(lo_f, hi_f): probability}`` from ``McSampler.bucket_probs()``.
        books:
            ``{token_id: {"mid_price": float, ...}}`` live CLOB quotes.
        markets:
            List of market dicts from the DB; each has at minimum
            ``token_id``, ``bucket_lo_f``, ``bucket_hi_f``, ``outcome``.

        Returns
        -------
        List of ``Signal`` instances to be passed to the risk / exchange layer.
        """
