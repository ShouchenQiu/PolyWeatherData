"""BaseStrategy ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @abstractmethod
    def compute_signals(
        self,
        probs:          dict[tuple[float, float], float],
        books:          dict[str, dict],
        markets:        list[dict],
        open_positions: dict[str, int] | None = None,
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
        open_positions:
            Mapping of token_id -> open-order count (pending|filled).
            Passed through to implementations that enforce per-token caps.
            Defaults to None (no per-token check at the strategy layer).

        Returns
        -------
        List of ``Signal`` instances to be passed to the risk / exchange layer.
        """
