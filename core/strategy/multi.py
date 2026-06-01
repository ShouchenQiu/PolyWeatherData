"""MultiStrategy: fan out to N strategies, deduplicate by token_id."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.strategy.base import BaseStrategy

if TYPE_CHECKING:
    from core.types import DirectionalSignal


class MultiStrategy(BaseStrategy):
    """Composite strategy that delegates to an ordered list of sub-strategies.

    Signals are collected in order; when two strategies emit a signal for the
    same token_id the first one wins (deduplication by token_id).

    Parameters
    ----------
    strategies:
        Non-empty list of BaseStrategy instances.  Evaluated in list order.

    Raises
    ------
    ValueError
        If ``strategies`` is empty.
    """

    def __init__(self, strategies: list[BaseStrategy]) -> None:
        if not strategies:
            raise ValueError(
                "MultiStrategy requires at least one sub-strategy."
            )
        self._strategies = strategies

    @property
    def name(self) -> str:
        return "multi"

    def compute_signals(
        self,
        probs:          dict[tuple[float, float], float],
        books:          dict[str, dict],
        markets:        list[dict],
        open_positions: dict[str, int] | None = None,
    ) -> list["DirectionalSignal"]:
        """Collect and deduplicate signals from all sub-strategies.

        Parameters
        ----------
        probs:
            ``{(lo_f, hi_f): probability}`` from ``McSampler.bucket_probs()``.
        books:
            ``{token_id: L2 book dict}`` live CLOB depth.
        markets:
            Market rows from the DB.
        open_positions:
            Forwarded to each sub-strategy unchanged.

        Returns
        -------
        Deduplicated list of DirectionalSignal objects; first strategy wins on
        yes_token_id collision.
        """
        seen: dict = {}
        for strategy in self._strategies:
            for sig in strategy.compute_signals(
                probs, books, markets, open_positions=open_positions
            ):
                key = sig.yes_token_id
                if key and key not in seen:
                    seen[key] = sig
        return list(seen.values())
