"""Edge-based trading strategy.

For each YES market token:
    edge = mc_prob - clob_price

    edge >  min_edge  =>  BUY signal
    edge < -min_edge  =>  SELL signal  (used to unwind existing positions)

For each NO market token:
    NO edge = (1 - mc_prob) - clob_price_no

    NO edge >  min_edge  =>  BUY signal on the NO token
    NO edge < -min_edge  =>  SELL signal on the NO token

Reference: Herbie-main/_sniper_backtest.py  (edge = mc_p - cp)
Threshold: min_edge = 0.04 (configurable via Settings.min_edge)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.strategy.base import BaseStrategy
from core.types import Signal

if TYPE_CHECKING:
    pass


class EdgeStrategy(BaseStrategy):
    """Generate BUY / SELL signals based on model-vs-market edge.

    Parameters
    ----------
    min_edge:
        Minimum absolute edge required to generate a signal.
        Default matches the project constant (0.04).
    """

    def __init__(self, min_edge: float = 0.04) -> None:
        self.min_edge = min_edge

    @property
    def name(self) -> str:
        return "edge"

    def compute_signals(
        self,
        probs:          dict[tuple[float, float], float],
        books:          dict[str, dict],
        markets:        list[dict],
        open_positions: dict[str, int] | None = None,
    ) -> list[Signal]:
        """Return signals for markets whose edge exceeds the threshold.

        YES and NO outcome tokens are both evaluated.  For YES tokens,
        edge = mc_prob - clob_price.  For NO tokens,
        edge = (1 - mc_prob) - clob_price_no.

        Parameters
        ----------
        probs:
            ``{(lo_f, hi_f): probability}`` from ``McSampler.bucket_probs()``.
        books:
            ``{token_id: {"mid_price": float, ...}}`` live CLOB quotes.
            Tokens absent from this dict are skipped.
        markets:
            Market rows from the DB.
        open_positions:
            Accepted for interface compatibility; not used by EdgeStrategy.

        Returns
        -------
        Sorted list of ``Signal`` objects (by absolute edge descending).
        """
        signals: list[Signal] = []

        for mkt in markets:
            outcome = str(mkt.get("outcome", "Yes")).upper()
            token_id    = str(mkt["token_id"])
            lo_f        = float(mkt.get("bucket_lo_f") or float("-inf"))
            hi_f        = float(mkt.get("bucket_hi_f") or float("inf"))
            condition_id = str(mkt.get("condition_id", ""))

            mc_prob = probs.get((lo_f, hi_f))
            if mc_prob is None:
                continue

            book = books.get(token_id)
            if book is None:
                continue
            mid = book.get("mid_price")
            if mid is None:
                continue

            clob_price = float(mid)

            if outcome == "YES":
                edge = mc_prob - clob_price
            elif outcome == "NO":
                edge = (1.0 - mc_prob) - clob_price
            else:
                continue

            if edge > self.min_edge:
                signals.append(Signal(
                    token_id     = token_id,
                    side         = "BUY",
                    mc_prob      = mc_prob,
                    clob_price   = clob_price,
                    edge         = edge,
                    bucket_lo_f  = lo_f,
                    bucket_hi_f  = hi_f,
                    condition_id = condition_id,
                ))
            elif edge < -self.min_edge:
                signals.append(Signal(
                    token_id     = token_id,
                    side         = "SELL",
                    mc_prob      = mc_prob,
                    clob_price   = clob_price,
                    edge         = edge,
                    bucket_lo_f  = lo_f,
                    bucket_hi_f  = hi_f,
                    condition_id = condition_id,
                ))

        # Sort: strongest absolute edge first
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals
