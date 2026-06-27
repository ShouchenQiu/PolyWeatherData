"""Edge-based trading strategy.

For each YES bucket token:
    reference_price = compute_quote(book, "BUY", quote_size_usd).vwap_price
    edge = mc_prob - reference_price

    edge >  min_edge  =>  DirectionalSignal(direction="bullish")
    edge < -min_edge  =>  DirectionalSignal(direction="bearish")

Bearish means YES is overpriced; TradingLoop decides SELL YES vs BUY NO.
Both YES and NO token_ids are carried in the signal for arbitration.

Reference: Herbie-main/_sniper_backtest.py  (edge = mc_p - cp)
Threshold: min_edge = 0.04 (configurable via Settings.min_edge)
"""
from __future__ import annotations

from core.pricing.quote import compute_quote
from core.strategy.base import BaseStrategy
from core.types import DirectionalSignal


class EdgeStrategy(BaseStrategy):
    """Generate directional mispricing signals based on model-vs-VWAP edge.

    Parameters
    ----------
    min_edge:
        Minimum absolute edge required to generate a signal.
    quote_size_usd:
        Proxy order size passed to compute_quote for VWAP estimation.
    """

    def __init__(self, min_edge: float = 0.04, quote_size_usd: float = 50.0) -> None:
        self.min_edge      = min_edge
        self.quote_size_usd = quote_size_usd

    @property
    def name(self) -> str:
        return "edge"

    def compute_signals(
        self,
        probs:          dict[tuple[float, float], float],
        books:          dict[str, dict],
        markets:        list[dict],
        open_positions: dict[str, int] | None = None,
    ) -> list[DirectionalSignal]:
        """Return directional signals for buckets whose YES VWAP edge clears the threshold.

        Parameters
        ----------
        probs:
            ``{(lo_f, hi_f): probability}`` from ``McSampler.bucket_probs()``.
        books:
            ``{token_id: {"bids": [...], "asks": [...]}}`` full L2 order books.
            Tokens absent from this dict are skipped.
        markets:
            Market rows from the DB; YES and NO rows for the same bucket are
            paired to populate both token IDs in the emitted signal.
        open_positions:
            Accepted for interface compatibility; not used by EdgeStrategy.

        Returns
        -------
        Sorted list of ``DirectionalSignal`` objects (by edge descending).
        """
        bucket_yes:  dict[tuple[float, float], str] = {}
        bucket_no:   dict[tuple[float, float], str] = {}
        bucket_cond: dict[tuple[float, float], str] = {}

        for mkt in markets:
            outcome = str(mkt.get("outcome", "Yes")).upper()
            lo_raw  = mkt.get("bucket_lo_f")
            hi_raw  = mkt.get("bucket_hi_f")
            lo_f    = float(lo_raw) if lo_raw is not None else float("-inf")
            hi_f    = float(hi_raw) if hi_raw is not None else float("inf")
            tid     = str(mkt["token_id"])
            cond    = str(mkt.get("condition_id", ""))
            if outcome == "YES":
                bucket_yes[(lo_f, hi_f)]  = tid
                bucket_cond[(lo_f, hi_f)] = cond
            elif outcome == "NO":
                bucket_no[(lo_f, hi_f)] = tid

        signals: list[DirectionalSignal] = []

        for bucket, yes_token_id in bucket_yes.items():
            lo_f, hi_f = bucket
            mc_prob = probs.get(bucket)
            if mc_prob is None:
                continue

            book = books.get(yes_token_id)
            if book is None:
                continue

            try:
                quote = compute_quote(book, "BUY", self.quote_size_usd)
            except Exception:
                continue
            if not quote.complete_fill:
                continue

            reference_price = quote.vwap_price
            yes_edge        = mc_prob - reference_price
            no_token_id     = bucket_no.get(bucket, "")
            condition_id    = bucket_cond.get(bucket, "")

            if yes_edge > self.min_edge:
                signals.append(DirectionalSignal(
                    yes_token_id    = yes_token_id,
                    no_token_id     = no_token_id,
                    direction       = "bullish",
                    model_prob      = mc_prob,
                    reference_price = reference_price,
                    edge            = yes_edge,
                    bucket_lo_f     = lo_f,
                    bucket_hi_f     = hi_f,
                    condition_id    = condition_id,
                ))
            elif yes_edge < -self.min_edge:
                signals.append(DirectionalSignal(
                    yes_token_id    = yes_token_id,
                    no_token_id     = no_token_id,
                    direction       = "bearish",
                    model_prob      = mc_prob,
                    reference_price = reference_price,
                    edge            = abs(yes_edge),
                    bucket_lo_f     = lo_f,
                    bucket_hi_f     = hi_f,
                    condition_id    = condition_id,
                ))

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals
