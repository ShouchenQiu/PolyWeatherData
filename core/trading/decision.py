"""Pure inventory-aware trading decision kernel."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.pricing.fees import taker_fee_usdc
from core.pricing.quote import compute_quote
from core.types import DirectionalSignal


@dataclass(frozen=True)
class FeeSchedule:
    enabled: bool = False
    rate: float | None = None
    exponent: float | None = 1.0


@dataclass
class PortfolioState:
    """Minimal mutable portfolio state shared by execution environments."""

    cash_usd: float
    shares_by_token: dict[str, float] = field(default_factory=dict)
    notional_by_token: dict[str, float] = field(default_factory=dict)
    event_by_token: dict[str, str] = field(default_factory=dict)
    gross_exposure_usd: float | None = None
    event_exposure_by_id: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TradingDecision:
    approved: bool
    token_id: str | None
    side: str | None
    size_usd: float
    executable_price: float | None
    edge: float
    reason: str | None
    fee_usd: float = 0.0
    cash_delta_usd: float = 0.0


class TradingDecisionKernel:
    """Resolve directional signals into executable, no-short intents."""

    def __init__(
        self,
        *,
        min_edge: float = 0.04,
        buy_size_usd: float = 50.0,
        max_gross_usd: float = 500.0,
        max_event_usd: float = 150.0,
    ) -> None:
        self.min_edge = float(min_edge)
        self.buy_size_usd = float(buy_size_usd)
        self.max_gross_usd = float(max_gross_usd)
        self.max_event_usd = float(max_event_usd)

    def decide(
        self,
        signal: DirectionalSignal,
        books: dict[str, dict],
        portfolio: PortfolioState,
        *,
        fee_schedule: FeeSchedule = FeeSchedule(),
    ) -> TradingDecision:
        if signal.direction == "bullish":
            opposing_token = signal.no_token_id
            buy_token = signal.yes_token_id
            buy_probability = signal.model_prob
        elif signal.direction == "bearish":
            opposing_token = signal.yes_token_id
            buy_token = signal.no_token_id
            buy_probability = 1.0 - signal.model_prob
        else:
            return self._reject(signal, "invalid_direction")

        held_shares = float(
            portfolio.shares_by_token.get(opposing_token, 0.0)
        )
        if opposing_token and held_shares > 0:
            quote = _sell_all_quote(books.get(opposing_token, {}), held_shares)
            if quote is None:
                return self._reject(signal, "insufficient_depth")
            proceeds, vwap = quote
            model_probability = (
                1.0 - signal.model_prob
                if opposing_token == signal.no_token_id
                else signal.model_prob
            )
            edge = vwap - model_probability
            shares = proceeds / vwap
            fee = taker_fee_usdc(
                shares=shares,
                price=vwap,
                fees_enabled=fee_schedule.enabled,
                fee_rate=fee_schedule.rate,
                fee_exponent=fee_schedule.exponent,
            )
            net_proceeds = proceeds - fee
            edge = (net_proceeds / shares) - model_probability
            return TradingDecision(
                approved=True,
                token_id=opposing_token,
                side="SELL",
                size_usd=proceeds,
                executable_price=vwap,
                edge=edge,
                reason=None,
                fee_usd=fee,
                cash_delta_usd=net_proceeds,
            )

        if not buy_token:
            return self._reject(signal, "token_unavailable")
        size_usd = self.buy_size_usd
        if portfolio.cash_usd < size_usd:
            return self._reject(signal, "insufficient_cash")
        gross = (
            float(portfolio.gross_exposure_usd)
            if portfolio.gross_exposure_usd is not None
            else sum(
                max(0.0, float(value))
                for value in portfolio.notional_by_token.values()
            )
        )
        if gross + size_usd > self.max_gross_usd:
            return self._reject(signal, "gross_cap")
        event_id = portfolio.event_by_token.get(
            buy_token,
            signal.condition_id,
        )
        event_exposure = portfolio.event_exposure_by_id.get(event_id)
        if event_exposure is None:
            event_exposure = sum(
                max(0.0, float(notional))
                for token_id, notional in portfolio.notional_by_token.items()
                if portfolio.event_by_token.get(token_id) == event_id
            )
        if event_exposure + size_usd > self.max_event_usd:
            return self._reject(signal, "event_cap")
        try:
            quote = compute_quote(
                books.get(buy_token, {}),
                "BUY",
                size_usd,
            )
        except (KeyError, TypeError, ValueError):
            return self._reject(signal, "quote_error")
        if not quote.complete_fill:
            return self._reject(signal, "insufficient_depth")
        fee = taker_fee_usdc(
            shares=quote.filled_shares,
            price=quote.vwap_price,
            fees_enabled=fee_schedule.enabled,
            fee_rate=fee_schedule.rate,
            fee_exponent=fee_schedule.exponent,
        )
        total_cost = size_usd + fee
        if portfolio.cash_usd < total_cost:
            return self._reject(signal, "insufficient_cash")
        effective_price = total_cost / quote.filled_shares
        edge = buy_probability - effective_price
        if edge < self.min_edge:
            return self._reject(signal, "stale_edge", edge=edge)
        return TradingDecision(
            approved=True,
            token_id=buy_token,
            side="BUY",
            size_usd=size_usd,
            executable_price=quote.vwap_price,
            edge=edge,
            reason=None,
            fee_usd=fee,
            cash_delta_usd=-total_cost,
        )

    @staticmethod
    def _reject(
        signal: DirectionalSignal,
        reason: str,
        *,
        edge: float = 0.0,
    ) -> TradingDecision:
        return TradingDecision(
            approved=False,
            token_id=None,
            side=None,
            size_usd=0.0,
            executable_price=None,
            edge=edge,
            reason=reason,
        )


def _sell_all_quote(
    book: dict,
    shares: float,
) -> tuple[float, float] | None:
    remaining = shares
    proceeds = 0.0
    sold = 0.0
    for level in sorted(
        book.get("bids", []),
        key=lambda item: float(item["price"]),
        reverse=True,
    ):
        available = max(0.0, float(level["size"]))
        taken = min(remaining, available)
        price = float(level["price"])
        proceeds += taken * price
        sold += taken
        remaining -= taken
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or sold <= 0:
        return None
    return proceeds, proceeds / sold
