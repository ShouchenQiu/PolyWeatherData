"""Stateful 5-minute point-in-time market replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.synthetic_quotes import build_synthetic_book
from core.pricing.quote import compute_quote
from core.trading.decision import (
    FeeSchedule,
    PortfolioState,
    TradingDecisionKernel,
)
from core.types import DirectionalSignal


@dataclass(frozen=True)
class ReplayStrategy:
    name: str
    max_lead_hours: float | None = None
    max_spread: float | None = None
    min_capacity_usd: float | None = None

    @classmethod
    def baseline(cls) -> "ReplayStrategy":
        return cls(name="baseline")

    @classmethod
    def near_term(cls) -> "ReplayStrategy":
        return cls(name="near_term", max_lead_hours=24.0)

    @classmethod
    def liquid(cls) -> "ReplayStrategy":
        return cls(
            name="liquid",
            max_spread=0.08,
            min_capacity_usd=50.0,
        )


@dataclass(frozen=True)
class ReplayResult:
    trades: pd.DataFrame
    settlements: pd.DataFrame
    starting_cash_usd: float
    ending_cash_usd: float
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)


class PointInTimeReplay:
    """Replay cached forecasts against exact-tick executable quote proxies."""

    def __init__(
        self,
        *,
        forecast_cache: ForecastCache,
        quotes: pd.DataFrame,
        markets: pd.DataFrame,
        station_id: str,
        kernel: TradingDecisionKernel | None = None,
        strategy: ReplayStrategy | None = None,
        starting_cash_usd: float = 500.0,
    ) -> None:
        self.forecast_cache = forecast_cache
        self.quotes = quotes.copy()
        self.quotes["timestamp"] = pd.to_datetime(
            self.quotes["timestamp"], utc=True
        )
        self._quotes_by_timestamp = {
            _utc(timestamp): group.reset_index(drop=True)
            for timestamp, group in self.quotes.groupby(
                "timestamp",
                sort=False,
            )
        }
        self.markets = markets.copy()
        self._market_by_condition = {
            str(row.condition_id): row._asdict()
            for row in self.markets.itertuples(index=False)
        }
        self.station_id = station_id
        self.kernel = kernel or TradingDecisionKernel()
        self.strategy = strategy or ReplayStrategy.baseline()
        self.starting_cash_usd = float(starting_cash_usd)
        self._decision_rows: list[dict] = []
        self._visible_cache_at: pd.Timestamp | None = None
        self._visible_cache_until: pd.Timestamp | None = None
        self._visible_cache = pd.DataFrame()
        self._visible_by_condition: dict[str, dict] = {}
        self._last_mid_by_condition: dict[str, float] = {}
        self._token_mark: dict[str, tuple[str, bool]] = {}
        event_by_token: dict[str, str] = {}
        for market in self.markets.itertuples(index=False):
            event_id = str(market.event_id)
            condition_id = str(market.condition_id)
            yes_token_id = str(market.yes_token_id)
            no_token_id = str(market.no_token_id)
            event_by_token[yes_token_id] = event_id
            event_by_token[no_token_id] = event_id
            self._token_mark[yes_token_id] = (condition_id, True)
            self._token_mark[no_token_id] = (condition_id, False)
        self.portfolio = PortfolioState(
            cash_usd=float(starting_cash_usd),
            event_by_token=event_by_token,
        )

    def tick(self, decision_time: str | pd.Timestamp) -> pd.DataFrame:
        """Evaluate and apply all executable signals at one decision tick."""
        tick = _utc(decision_time)
        visible = self._visible_probabilities(tick)
        tick_quotes = self._quotes_by_timestamp.get(
            tick,
            self.quotes.iloc[0:0],
        )
        for quote in tick_quotes.itertuples(index=False):
            self._last_mid_by_condition[str(quote.condition_id)] = (
                float(quote.bid_price_yes) + float(quote.ask_price_yes)
            ) / 2.0
        if not self._visible_by_condition or tick_quotes.empty:
            return _empty_ledger()
        evidence: list[SimpleNamespace] = []
        for quote in tick_quotes.itertuples(index=False):
            condition_id = str(quote.condition_id)
            forecast = self._visible_by_condition.get(condition_id)
            market = self._market_by_condition.get(condition_id)
            if forecast is None or market is None:
                continue
            if str(forecast["target_date"]) != str(market["target_date"]):
                continue
            evidence.append(
                SimpleNamespace(
                    **{
                        **forecast,
                        **market,
                        **quote._asdict(),
                    }
                )
            )
        candidates: list[
            tuple[
                float,
                DirectionalSignal,
                dict[str, dict],
                FeeSchedule,
            ]
        ] = []
        for row in evidence:
            market_end_at = getattr(row, "market_end_at", None)
            if (
                market_end_at is not None
                and not pd.isna(market_end_at)
                and tick >= _utc(market_end_at)
            ):
                self._record_filter(row, tick, "market_closed")
                continue
            if self.strategy.max_lead_hours is not None:
                if market_end_at is None or pd.isna(market_end_at):
                    self._record_filter(row, tick, "missing_market_end")
                    continue
                lead_hours = (
                    _utc(market_end_at) - tick
                ).total_seconds() / 3600.0
                if not 0.0 <= lead_hours <= self.strategy.max_lead_hours:
                    self._record_filter(row, tick, "lead_time_filter")
                    continue
            if (
                self.strategy.max_spread is not None
                and float(row.estimated_spread) > self.strategy.max_spread
            ):
                self._record_filter(row, tick, "spread_filter")
                continue
            if (
                self.strategy.min_capacity_usd is not None
                and float(row.capacity_usdc)
                < self.strategy.min_capacity_usd
            ):
                self._record_filter(row, tick, "capacity_filter")
                continue
            yes_book = build_synthetic_book(vars(row), outcome="YES")
            no_book = build_synthetic_book(vars(row), outcome="NO")
            books = {
                str(row.yes_token_id): yes_book,
                str(row.no_token_id): no_book,
            }
            yes_quote = compute_quote(
                yes_book,
                "BUY",
                self.kernel.buy_size_usd,
            )
            no_quote = compute_quote(
                no_book,
                "BUY",
                self.kernel.buy_size_usd,
            )
            probability = float(row.yes_probability)
            yes_edge = (
                probability - yes_quote.vwap_price
                if yes_quote.complete_fill
                else float("-inf")
            )
            no_edge = (
                (1.0 - probability) - no_quote.vwap_price
                if no_quote.complete_fill
                else float("-inf")
            )
            best_edge = max(yes_edge, no_edge)
            if best_edge < self.kernel.min_edge:
                self._record_filter(row, tick, "edge_filter")
                continue
            direction = "bullish" if yes_edge >= no_edge else "bearish"
            signal = DirectionalSignal(
                yes_token_id=str(row.yes_token_id),
                no_token_id=str(row.no_token_id),
                direction=direction,
                model_prob=probability,
                reference_price=(
                    yes_quote.vwap_price
                    if yes_quote.complete_fill
                    else float(row.ask_price_yes)
                ),
                edge=best_edge,
                condition_id=str(row.condition_id),
            )
            candidates.append(
                (
                    best_edge,
                    signal,
                    books,
                    FeeSchedule(
                        enabled=bool(getattr(row, "fees_enabled", False)),
                        rate=getattr(row, "fee_rate", None),
                        exponent=getattr(row, "fee_exponent", 1.0),
                    ),
                )
            )

        ledger_rows: list[dict] = []
        for _edge, signal, books, fee_schedule in sorted(
            candidates,
            key=lambda candidate: candidate[0],
            reverse=True,
        ):
            decision = self.kernel.decide(
                signal,
                books,
                self.portfolio,
                fee_schedule=fee_schedule,
            )
            self._decision_rows.append(
                {
                    "timestamp": tick,
                    "strategy": self.strategy.name,
                    "condition_id": signal.condition_id,
                    "direction": signal.direction,
                    "model_probability": signal.model_prob,
                    "signal_edge": signal.edge,
                    "approved": decision.approved,
                    "reason": decision.reason,
                    "token_id": decision.token_id,
                    "side": decision.side,
                    "size_usd": decision.size_usd,
                    "executable_price": decision.executable_price,
                    "fee_usd": decision.fee_usd,
                    "executable_edge": decision.edge,
                }
            )
            if not decision.approved:
                continue
            price = float(decision.executable_price)
            shares = decision.size_usd / price
            token_id = str(decision.token_id)
            if decision.side == "BUY":
                self.portfolio.cash_usd += decision.cash_delta_usd
                self.portfolio.shares_by_token[token_id] = (
                    self.portfolio.shares_by_token.get(token_id, 0.0) + shares
                )
                self.portfolio.notional_by_token[token_id] = (
                    self.portfolio.notional_by_token.get(token_id, 0.0)
                    + decision.size_usd
                )
            else:
                held = self.portfolio.shares_by_token.get(token_id, 0.0)
                self.portfolio.shares_by_token[token_id] = max(
                    0.0,
                    held - shares,
                )
                self.portfolio.notional_by_token[token_id] = 0.0
                self.portfolio.cash_usd += decision.cash_delta_usd
            ledger_rows.append(
                {
                    "timestamp": tick,
                    "condition_id": signal.condition_id,
                    "token_id": token_id,
                    "side": decision.side,
                    "size_usd": decision.size_usd,
                    "shares": shares,
                    "price": price,
                    "edge": decision.edge,
                    "fee_usd": decision.fee_usd,
                    "cash_after_usd": self.portfolio.cash_usd,
                }
            )
        return pd.DataFrame(ledger_rows, columns=_LEDGER_COLUMNS)

    def run(self, *, actual_highs: dict[str, float]) -> ReplayResult:
        """Replay all quote ticks and settle each event at its market close."""
        trades: list[pd.DataFrame] = []
        settlements: list[pd.DataFrame] = []
        equity_rows: list[dict] = []
        event_rows = (
            self.markets.dropna(subset=["market_end_at"])
            .assign(
                market_end_at=lambda frame: pd.to_datetime(
                    frame["market_end_at"],
                    utc=True,
                )
            )
            .groupby("event_id", as_index=False)
            .agg(
                market_end_at=("market_end_at", "max"),
                target_date=("target_date", "first"),
            )
            .sort_values("market_end_at")
        ) if "market_end_at" in self.markets else pd.DataFrame()
        unsettled = {
            str(row.event_id): row
            for row in event_rows.itertuples(index=False)
        }
        for tick in sorted(self.quotes["timestamp"].unique()):
            tick_at = _utc(tick)
            due = [
                event_id
                for event_id, row in unsettled.items()
                if _utc(row.market_end_at) <= tick_at
            ]
            for event_id in due:
                row = unsettled.pop(event_id)
                target_date = str(row.target_date)
                if target_date not in actual_highs:
                    raise ValueError(
                        f"missing actual high for {target_date}"
                    )
                settlements.append(
                    self.settle_event(
                        event_id,
                        actual_high_f=actual_highs[target_date],
                        settled_at=row.market_end_at,
                    )
                )
            trades.append(self.tick(tick_at))
            equity_rows.append(self._equity_at(tick_at))

        for event_id, row in sorted(
            unsettled.items(),
            key=lambda item: _utc(item[1].market_end_at),
        ):
            target_date = str(row.target_date)
            if target_date not in actual_highs:
                raise ValueError(f"missing actual high for {target_date}")
            settlements.append(
                self.settle_event(
                    event_id,
                    actual_high_f=actual_highs[target_date],
                    settled_at=row.market_end_at,
                )
            )
            equity_rows.append(self._equity_at(_utc(row.market_end_at)))
        return ReplayResult(
            trades=_concat_or_empty(trades, _LEDGER_COLUMNS),
            settlements=_concat_or_empty(
                settlements,
                _SETTLEMENT_COLUMNS,
            ),
            starting_cash_usd=self.starting_cash_usd,
            ending_cash_usd=self.portfolio.cash_usd,
            equity_curve=_rows_frame(
                equity_rows,
                _EQUITY_COLUMNS,
                sort_by="timestamp",
                deduplicate_by="timestamp",
            ),
            decisions=_rows_frame(
                self._decision_rows,
                _DECISION_COLUMNS,
            ),
        )

    def _record_filter(
        self,
        row: object,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        self._decision_rows.append(
            {
                "timestamp": timestamp,
                "strategy": self.strategy.name,
                "condition_id": str(getattr(row, "condition_id")),
                "direction": None,
                "model_probability": float(
                    getattr(row, "yes_probability")
                ),
                "signal_edge": None,
                "approved": False,
                "reason": reason,
                "token_id": None,
                "side": None,
                "size_usd": 0.0,
                "executable_price": None,
                "fee_usd": 0.0,
                "executable_edge": None,
            }
        )

    def _equity_at(self, timestamp: pd.Timestamp) -> dict:
        positions_value = 0.0
        for token_id, shares in self.portfolio.shares_by_token.items():
            if shares <= 0.0 or token_id not in self._token_mark:
                continue
            condition_id, is_yes = self._token_mark[token_id]
            yes_mid = self._last_mid_by_condition.get(condition_id)
            if yes_mid is None:
                continue
            mark = yes_mid if is_yes else 1.0 - yes_mid
            positions_value += float(shares) * mark
        return {
            "timestamp": timestamp,
            "cash_usd": self.portfolio.cash_usd,
            "positions_value_usd": positions_value,
            "equity_usd": self.portfolio.cash_usd + positions_value,
            "gross_exposure_usd": sum(
                max(0.0, float(value))
                for value in self.portfolio.notional_by_token.values()
            ),
        }

    def _visible_probabilities(self, tick: pd.Timestamp) -> pd.DataFrame:
        if (
            self._visible_cache_at is not None
            and tick >= self._visible_cache_at
            and (
                self._visible_cache_until is None
                or tick < self._visible_cache_until
            )
        ):
            return self._visible_cache
        visible = self.forecast_cache.probabilities_at(
            tick,
            station_id=self.station_id,
        )
        future = self.forecast_cache.snapshots[
            (self.forecast_cache.snapshots["station_id"] == self.station_id)
            & (self.forecast_cache.snapshots["available_at"] > tick)
        ]["available_at"]
        self._visible_cache_at = tick
        self._visible_cache_until = (
            _utc(future.min()) if not future.empty else None
        )
        self._visible_cache = visible
        self._visible_by_condition = {
            str(row.condition_id): row._asdict()
            for row in visible.itertuples(index=False)
        }
        return visible

    def settle_event(
        self,
        event_id: str,
        *,
        actual_high_f: float,
        settled_at: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Redeem all held YES/NO inventory for one resolved weather event."""
        event_markets = self.markets[
            self.markets["event_id"].astype(str) == str(event_id)
        ]
        rows: list[dict] = []
        for market in event_markets.itertuples(index=False):
            in_bucket = (
                float(actual_high_f) >= float(market.bucket_lo_f)
                and float(actual_high_f) < float(market.bucket_hi_f)
            )
            for token_id, outcome_wins in [
                (str(market.yes_token_id), in_bucket),
                (str(market.no_token_id), not in_bucket),
            ]:
                shares = float(
                    self.portfolio.shares_by_token.get(token_id, 0.0)
                )
                if shares <= 0:
                    continue
                cost_basis = float(
                    self.portfolio.notional_by_token.get(token_id, 0.0)
                )
                payout = shares if outcome_wins else 0.0
                self.portfolio.cash_usd += payout
                self.portfolio.shares_by_token[token_id] = 0.0
                self.portfolio.notional_by_token[token_id] = 0.0
                rows.append(
                    {
                        "timestamp": _utc(settled_at),
                        "event_id": str(event_id),
                        "condition_id": str(market.condition_id),
                        "token_id": token_id,
                        "shares": shares,
                        "payout_usd": payout,
                        "cost_basis_usd": cost_basis,
                        "realized_pnl_usd": payout - cost_basis,
                        "cash_after_usd": self.portfolio.cash_usd,
                    }
                )
        return pd.DataFrame(rows, columns=_SETTLEMENT_COLUMNS)


_LEDGER_COLUMNS = [
    "timestamp",
    "condition_id",
    "token_id",
    "side",
    "size_usd",
    "shares",
    "price",
    "edge",
    "fee_usd",
    "cash_after_usd",
]
_SETTLEMENT_COLUMNS = [
    "timestamp",
    "event_id",
    "condition_id",
    "token_id",
    "shares",
    "payout_usd",
    "cost_basis_usd",
    "realized_pnl_usd",
    "cash_after_usd",
]
_EQUITY_COLUMNS = [
    "timestamp",
    "cash_usd",
    "positions_value_usd",
    "equity_usd",
    "gross_exposure_usd",
]
_DECISION_COLUMNS = [
    "timestamp",
    "strategy",
    "condition_id",
    "direction",
    "model_probability",
    "signal_edge",
    "approved",
    "reason",
    "token_id",
    "side",
    "size_usd",
    "executable_price",
    "fee_usd",
    "executable_edge",
]


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=_LEDGER_COLUMNS)


def _concat_or_empty(
    frames: list[pd.DataFrame],
    columns: list[str],
) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns)
    return pd.concat(nonempty, ignore_index=True)


def _rows_frame(
    rows: list[dict],
    columns: list[str],
    *,
    sort_by: str | None = None,
    deduplicate_by: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if deduplicate_by is not None and not frame.empty:
        frame = frame.drop_duplicates(subset=[deduplicate_by], keep="last")
    if sort_by is not None and not frame.empty:
        frame = frame.sort_values(sort_by)
    return frame.reset_index(drop=True)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
