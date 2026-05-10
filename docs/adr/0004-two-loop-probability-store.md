# ForecastLoop and TradingLoop are separate coroutines sharing an in-process ProbabilityStore

The production pipeline runs two independent asyncio loops: **ForecastLoop** (1-hour
cadence) handles all data ingestion, feature engineering, TFT inference, and MC sampling;
**TradingLoop** (5-minute cadence, interval configurable via `trading_loop_interval_s`)
handles order-book fetching, strategy evaluation, risk gating, and order submission.
They share state through a single in-process **ProbabilityStore** object written by
ForecastLoop and read by TradingLoop. ProbabilityStore holds, per
`condition_id x day_offset x bucket`: the MC bucket probability, the TFT p10/p50/p90
quantiles for each decoder hour, and `last_tick_utc` for staleness computation.
TradingLoop uses `now - last_tick_utc` to determine whether to halt new order
submission when the forecast is stale [ADR-0006].

**Considered option (Plan B)**: Persist bucket probabilities to SQLite between loops.
Rejected because both loops run in the same process, the DB round-trip adds latency
with no benefit, and staleness tracking becomes more complex without a live timestamp
object.

**Consequence**: ForecastLoop and TradingLoop must be co-located in the same process.
If the architecture is ever split into separate services, ProbabilityStore must be
replaced with an external store (Redis, SQLite, or similar) at that point.
