# TradingLoop halts new orders when forecast data is stale; does not close existing positions

When all HRRR sources fail and ForecastLoop cannot produce a fresh ProbabilityStore update,
TradingLoop detects staleness via `now - last_tick_utc > staleness_threshold_min` and stops
submitting new orders for the duration of the outage. Existing open positions are left
untouched ¡ª they were entered on valid data and closing them speculatively would incur
unnecessary slippage. When ForecastLoop recovers and writes a fresh tick, TradingLoop
resumes automatically. An `on_data_stale(station, age_min)` alert event is emitted on
each TradingLoop tick while halted, wired to the UI status banner and a future Telegram
bot hook.

**Considered option (A ¡ª graduated response)**: Continue trading but widen `min_edge`
by `staleness_edge_bump` per stale tick, reducing order frequency without a hard stop.
Rejected because a widened threshold still allows orders on stale probability estimates;
in a fast-moving market the edge calculation may be systematically wrong, not just
noisier, so a hard halt is the correct conservative response.

**Consequence**: `staleness_threshold_min` (default 65 min) must be tuned to be longer
than the worst-case ForecastLoop latency (HRRR data arrives ~51 min past the hour plus
processing time) but short enough to catch genuine outages. The UI and Telegram hook
must implement the `on_data_stale` interface to surface the alert to the operator.
