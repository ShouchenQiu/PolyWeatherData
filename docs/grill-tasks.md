# Grill Session -- PolyWeatherData Cleanup Tasks

Each entry = one confirmed decision from grill Q&A.
Append immediately after confirmation; never batch.

## Format rules

```
### path/to/file.py
- VERB  description  [ADR-NNNN]
```

Canonical verbs: ADD / REMOVE / CHANGE / FIX / MOVE / WIRE
New files: `### path/to/newfile.py  [NEW]`

---

## Layer 0: Project Identity

### pyproject.toml
- CHANGE project.name: "polyweatherbot" -> "polyweatherdata"

### CONTEXT.md
- CHANGE title: "PolyWeatherBot Context" -> "PolyWeatherData Context"
- CHANGE purpose: no model inference, no HRRR download, no order submission
- ADD "market probability" term: CLOB-implied YES probability from normalized synthetic midpoint
- ADD "settlement truth" term: IEM daily-high binary outcome
- ADD "mispricing" term: persistent deviation between market probability and settlement-truth rate
- CHANGE settlement truth source: IEM primary, Weather.com supplementary

### config/stations.py
- REMOVE hrrr_grid_row_zarr, hrrr_grid_col_zarr fields from StationConfig
- ADD stations: KORD, KATL, KSEA, KDFW (plus keep KLGA = 5 total)
- KEEP station_id, lat, lon, timezone, tz_offset, iem_network, slug_pattern, slug_aliases

### data/clob/history.py
- CHANGE: use Gamma API public endpoints for historical trades/orderbook (REST pagination)
- CHANGE granularity: 5-minute snapshot mid-price / best bid-ask (not tick-level)
- WIRE ParquetStore: partition by station, compatible with existing data_store/backtest/ layout

### scripts/
### db/schema.py
- KEEP tables: stations, events, markets
- REMOVE tables: signals, orders, model_versions, station_max_bias, bot_state, redeem_tasks, loop_ticks, forecast_snapshots, forecast_bucket_probs, forecast_quantiles
- ADD market_price_snapshots table: (price_snapshot_id INTEGER PK AUTOINCREMENT, token_id TEXT REFERENCES markets, condition_id TEXT REFERENCES events, tick_utc TEXT NOT NULL, best_bid REAL, best_ask REAL, mid_price REAL, UNIQUE(token_id, tick_utc))

### loops/ directory
- REMOVE entire directory (no real-time event loops needed)

### core/bot.py
- REMOVE BotManager class entirely
- CHANGE: to simple CLI entry point

### main.py
- CHANGE: to `streamlit run ui/app.py` entry point

### ui/app.py
- CHANGE: from trading dashboard to mispricing analysis dashboard
- ADD page 1: Mispricing Overview (calibration gap heatmap by station x period)
- ADD page 2: Calendar Stability (rolling-window calibration gap trend)
- ADD page 3: Cross-Station Generalisation (leave-one-out test results)

### scripts/
- ADD run_mispricing_analysis.py [NEW]: CLI entry point for batch mispricing scan

### backtest/
- REMOVE daily_high_pmf_evaluation.py, conditional_residual_evaluation.py
- REMOVE two_bucket_probability_stack.py, two_bucket_union_study.py
- REMOVE peak_margin_crossfit.py, signal_slice_study.py, signal_study.py, edge_study.py
- REMOVE two_bucket_probability_stack_frozen.py, two_bucket_union_frozen_diagnostic.py, peak_union_diagnostics.py
- REMOVE synthetic_quotes.py, market_calibration.py, forecast_cache.py, selection.py, study.py
- KEEP engine.py, analysis.py, reporting.py, observations.py, replay.py

### backtest/analysis.py
- CHANGE: compare "market vs truth" instead of "model vs market"
- ADD calibration_gap() metric: E[truth] - E[market_prob] per stratum
- ADD paired bootstrap CI for calibration gap
- ADD rolling-window stability diagnostic
- KEEP existing proper scores (Brier, log score, RPS, ECE/MCE) adapted for market-vs-truth

### backtest/mispricing_metric.py  [NEW]
- ADD MispricingMetric class: stratum = (station, lead_time_band, calendar_period)
- ADD compute(): returns calibration gap + bootstrap CI per stratum
- ADD generalisability test: leave-one-station-out cross-validation

### docs/archieve/
- KEEP all original engineering conclusion reports (do not delete)

### docs/archieve/
- KEEP all original engineering conclusion reports (do not delete)

- ADD `trading_loop_interval_s: int = 300` (TradingLoop 5-min tick, configurable)
- ADD `staleness_threshold_min: int = 65` (TradingLoop halts new orders when forecast older than this)
- ADD `staleness_edge_bump: float = 0.02` (reserved for future graduated response; not used in halt mode)
- ADD `hrrr_source_priority: list[str] = ["s3", "nomads"]` (production fallback chain; zarr excluded)
- ADD `max_open_per_token: int = 1` (risk: reject new order for a token_id when
  `open_positions_by_token()[token_id] >= max_open_per_token`; "open" = pending OR
  filled; enforced via `PositionTracker.open_positions_by_token()`)
- ADD `cold_start_hours: int = 72` (Cleaner drops first 72 h)
- ADD `enable_sniper: bool = False` (feature flag, reads env ENABLE_SNIPER)
- ADD `train_start: str = "2024-10-01"` (ISO date; inclusive start of training window)
- ADD `train_end: str = "2025-12-31"` (ISO date; inclusive end of training window)
- ADD `val_start: str = "2026-01-01"` (ISO date; inclusive start of validation window)
- ADD `val_end: str = "2026-02-28"` (ISO date; inclusive end of validation window)
- ADD `test_start: str = "2026-03-01"` (ISO date; inclusive start of test window)
- ADD `test_end: str = "2026-04-30"` (ISO date; inclusive end of test window)

### config/stations.py
- ADD remaining target stations: Toronto, Atlanta, Chicago, Seattle, Dallas, Miami
  (row/col values TBD -- need HRRR grid lookup per station)

---

## Layer 1b: db/

### db/schema.py
- ADD `orders` table with full lifecycle fields:
  `(order_id TEXT PK, token_id, condition_id, side, size_usd, price, fill_price,
  status TEXT, created_at, filled_at, closed_at, cancelled_at, settled_payout REAL)`;
  `status` domain: `pending | filled | cancelled | closed | settled`;
  `fill_price`, `filled_at`, `closed_at`, `cancelled_at`, and `settled_payout`
  are NULL until the relevant transition occurs

---

## Layer 2: data/

### data/forecast/hrrr_fill.py  [NEW]
- ADD `HrrrFill` class: splices non-MAJOR 18h HRRR data to 48h using nearest prior
  MAJOR-hour run, aligned by valid_time (not fxx offset)
- ADD tail-gap strategy: for valid_times not covered by nearest prior MAJOR-hour run
  (up to 6 h gap at worst), fall back to the MAJOR-hour before that; log WARNING
  when fallback MAJOR is used  [ADR-0005]

### data/forecast/hrrr_zarr.py
- CHANGE role to batch-download only; remove from production fallback chain
  (too much per-call overhead from opening zarr stores; use scripts/ only)

### data/forecast/_grib_source.py
- REMOVE hardcoded `_klga_idx`; replace with per-station grid-index dict keyed by station_id,
  populated from `STATIONS[station_id].hrrr_grid_row_zarr / hrrr_grid_col_zarr`
- CHANGE fetch(init_time) signature to fetch(init_time) -> dict[station_id, pd.DataFrame]:
  decode all configured stations from a single GRIB2 download in one eccodes pass
- ADD batch multi-station index lookup: build {station_id: (grid_row, grid_col)} from
  STATIONS at init time; extract all grid points in one _fetch_range + eccodes loop
- WIRE `HrrrFill` for non-MAJOR init_times
- ADD auto-switch fallback between sources on download or decode failure (not manual)
- ADD error handling + logging for: _fetch_idx failure, _fetch_range failure, GRIB2 decode failure
- ADD gap-detection for missing init_times (time-series holes)

### data/obs/iem.py
- CHANGE fetch(): IEM publishes obs every 15 min; only the :51 reading is used as the
  representative value for the nearest whole hour (it is the most accurate reading);
  fetch only the :51 timestamp for hour H-1 (e.g. `(now_utc - 1h).replace(minute=51)`),
  do NOT re-download past 72h each tick
- FIX fallback source: switch automatically when data is incomplete, not only when all-NaN
- ADD fine-grained warnings: e.g. when high-precision temp unavailable

### data/obs/noaa_file.py
- ADD METAR parse robustness (multiple numeric fields, incomplete records)
- ADD fine-grained warnings for parse and data anomalies

### data/storage/cleaner.py
- WIRE into data flow (currently isolated)
- OWN all feature engineering: wind component decomposition, log1p smoothing,
  K->C conversion (HRRR), F->C (IEM), temp_error = IEM_obs - f01_HRRR
- DROP cold-start rows (first `cold_start_hours` rows per dataset/station)
- SAVE processed output back to ParquetStore, replacing raw columns

### data/storage/parquet_store.py
- ENFORCE f01-only storage for HRRR (drop all fxx > 1)  [ADR-0003]

### data/manager.py
- REMOVE any feature transforms that duplicate Cleaner's work
- ADD station parameter threading through all fetch calls
- ADD build_encoder_obs_window(station, init_time) -> pd.DataFrame: reads last 72
  hourly rows from ParquetStore -- both f01 HRRR forecast values and IEM observed
  values -- merged on valid_time; no re-download; raises if fewer than
  cold_start_hours rows available for either series

---

## Layer 3: model/

### model/mc.py
- REMOVE AR(1) covariance; REMOVE `bias_corr_c` bias correction  [ADR-0002]
- CHANGE `n_samp` default to 10000
- ADD `fit_cov(residuals: np.ndarray) -> None` method
- FIX `bucket_probs` return: include `p10_f`, `p50_f`, `p90_f` computed across all
  samples together (not per-bucket)
- ADD apply `max_bias` correction: subtract per-station `max_bias` (loaded from registry)
  from E[max(samples)] before mapping to bucket probabilities

### model/tft.py
- CHANGE `MAX_ENCODER_LENGTH` 24 -> 72  [ADR-0001]
- MOVE `_prep()` data processing -> `data/storage/cleaner.py`
- FIX line 277: un-residualise output (add f01 back to get absolute temperature)

### model/pipeline.py
- FIX `fetch_hrrr` call at line 79: pass missing `station` argument
- FIX encoder window: use -71..0 (not -23..0)
- FIX `assemble_frames`: encoder = 72h, merge tolerance = 20 min (not 30 min)
- ADD `time_idx` column construction
- ADD seasonal/periodic time features (month, day-of-year, hour-of-day)
- ADD superset column alignment before encoder/decoder concat
- ADD staleness timestamp to output written to ProbabilityStore

### model/store/registry.py
- FIX `_db_latest_row()`: replace `db._conn` direct access with DbClient public interface
- ADD `store_max_bias(station_id: str, max_bias: float) -> None`: writes per-station
  max_bias to DB
- ADD `load_max_bias(station_id: str) -> float`: returns stored bias; 0.0 if not yet
  calibrated
- ADD `calibrate_max_bias(station_id: str, val_residuals: np.ndarray) -> float`:
  measures E[max(samples)] - mean(observed daily max) on validation set;
  stores result via `store_max_bias`

---

## Layer 4: loops/

### loops/forecast_loop.py
- SPLIT into ForecastLoop (1h) and TradingLoop (5min)  [ADR-0004]
- CHANGE ForecastLoop cadence: wall-clock UTC hour alignment (not sleep(3600))
- ADD write pipeline output to ProbabilityStore after each run: for each active Event
  keyed by (condition_id, day_offset) where day_offset in {0, 1, 2} (today/tomorrow/
  day-after); writes MC prob, TFT p10/p50/p90, last_tick_utc per bucket  [ADR-0004]

### loops/trading_loop.py  [NEW]
- ADD TradingLoop: reads ProbabilityStore, checks staleness, iterates all active Events,
  runs strategy+risk, submits orders to CLOB  [ADR-0004]
- CHANGE TradingLoop cadence: `settings.trading_loop_interval_s` (default 300s)
- ADD YES and NO token handling (currently only YES)
- ADD staleness gate: when `now - last_tick_utc > staleness_threshold_min`, halt all
  new order submission (do not close existing positions); emit alert event  [ADR-0006]
- ADD alert emission interface: `on_data_stale(station, age_min)` emits
  `AlertEvent(alert_type="data_stale", ...)` via `AlertBus`; wired to UI and
  Telegram stub subscriber
- ADD `risk_rejected` alert: emit `AlertEvent(alert_type="risk_rejected", ...)` when
  `RiskPipeline.approve_signal` rejects a signal (structured log; not Telegram)
- ENFORCE TradingLoop holds exactly one `MultiStrategy` instance (never a bare
  `BaseStrategy`); all strategy composition happens inside `MultiStrategy`

### loops/discovery_loop.py
- CHANGE discovery interval: daily -> hourly
- ADD deduplication: skip Events/Markets already in DB
- REMOVE any cutoff-hour logic

### events/discovery.py
- CHANGE parse_slug: replace ad-hoc if/elif branches with an internal registry --
  a module-level list of `(compiled_re, handler: Callable[[re.Match], tuple[float, float]])`
  entries defined in `events/discovery.py`; parser iterates the list, returns result of
  first match, logs WARNING and returns `None` if no pattern matches; new slug formats
  are added by appending to the list, no other code changes required
- ADD registry entries covering all known patterns: `or-above`, `or-higher`,
  `or-below`, `be-{N}f-or-below`, and any variants observed in production slugs

### core/probability_store.py  [NEW]
- ADD `ProbabilityStore`: in-process shared object; store key = (condition_id, day_offset)
  where day_offset in {0, 1, 2} (up to 3 simultaneous active Events); bucket is NOT
  part of the store key -- value per key = dict[bucket_tuple, BucketPayload] where
  bucket_tuple = (lo_f, hi_f) and BucketPayload contains:
    mc_prob: float,
    quantiles_by_fxx: dict[int, {"p10": float, "p50": float, "p90": float}]
      (fxx = decoder hour offset, e.g. 1..48),
    last_tick_utc: datetime  (shared across all buckets for the same key, written
      once per ForecastLoop run);
  asyncio Lock for thread safety  [ADR-0004]

---

## Layer 5: core/

### core/positions.py
- ADD order lifecycle tracking: `record_order(order_id, token_id, condition_id, side,
  size_usd, price, created_at=None) -> None`: writes a `pending` row to `orders`
  table; `created_at` defaults to current UTC if not provided
- ADD `mark_filled(order_id, fill_price, filled_at=None) -> None`: updates
  status -> "filled"; `filled_at` defaults to current UTC if not provided
- ADD `mark_closed(order_id, closed_at=None) -> None`: updates status -> "closed";
  `closed_at` defaults to current UTC if not provided
- ADD `mark_cancelled(order_id, cancelled_at=None) -> None`: updates
  status -> "cancelled"; `cancelled_at` defaults to current UTC if not provided;
  cancelled orders are excluded from `open_positions_by_token()` and `total_exposure()`
- ADD `mark_settled(order_id, settled_payout) -> None`: updates status -> "settled";
  writes `settled_payout`
- ADD `open_positions_by_token() -> dict[str, int]`: counts `pending|filled` rows
  per `token_id`; used by `RiskPipeline` for duplicate-order prevention
- ADD `total_exposure() -> float`: sum of `size_usd` for all `pending|filled` rows
- ADD `exposure_by_condition(condition_id) -> float`: sum of `size_usd` for all
  `pending|filled` rows matching `condition_id`; reserved for event-risk queries

### core/alerts.py  [NEW]
- ADD `AlertEvent` dataclass: `(alert_type: str, station: str | None, message: str,
  occurred_at: datetime, context: dict[str, Any] = field(default_factory=dict))`;
  alert_type domain: `"data_stale" | "order_failure" | "risk_rejected"`;
  `context` carries alert-specific fields: e.g. token_id, condition_id, order_id,
  strategy/risk name, exception message; UI shows only `message`; downstream
  subscribers (Telegram, dashboard) consume `context`
- ADD `AlertBus`: holds a list of registered async handler callables;
  `subscribe(handler: Callable[[AlertEvent], Awaitable[None]]) -> None`;
  `emit(event: AlertEvent) -> None`: must be called from within a running asyncio
  event loop; schedules each handler via `asyncio.create_task`; each task wraps the
  handler call in try/except -- any exception is logged at ERROR level and not
  propagated to the caller or to other handlers
- NOTE: `DataHealthMonitor`, `ModelHealthMonitor`, `SystemHealthMonitor` are
  deferred; future monitors subscribe to `AlertBus` without changing emitters

### core/strategy/base.py
- CHANGE `compute_signals` signature: add `open_positions: dict[str, int]` parameter
  (maps token_id -> count of current open orders); required for NO-token handling
  and per-token duplicate prevention at the strategy level
- ADD `name: str` abstract property (used for logging and UI display)

### core/strategy/edge.py
- FIX `compute_signals`: evaluate NO tokens as well as YES tokens;
  NO edge = (1 - mc_prob) - clob_price_no; generate BUY signal for NO token
  when NO edge > min_edge
- ADD `name` property returning `"edge"`

### core/strategy/multi.py  [NEW]
- ADD `MultiStrategy`: minimal viable implementation; constructor accepts non-empty
  `list[BaseStrategy]` (raises `ValueError` if list is empty); `compute_signals`
  calls each strategy in order and returns merged de-duplicated signal list
  (dedup by token_id, first signal wins); `name` property returns `"multi"`

### core/risk/base.py
- CHANGE `approve_signal` signature: add `open_positions: dict[str, int]` parameter
  (must match new BaseStrategy signature so call sites are consistent)
- CHANGE `compute_size` signature: add `open_positions: dict[str, int]` parameter

### core/risk/fixed_fraction.py
- ADD per-token duplicate position prevention: reject signal when
  `open_positions.get(signal.token_id, 0) >= settings.max_open_per_token`
  (`open_positions` is the `dict[str, int]` passed by the caller, not a direct
  call to `PositionTracker`; caller is responsible for providing current counts)

### core/risk/kelly.py  [NEW]
- ADD `KellyCriterion` stub: inherits `BaseRiskModel`; `approve_signal` and
  `compute_size` both raise `NotImplementedError` with
  "KellyCriterion not yet implemented" message

### core/risk/pipeline.py  [NEW]
- ADD `RiskPipeline`: inherits `BaseRiskModel`; accepts non-empty `list[BaseRiskModel]`
  in constructor (raises `ValueError` if empty); `approve_signal` applies AND logic --
  signal is approved only if every sub-model approves; `compute_size` calls each
  sub-model's `compute_size` and returns the minimum positive value across all results
  (guards against conflicting size requests from different risk layers);
  TradingLoop holds exactly one `RiskPipeline`, never a bare `BaseRiskModel`

### core/exchange/clob.py
- ADD authentication: wire full L2 ApiCreds (api_key, secret, passphrase) from
  settings into `ClobClient` constructor in `core/bot.py` (live mode only)
- ADD order failure alert: emit `AlertEvent(alert_type="order_failure", ...)` via
  `AlertBus` on CLOB submission exception, auth failure, or rejected-order response;
  log full error details at ERROR level regardless

### core/exchange/paper.py
- CHANGE from mid-price simulation to real CLOB order book driven by `OrderbookFetcher`
- ADD `OrderbookFetcher` wrapper in `core/exchange/paper.py`: wraps a read-only /
  L1-capable `ClobClient` (no ApiCreds); exposes only `get_book(token_id)`;
  paper mode never holds or requires L2 signing credentials
- ADD order failure alert: emit `AlertEvent(alert_type="order_failure", ...)` via
  `AlertBus` when `get_book()` fails or simulated fill cannot be computed

### core/bot.py
- FIX paper mode: mount real data sources to DataManager
- CHANGE authentication wire-up: in live mode, construct full L2 `ClobClient` with
  ApiCreds from settings and pass to `ClobExchange`; in paper mode, construct a
  separate L1-only `ClobClient` (no ApiCreds), wrap as `OrderbookFetcher`, pass
  only to `PaperExchange` -- L2 credentials are never loaded in paper mode  [ADR-0008]

### ui/app.py
- ADD data-staleness alert display: subscribe to `AlertBus` for `"data_stale"` events;
  show banner / status indicator in UI
- ADD Telegram stub subscriber: registers with `AlertBus`; handler is a no-op that
  logs at DEBUG level only ("Telegram not yet implemented"); wired at startup so the
  subscription point exists for future implementation without changing emitter code

---

## Layer 6: events/

### events/lifecycle.py
- REMOVE `_CUTOFF_HOUR_UTC` and `min_hours_before_cutoff` gating from Event state logic
- CHANGE `advance()`: `"pending_settle"` entered only when `event_end_date < now_utc`
  (midnight boundary); remove cutoff-hour gating entirely
- ADD `set_settled(event_id)`: called by `settlement_loop` after IEM confirmation;
  writes `"settled"` to DB; NOT called by `advance()`  [ADR-0007]

---

## Layer 7: backtest/

### backtest/engine.py
- ADD cold-start guard: skip first `cold_start_hours` rows instead of padding with zeros
- CHANGE `_process_init()`: update `self.pipeline.run(station, init_utc, buckets)` call
  site to match new `ModelPipeline.run()` signature after overhaul; do this as part of
  the Layer 3 pipeline.py changes, not as a separate pass

---

## Layer 8: settlement + snipers/

### loops/settlement_loop.py  [NEW]
- ADD `confirm_settlement(event, station)`: reads IEM ParquetStore for observed max temp
  on event date; compares to bucket boundaries; determines winning bucket  [ADR-0007]
- ADD after bucket confirmation: call `PositionTracker.mark_settled(order_id,
  settled_payout)` for every `filled` or `closed` order in that condition_id;
  winning bucket payout computed per Polymarket rules; losing bucket payout = 0
- ADD event status update: call `EventLifecycle.set_settled(event_id)` after all
  orders are marked settled
- ADD `SettlementRelayer` stub in `core/exchange/settlement.py`: abstract base with
  single method `resolve_on_chain(event_id) -> bool`; raises `NotImplementedError`
  referencing ADR-0007; Relayer path deferred
- WIRE `SettlementLoop`: call `confirm_settlement()` for each `"pending_settle"` Event;
  fall through to `SettlementRelayer` only if IEM data unavailable

### snipers/sniper_manager.py  [RENAME from snipers/manager.py]
- MOVE `snipers/manager.py` -> `snipers/sniper_manager.py` (removes import ambiguity
  with `data/manager.py`)
- CHANGE all import sites: `from snipers.manager import` -> `from snipers.sniper_manager import`

---

## Testing strategy (applies across all layers)

### Principles
- Implement and test one module at a time; do not defer all tests to end.
- Every interface signature change: update the calling-side tests in the same pass
  to avoid false-green suites.
- New stub / deferred components: only test that the interface exists and raises
  `NotImplementedError` with the expected message; do not write behaviour tests for
  unimplemented stubs. Components that are no-op (e.g. Telegram subscriber) or
  minimal viable implementations (e.g. `MultiStrategy`) get behaviour tests, not
  `NotImplementedError` tests.
- No large test investment for backtest replay, Telegram, or Monitor bodies
  this overhaul -- they are not on the critical path.
- Unit tests mock dependencies; integration tests (cross-layer) are written once
  the relevant layer boundary has stabilised.

### High-priority test targets (in layer order)

**Layer 2 -- data/**
- `Cleaner`: unit-convert K->C and F->C, wind decomposition, log1p smoothing,
  cold-start drop, feature output shape
- `DataManager.build_encoder_obs_window`: returns 72 merged rows, raises when
  fewer than `cold_start_hours` rows present for either f01 or IEM series
- `IemSource.fetch`: retrieves the :51 reading for the correct hour; falls back
  when :51 row absent

**Layer 3 -- model/**
- `ProbabilityStore`: correct key structure `(condition_id, day_offset)`,
  asyncio Lock prevents concurrent write corruption, read after write returns
  latest values, day_offset in {0, 1, 2}

**Layer 4 -- loops/**
- `ForecastLoop`: wall-clock UTC hour alignment fires within �5 s of the hour
- `TradingLoop`: 5-min tick cadence; staleness gate halts order submission and
  emits `AlertEvent(alert_type="data_stale")` when age exceeds threshold;
  holds exactly one `MultiStrategy` and one `RiskPipeline`

**Layer 5 -- core/**
- `MultiStrategy.compute_signals`: merges signals from N strategies, de-duplicates
  by token_id
- `RiskPipeline.approve_signal`: AND logic -- one rejection blocks all; does not
  call `compute_size` on a rejected signal
- `RiskPipeline.compute_size`: returns min positive value across sub-models
- `FixedFraction`: rejects signal when
  `open_positions_by_token().get(token_id, 0) >= max_open_per_token`
- `EdgeStrategy.compute_signals`: generates BUY for NO token when
  `(1 - mc_prob) - clob_price_no > min_edge`
- `PositionTracker`: full lifecycle transitions `pending->filled->closed->settled`,
  `pending->cancelled`; `open_positions_by_token()` counts only `pending|filled`
  rows; `total_exposure()` sums `size_usd` for `pending|filled` rows;
  `mark_cancelled()` removes the order from open counts immediately

**Layer 6 -- events/**
- `parse_slug` registry: every known pattern (or-above, or-higher, or-below,
  be-{N}f-or-below) returns correct `(lo, hi)` tuple; unmatched slug returns
  `None` and logs WARNING

**Layer 8 -- settlement/**
- `SettlementLoop.confirm_settlement`: correct bucket identified from IEM max temp;
  winning-bucket orders receive correct payout; losing-bucket orders receive 0;
  `EventLifecycle.set_settled()` called exactly once per event

## Deferred -- future extension points (do not implement in this overhaul)
- `DataHealthMonitor`: beyond staleness gate; drift / completeness monitoring
- `ModelHealthMonitor`: prob sanity checks, calibration drift, bucket coverage
- `SystemHealthMonitor`: loop heartbeat, process health, resource monitoring
- Full Telegram routing, notification dedup, recovery alerts, dashboard
