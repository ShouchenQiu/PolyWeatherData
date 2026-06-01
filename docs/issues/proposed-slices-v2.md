# Proposed Implementation Slices V2

Source of truth: `docs/grill-tasks-v2.md`
ADRs: `docs/adr/0009-polymarket-client-sdk-migration.md`,
`docs/adr/0010-daily-high-fxx-window-local-calendar-day.md`,
`docs/adr/0011-l2-vwap-pricing-layer.md`, plus earlier ADRs where referenced.

These slices are for the second overhaul only. The first-overhaul slice list remains
`docs/issues/proposed-slices.md`.

Total: 5 foundation + 12 vertical = **17 slices**

---

## Dependency Graph

Recommended first batch:
- V2F01 Schema Expansion
- V2F02 Polymarket SDK Client Split
- V2F03 L2 Quote Primitive
- V2F04 ProbabilitySnapshot + Feature Contract
- V2F05 Position Accounting + PnL/Risk Primitives

Then:
- V2V01 Discovery + Lifecycle Hardening
- V2V02 HRRR Source Hardening + Extrapolation Flags
- V2V03 Obs + Parquet Hardening
- V2V04 Encoder Readiness Gate
- V2V05 ModelPipeline Validation + Daily-High Target

Then:
- V2V06 MC Covariance + Production Model Wiring
- V2V07 ForecastLoop V2 Write Path
- V2V08 Directional Trading Arbitration
- V2V09 Exchange Fill Semantics
- V2V10 Backtest V2 Semantics
- V2V11 SniperLoop Alignment
- V2V12 Redeem Loop

---

## Foundation Slices

Foundation slices deliver shared primitives that downstream vertical slices depend on.

---

### V2F01 -- Schema Expansion

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Extend the database schema for V2 trading, discovery, observability, and redeem
  semantics. Add the new order accounting columns, the `signals` decision-log table,
  the condition-level `redeem_tasks` table, and `markets.discovery_status`.
- **Acceptance criteria**:
  - [ ] Fresh DB contains `orders.is_paper`, `orders.executable_price`, `orders.filled_shares`, `orders.signed_shares`, and `orders.cash_flow_usd`
  - [ ] Fresh DB contains `signals` with directional, arbitration, risk, and order-linkage columns
  - [ ] Fresh DB contains `redeem_tasks` keyed by `condition_id` with status, attempts, error, and timestamp fields
  - [ ] `markets.discovery_status` defaults to `parsed` and accepts `parsed`, `unrecognized_slug`, and `invalid_market`
  - [ ] Schema tests document status domains for `redeem_tasks` and any CHECK constraints used
- **Tests**:
  - Create an in-memory DB and assert all new columns/tables exist with defaults
  - Insert representative rows for paper/live orders, signal decisions, redeem tasks, and unrecognized markets
- **Source refs**:
  - `docs/grill-tasks-v2.md:5-39` (`db/schema.py`)
- **Out of scope**:
  Position/PnL calculations (V2F05); TradingLoop writes to `signals` (V2V08);
  redeem processing (V2V12); discovery behavior (V2V01)

---

### V2F02 -- Polymarket SDK Client Split

- **Type**: AFK
- **Blocked by**: V2F01
- **What to build**:
  Replace old Polymarket credential fields and `py_clob_client` usage with the new
  SDK split: `SecureClient` for live writes, `PublicClient` for paper/read-only
  market data, and a normalized `OrderbookFetcher` in `data/clob/orderbook.py`.
- **Acceptance criteria**:
  - [ ] Old five credential settings are removed and no code reads them
  - [ ] New private-key and wallet-address settings exist and are used only for live `SecureClient`
  - [ ] Paper mode constructs only `PublicClient`; private key is not loaded in paper mode
  - [ ] `OrderbookFetcher.get_book()` returns sorted bids/asks plus `best_bid`, `best_ask`, and `mid_price`
  - [ ] `ClobExchange` uses `SecureClient.place_limit_order()` and `SecureClient.cancel_orders()`
  - [ ] No import of `py_clob_client` remains
- **Tests**:
  - Settings tests for removed and added credential fields
  - Fake `PublicClient` order book tests for bid/ask sorting and best prices
  - Fake `SecureClient` tests for live place/cancel adapter calls
  - Bot wiring tests proving paper mode does not touch private-key settings
- **Source refs**:
  - `docs/grill-tasks-v2.md:43-51` (`config/settings.py`)
  - `docs/grill-tasks-v2.md:213-221` (`data/clob/orderbook.py`)
  - `docs/grill-tasks-v2.md:434-443` (`core/exchange/clob.py`)
  - `docs/grill-tasks-v2.md:522-527` (`core/bot.py`)
  - ADR-0009
- **Out of scope**:
  VWAP pricing math (V2F03); paper fill simulation (V2V09); trading arbitration
  (V2V08)

---

### V2F03 -- L2 Quote Primitive

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Add the shared L2 VWAP pricing primitive. `compute_quote()` walks normalized book
  depth for BUY/SELL, returns a `Quote`, and enforces full-fill-or-skip semantics for
  callers.
- **Acceptance criteria**:
  - [ ] `core/pricing/quote.py` exists as a leaf module
  - [ ] BUY consumes asks ascending; SELL consumes bids descending
  - [ ] `Quote` includes VWAP, max fillable USD, filled shares, slippage bps, best bid/ask, mid, and `complete_fill`
  - [ ] Depth-insufficient quotes set `complete_fill=False`
  - [ ] `settings.quote_size_usd` exists with the V2 default
- **Tests**:
  - Unit tests for full BUY, full SELL, insufficient depth, and slippage calculation
  - Unit test that the module imports without importing data/model/loops
- **Source refs**:
  - `docs/grill-tasks-v2.md:397-408` (`core/pricing/quote.py`, `quote_size_usd`)
  - ADR-0011
- **Out of scope**:
  Strategy edge computation (V2V08); exchange fills (V2V09); backtest use (V2V10)

---

### V2F04 -- ProbabilitySnapshot + Feature Contract

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Centralize model feature names and update the in-process probability store shape.
  The probability store moves key-level quantiles and timestamp into
  `ProbabilitySnapshot`; model/train/TFT code reads feature names from one contract.
- **Acceptance criteria**:
  - [ ] `ProbabilitySnapshot` stores bucket probabilities separately from `quantiles_by_fxx` and `last_tick_utc`
  - [ ] Existing ProbabilityStore read/write tests are updated for the new shape
  - [ ] `model/features.py` defines canonical cleaned HRRR/obs feature lists and generated columns
  - [ ] `model/tft.py` and `scripts/train_tft.py` import feature lists instead of duplicating them
  - [ ] `model/features.py` does not import from other project packages
- **Tests**:
  - ProbabilityStore read/write/concurrency tests with `ProbabilitySnapshot`
  - Feature-contract tests asserting expected canonical columns and no package imports
  - TFT/train import smoke tests for the shared lists
- **Source refs**:
  - `docs/grill-tasks-v2.md:227-247` (`model/features.py`, `model/tft.py`)
  - `docs/grill-tasks-v2.md:306-308` (`scripts/train_tft.py`)
  - `docs/grill-tasks-v2.md:390-395` (`core/probability_store.py`)
- **Out of scope**:
  Pipeline validation behavior (V2V05); ForecastLoop write path (V2V07)

---

### V2F05 -- Position Accounting + PnL/Risk Primitives

- **Type**: AFK
- **Blocked by**: V2F01, V2F03
- **What to build**:
  Make order accounting share-aware and cash-flow-aware. PositionTracker writes
  filled shares, signed shares, and cash flow; PnL uses signed shares; risk permits
  scale-in under USD exposure limits and computes SELL exits from share inventory.
- **Acceptance criteria**:
  - [ ] `PositionTracker.record_order()` accepts `is_paper` and `executable_price`
  - [ ] `mark_filled()` writes `filled_shares`, `signed_shares`, and `cash_flow_usd`
  - [ ] `position_shares()` returns net signed token shares
  - [ ] PnLCalculator computes realized/unrealized PnL from cash flows and signed shares
  - [ ] `max_open_per_token` no longer blocks BUY scale-in
  - [ ] SELL sizing uses `position_shares * executable_price * max_exit_fraction`
- **Tests**:
  - BUY and SELL fills produce expected signed shares and cash flows
  - Realized and unrealized PnL tests for paper/live filters and net positions
  - Risk tests for allowed scale-in, max-total-USD rejection, and SELL exit sizing
- **Source refs**:
  - `docs/grill-tasks-v2.md:453-508` (`core/positions.py`, `core/pnl.py`, risk/settings)
  - `docs/grill-tasks-v2.md:13-22` (order accounting columns from V2F01)
- **Out of scope**:
  TradingLoop arbitration (V2V08); exchange fill simulation (V2V09); settlement/redeem
  (V2V12)

---

## Vertical Slices

Each vertical slice should be independently verifiable once its blockers are complete.

---

### V2V01 -- Discovery + Lifecycle Hardening

- **Type**: AFK
- **Blocked by**: V2F01, V2F02
- **What to build**:
  Make market discovery robust under the new SDK and slug strategy. Discovery uses the
  SDK event paginator, station aliases, token/outcome length checks, unrecognized-slug
  persistence with alerting, downstream hard filters, summary logging, and lifecycle
  advancement over locally stored non-terminal events.
- **Acceptance criteria**:
  - [ ] Discovery calls the new SDK with live/weather filters
  - [ ] `_infer_station()` uses configured aliases and raises on no match
  - [ ] Token/outcome length mismatch logs and skips without truncation
  - [ ] Unrecognized slugs are upserted with `discovery_status="unrecognized_slug"` and emit `discovery_warn`
  - [ ] Tradeable/active event queries exclude unparsed or null-bucket markets
  - [ ] Local non-terminal events advance to `pending_settle` even if no longer returned by live API
  - [ ] Lifecycle transition logs are emitted and no `cutoff` state remains
- **Tests**:
  - Discovery unit tests with fake SDK events for parsed, unrecognized, and mismatched markets
  - DB query tests for discovery-status hard gate
  - Lifecycle tests for locally stored just-expired events
  - Settings/station tests for `discovery_loop_interval_s` and `slug_aliases`
- **Source refs**:
  - `docs/grill-tasks-v2.md:36-39` (`markets.discovery_status`)
  - `docs/grill-tasks-v2.md:335-384` (`events/lifecycle.py`, discovery settings/stations/events)
  - ADR-0009
- **Out of scope**:
  Forecast invalid-bucket guard (V2V07); settlement/redeem (V2V12)

---

### V2V02 -- HRRR Source Hardening + Extrapolation Flags

- **Type**: AFK
- **Blocked by**: V2F02
- **What to build**:
  Harden all HRRR forecast sources and produce a consistent `is_extrapolated` flag.
  HrrrFill handles non-MAJOR tail gaps deterministically; GRIB sources parse `.idx`
  precisely and validate byte ranges; S3/NOMADS/Zarr read configured base URLs; Zarr
  fails fast on critical variable/grid errors and delegates non-MAJOR extension to
  HrrrFill.
- **Acceptance criteria**:
  - [ ] All HRRR source outputs include `is_extrapolated`
  - [ ] HrrrFill handles duplicate/sub-second valid_times and output column union
  - [ ] HrrrFill tail extrapolation is capped and marks rows true
  - [ ] GRIB `.idx` matching rejects ambiguous or duplicate matches
  - [ ] GRIB range fetch validates 206/Content-Range and supports open-ended final ranges
  - [ ] Zarr raises for critical `TMP_2m`, bad shape probes, missing grid config, and row/col bounds
  - [ ] S3/NOMADS/Zarr base URLs come from settings or test overrides
- **Tests**:
  - HrrrFill splice, duplicate valid_time, floor-to-second, union columns, and capped tail tests
  - GRIB index/range parser tests with ambiguous rows, 200/206 responses, final message
  - Zarr failure-path tests for critical variable, bounds, and consistency assertion
  - URL tests for S3/NOMADS/Zarr configured bases
- **Source refs**:
  - `docs/grill-tasks-v2.md:57-128` (HRRR forecast source items)
- **Out of scope**:
  Parquet write behavior (V2V03); model exclusion of extrapolated fxx rows (V2V05)

---

### V2V03 -- Obs + Parquet Hardening

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Harden ParquetStore, IEM cleaning, and NOAA tick fallback. Reads become side-effect
  free; dedup keys normalize time; obs completeness checks require more than
  temperature; IEM/NOAA wind, sky, and tick semantics are made consistent.
- **Acceptance criteria**:
  - [ ] Parquet dedup normalizes timestamp keys before comparison
  - [ ] Parquet reads do not create station directories
  - [ ] Parquet bad schema raises a clear `ValueError`
  - [ ] ParquetStore has an explicit concurrency TODO
  - [ ] IEM :51 observations are rewritten to the represented hourly `valid_time`
  - [ ] NOAA only acts as now-utc tick fallback and rejects broad windows
  - [ ] IEM and NOAA treat VRB/missing wind direction as NaN before U/V calculation
  - [ ] `skyc1`, U/V fill order, and obs completeness rules match V2 requirements
- **Tests**:
  - Parquet unit tests for dedup, read side effects, and schema errors
  - IEM cleaner tests for :51 valid_time, sort before fills, sky fill, wind handling
  - NOAA parser/fetch tests for tick window, range guard, and VRB handling
  - DataManager completeness tests rejecting temperature-only rows
- **Source refs**:
  - `docs/grill-tasks-v2.md:130-167` (`parquet_store.py`, `iem.py`, `noaa_file.py`)
  - `docs/grill-tasks-v2.md:169-171` (`cleaner.py` sort order)
  - `docs/grill-tasks-v2.md:207-211` (`DataManager._obs_is_complete()`)
- **Out of scope**:
  Encoder readiness and fixed 72h window behavior (V2V04); production bot wiring for
  NOAA fallback (V2V06)

---

### V2V04 -- Encoder Readiness Gate

- **Type**: AFK
- **Blocked by**: V2V03, V2F04
- **What to build**:
  Separate cold-start burn-in from encoder-window construction. DataManager owns a
  shared readiness gate and defensive encoder-window builder; ForecastLoop and
  BacktestEngine skip not-ready inits instead of crashing or inventing rows.
- **Acceptance criteria**:
  - [ ] Cleaner does not drop cold-start rows on each live single-tick append
  - [ ] `build_encoder_obs_window()` reads exactly `[init_time - 72h, init_time - 1h]`
  - [ ] Missing/incomplete encoder rows raise precise errors
  - [ ] Store reads are copied before mutation
  - [ ] `inference_readiness()` returns ready, reason, and first eligible init without double-counting cold start
  - [ ] ForecastLoop logs structured skip messages and continues when not ready
  - [ ] BacktestEngine skips not-ready historical inits and records reasons
- **Tests**:
  - DataManager tests for first clean row + encoder hours, incomplete HRRR/obs windows, missing columns, and `.copy()`
  - ForecastLoop readiness-skip test
  - Backtest readiness-skip test
- **Source refs**:
  - `docs/grill-tasks-v2.md:172-206` (`cleaner.py`, `data/manager.py`)
  - `docs/grill-tasks-v2.md:538-543` (`backtest/engine.py`)
  - `docs/grill-tasks-v2.md:583-589` (`loops/forecast_loop.py`)
  - ADR-0001
- **Out of scope**:
  TFT/pipeline validation after readiness passes (V2V05); ForecastLoop bucket write
  behavior (V2V07)

---

### V2V05 -- ModelPipeline Validation + Daily-High Target

- **Type**: AFK
- **Blocked by**: V2F04, V2V02, V2V04
- **What to build**:
  Make TFT and ModelPipeline fail fast on malformed inputs, use canonical feature
  contracts, reject invalid quantiles, exclude extrapolated target rows, and compute
  daily-high probabilities for the event's local calendar day.
- **Acceptance criteria**:
  - [ ] TFT sorts by `time_idx`, checks encoder/decoder row counts, decoder fxx 1..48, model output shape, and non-null `TMP_2m`
  - [ ] Pipeline requires `valid_time`, required source columns, decoder completeness, sorted fxx, finite ordered quantiles, and positive sigma
  - [ ] Pipeline rejects `tft=None`
  - [ ] Pipeline excludes `is_extrapolated=True` target rows and raises if no target rows remain
  - [ ] `pipeline.run()` accepts `station_config` and `target_date`
  - [ ] Daily-high target fxx rows are selected by station local-calendar-day boundaries
- **Tests**:
  - TFT validation tests for shape, fxx completeness, row counts, sort, and NaN TMP_2m
  - Pipeline tests for missing valid_time/source columns, decoder fail-fast, quantile ordering, sigma guard, no tft, and extrapolated target exclusion
  - ADR-0010 tests for local-calendar-day fxx selection and `target_date` semantics
- **Source refs**:
  - `docs/grill-tasks-v2.md:245-305` (`model/tft.py`, `model/pipeline.py`)
  - ADR-0010
- **Out of scope**:
  ForecastLoop event iteration/write behavior (V2V07); MC covariance loading (V2V06)

---

### V2V06 -- MC Covariance + Production Model Wiring

- **Type**: AFK
- **Blocked by**: V2V03, V2V05
- **What to build**:
  Wire the production model stack correctly. Registry can load covariance beside the
  model checkpoint; McSampler accepts covariance and validates inputs; Bot constructs
  McSampler, NOAA fallback, TFT, covariance, and rejects missing TFT checkpoints.
- **Acceptance criteria**:
  - [ ] `ModelRegistry.load_covariance()` returns a matrix or logs a diagonal fallback warning
  - [ ] `McSampler.set_cov()` exists and production code uses it when covariance exists
  - [ ] `fit_cov()`, `_mc_sample()`, and `bucket_probs()` validate inputs and bucket overlaps
  - [ ] Bot constructs `McSampler(settings.mc_n_samp)`, not `McSampler(settings)`
  - [ ] Bot wires `NoaaFileSource` into `IemSource`
  - [ ] Bot raises a clear error when the TFT checkpoint is absent
- **Tests**:
  - Registry tests for present/missing covariance
  - McSampler validation tests for empty residuals, bad mu/sigma, empty samples, and overlapping buckets
  - Bot construction tests with fake model/covariance paths and NOAA fallback
- **Source refs**:
  - `docs/grill-tasks-v2.md:310-330` (`model/store/registry.py`, `model/mc.py`)
  - `docs/grill-tasks-v2.md:510-521` (`core/bot.py`)
- **Out of scope**:
  ForecastLoop probability writes (V2V07); exchange/client wiring (V2F02/V2V09)

---

### V2V07 -- ForecastLoop V2 Write Path

- **Type**: AFK
- **Blocked by**: V2V01, V2V04, V2V05, V2V06
- **What to build**:
  Update ForecastLoop to call ModelPipeline with event station/target-date semantics,
  filter invalid buckets, avoid empty ProbabilityStore writes, and survive top-level
  tick exceptions.
- **Acceptance criteria**:
  - [ ] ForecastLoop derives `station_config` from event station id
  - [ ] ForecastLoop passes event `event_end_date` as `target_date`
  - [ ] `day_offset` remains only the ProbabilityStore key
  - [ ] Buckets with null bounds are filtered with warning logs
  - [ ] Empty bucket maps skip ProbabilityStore writes
  - [ ] `run()` catches top-level tick exceptions, logs traceback, and continues
  - [ ] Writes use the V2 `ProbabilitySnapshot` shape
- **Tests**:
  - ForecastLoop test for station-specific pipeline call and target-date propagation
  - Invalid/null bucket filtering tests
  - Empty-probs no-write test
  - Top-level exception guard test
- **Source refs**:
  - `docs/grill-tasks-v2.md:590-602` (`loops/forecast_loop.py`)
  - `docs/grill-tasks-v2.md:390-395` (`core/probability_store.py`)
  - ADR-0010
- **Out of scope**:
  Data readiness logic itself (V2V04); ModelPipeline internals (V2V05)

---

### V2V08 -- Directional Trading Arbitration

- **Type**: AFK
- **Blocked by**: V2F01, V2F03, V2F05, V2V07
- **What to build**:
  Change strategy output from executable orders to directional mispricing signals and
  let TradingLoop arbitrate SELL YES vs BUY NO using inventory, L2 depth checks,
  risk, re-quote, per-tick local state updates, and signal DB observability.
- **Acceptance criteria**:
  - [ ] `DirectionalSignal` and `OrderIntent` dataclasses exist with V2 fields
  - [ ] EdgeStrategy uses `compute_quote()` at `quote_size_usd` and emits directional signals only
  - [ ] Bearish signals become SELL YES when YES inventory exists, otherwise BUY NO
  - [ ] Both arbitration paths require complete L2 quote checks
  - [ ] TradingLoop re-quotes after final size and skips if edge/depth no longer clears
  - [ ] Local open positions and exposure update immediately after successful order record
  - [ ] `signals` rows are written/updated from signal through order linkage
  - [ ] Tick summary logs are emitted; non-actionable observability does not emit AlertBus events
- **Tests**:
  - EdgeStrategy tests for quote-based bullish/bearish directional signals
  - TradingLoop tests for SELL YES vs BUY NO arbitration
  - Re-quote rejection tests for stale edge and insufficient depth
  - Per-tick state update test with two signals in one tick
  - Signal DB lifecycle tests
- **Source refs**:
  - `docs/grill-tasks-v2.md:410-432` (`core/types.py`, `core/strategy/edge.py`)
  - `docs/grill-tasks-v2.md:553-581` (`loops/trading_loop.py`)
  - ADR-0011
- **Out of scope**:
  Exchange adapter internals (V2V09); SniperLoop behavior (V2V11)

---

### V2V09 -- Exchange Fill Semantics

- **Type**: AFK
- **Blocked by**: V2F02, V2F03, V2F05, V2V08
- **What to build**:
  Make live and paper exchanges obey the same price vocabulary. Live exchange only
  submits already-validated limit orders; paper exchange simulates fills by walking
  L2 depth and returns order ids while persistence remains in TradingLoop/PositionTracker.
- **Acceptance criteria**:
  - [ ] `ClobExchange` submits only the provided validated limit price
  - [ ] `ClobExchange` does not fetch books, re-quote, or evaluate edge
  - [ ] Paper fills BUY at asks VWAP and SELL at bids VWAP
  - [x] PaperExchange trusts the caller-validated price; TradingLoop owns depth
        validation via two `compute_quote()` passes before exchange submission
  - [ ] Paper exchange no longer owns in-memory orders/positions
  - [ ] TradingLoop records returned live/paper order ids with `is_paper` and `executable_price`
- **Tests**:
  - Fake exchange tests proving live adapter passes through order parameters
  - PaperExchange tests for BUY/SELL VWAP and insufficient depth rejection
  - Integration-style TradingLoop test for paper order persistence fields
- **Source refs**:
  - `docs/grill-tasks-v2.md:434-451` (`core/exchange/clob.py`, `core/exchange/paper.py`)
  - `docs/grill-tasks-v2.md:522-527` (`core/bot.py`)
  - ADR-0009, ADR-0011
- **Out of scope**:
  Trading signal arbitration (V2V08); PnL calculations (V2F05)

---

### V2V10 -- Backtest V2 Semantics

- **Type**: AFK
- **Blocked by**: V2F03, V2V04
- **What to build**:
  Align backtest pricing with V2 live semantics while preserving the readiness gate
  introduced in V2V04. Backtest uses the same quote primitive when historical L2
  data exists; legacy mid-price rows are explicitly marked as legacy.
- **Acceptance criteria**:
  - [ ] Backtest readiness skip introduced in V2V04 remains covered by regression tests
  - [ ] Historical L2 rows use `compute_quote()`
  - [ ] Historical mid-only rows are stored as `legacy_mid_price`
  - [ ] `legacy_mid_price` is not passed through `compute_quote()` or labelled `clob_price`
- **Tests**:
  - Backtest readiness-skip regression test
  - Backtest L2 quote path test
  - Backtest legacy mid fallback test
- **Source refs**:
  - `docs/grill-tasks-v2.md:538-547` (`backtest/engine.py`)
  - ADR-0011
- **Out of scope**:
  ForecastLoop readiness (V2V04); full historical L2 data ingestion

---

### V2V11 -- SniperLoop Alignment

- **Type**: AFK
- **Blocked by**: V2F02, V2F03, V2F05, V2V01
- **What to build**:
  Bring SniperLoop onto the same risk, orderbook, persistence, and event-query
  contracts as the main TradingLoop while preserving its exemption from forecast
  staleness.
- **Acceptance criteria**:
  - [x] SniperLoop calls risk approve/size with `positions` (PositionTracker for
        SELL sizing via `position_shares()`; matches V2F05 contract)
  - [x] SniperLoop prices via `compute_quote()` instead of `best_ask`/mid fallback;
        no-orderbook / incomplete-fill / quote-error all skip (no 0.95 fallback)
  - [x] SniperLoop records orders with `is_paper` and `executable_price`
  - [x] SniperLoop always uses `get_tradeable_events`
  - [x] Missing `get_tradeable_events` raises loudly
  - [x] SniperLoop filters events by `station_id` (avoids cross-station trading)
- **Tests**:
  - SniperLoop tests for updated risk method signatures
  - SniperLoop quote-based price test
  - SniperLoop order-record argument test
  - SniperLoop event-query failure test
- **Source refs**:
  - `docs/grill-tasks-v2.md:630-643` (`loops/sniper_loop.py`)
- **Out of scope**:
  TradingLoop arbitration (V2V08); discovery implementation (V2V01)

---

### V2V12 -- Redeem Loop

- **Type**: AFK
- **Blocked by**: V2F01, V2F02
- **What to build**:
  Implement condition-level redeem task creation and retry processing. Settlement
  enqueues one task for live winning conditions; RedeemLoop retries failed tasks with
  `SecureClient.ctf_redeem_positions_call()`; Bot wires the loop only in live mode.
- **Acceptance criteria**:
  - [ ] SettlementLoop upserts a pending live redeem task only when winning payout exists
  - [ ] Paper mode explicitly skips or marks redeem tasks as `skipped_paper`
  - [ ] RedeemLoop polls pending/failed tasks, calls the SDK once per condition, and records success/failure fields
  - [ ] Failures increment attempts, store error text, and retry next cycle
  - [ ] `redeem_loop_interval_s` exists and controls sleep cadence
  - [ ] Bot starts RedeemLoop only when a live `SecureClient` exists
- **Tests**:
  - Settlement enqueue tests for winning, losing, live, and paper conditions
  - RedeemLoop success/failure/retry tests with fake SecureClient
  - Bot wiring tests for live vs paper mode
- **Source refs**:
  - `docs/grill-tasks-v2.md:30-35` (`redeem_tasks` table)
  - `docs/grill-tasks-v2.md:528-532` (`core/bot.py` redeem wiring)
  - `docs/grill-tasks-v2.md:604-628` (`settlement_loop.py`, `redeem_loop.py`, settings)
- **Out of scope**:
  On-chain settlement relayer changes beyond the SDK redeem call

---

## Coverage Matrix

| Source item | Covered by slice | Status |
|---|---|---|
| `docs/grill-tasks-v2.md:5-39` db/schema.py | V2F01, V2F05, V2V08, V2V12 | Covered |
| `docs/grill-tasks-v2.md:43-51` Polymarket credential settings | V2F02 | Covered |
| `docs/grill-tasks-v2.md:57-128` HRRR forecast sources | V2V02 | Covered |
| `docs/grill-tasks-v2.md:130-143` ParquetStore hardening | V2V03 | Covered |
| `docs/grill-tasks-v2.md:145-167` IEM/NOAA obs hardening | V2V03 | Covered |
| `docs/grill-tasks-v2.md:169-178` Cleaner cold-start/sort semantics | V2V03, V2V04 | Covered |
| `docs/grill-tasks-v2.md:180-211` DataManager encoder/readiness/completeness | V2V03, V2V04 | Covered |
| `docs/grill-tasks-v2.md:213-221` OrderbookFetcher | V2F02 | Covered |
| `docs/grill-tasks-v2.md:227-247` model/features and TFT feature imports | V2F04 | Covered |
| `docs/grill-tasks-v2.md:248-257` TFT input/output validation | V2V05 | Covered |
| `docs/grill-tasks-v2.md:259-305` ModelPipeline validation and daily target | V2V05 | Covered |
| `docs/grill-tasks-v2.md:306-308` train_tft feature imports | V2F04 | Covered |
| `docs/grill-tasks-v2.md:310-330` covariance registry and McSampler hardening | V2V06 | Covered |
| `docs/grill-tasks-v2.md:335-384` lifecycle and discovery hardening | V2V01 | Covered |
| `docs/grill-tasks-v2.md:390-395` ProbabilitySnapshot | V2F04, V2V07 | Covered |
| `docs/grill-tasks-v2.md:397-408` L2 quote primitive and quote size | V2F03 | Covered |
| `docs/grill-tasks-v2.md:410-432` DirectionalSignal, OrderIntent, EdgeStrategy | V2V08 | Covered |
| `docs/grill-tasks-v2.md:434-451` live/paper exchange semantics | V2F02, V2V09 | Covered |
| `docs/grill-tasks-v2.md:453-508` positions, PnL, risk, exit sizing | V2F05 | Covered |
| `docs/grill-tasks-v2.md:510-532` Bot wiring | V2F02, V2V06, V2V12 | Covered |
| `docs/grill-tasks-v2.md:538-547` Backtest readiness regression and L2 pricing | V2V10 | Covered |
| `docs/grill-tasks-v2.md:553-581` TradingLoop arbitration/observability | V2V08 | Covered |
| `docs/grill-tasks-v2.md:583-602` ForecastLoop V2 path | V2V04, V2V07 | Covered |
| `docs/grill-tasks-v2.md:604-628` Settlement/RedeemLoop | V2V12 | Covered |
| `docs/grill-tasks-v2.md:630-643` SniperLoop alignment | V2V11 | Covered |

Uncovered items: none.

---

## Issue Coverage Audit

| Issue | Covered by slice |
|---|---|
| #1 | V2F02 |
| #2 | V2F04 |
| #3 | V2V04, V2V10 |
| #4 | V2V05, V2V07 |
| #5 | V2F03, V2V09, V2V10 |
| #6 | V2V08 |
| #7 | V2F05, V2V08 |
| #8 | V2F01, V2V09 |
| #9 | V2F05, V2V08 |
| #10 | V2F05 |
| #11 | V2V11 |
| #12 | V2F01, V2V08 |
| #13 | V2V01 |
| #14 | V2V01 |
| #15 | V2F01, V2V01 |
| #16 | V2V01 |
| #17 | V2V01 |
| #18 | V2F01, V2V01, V2V07 |
| #19 | V2V01 |
| #20 | V2V01 |
| #21 | V2F04, V2V05 |
| #22 | V2V05, V2V07 |
| #23 | V2V05 |
| #24 | V2V05, V2V07 |
| #25 | V2V05 |
| #26 | V2F04, V2V05 |
| #27 | V2V04 |
| #28 | V2V02, V2V05 |
| #29 | V2V02 |
| #30 | V2V02 |
| #31 | V2V02 |
| #32 | V2V02 |
| #33 | V2V02 |
| #34 | V2V02 |
| #35 | V2V02 |
| #36 | V2V02 |
| #37 | V2V02 |
| #38 | V2V02 |
| #39 | V2V02 |
| #40 | V2V02 |
| #41 | V2V02 |
| #42 | V2V02 |
| #43 | V2V02 |
| #44 | V2V02 |
| #45 | V2V03 |
| #46 | V2V03 |
| #47 | V2V03 |
| #48 | V2V03 |
| #49 | V2V03 |
| #50 | V2V03 |
| #51 | V2V03 |
| #52 | V2V03 |
| #53 | V2V03 |
| #54 | V2V03 |
| #55 | V2V03 |
| #56 | V2V03 |
| #57 | V2V03 |
| #58 | V2V04 |
| #59 | V2V04 |
| #60 | V2V05 |
| #61 | V2V05 |
| #62 | V2V05 |
| #63 | V2V05 |
| #64 | V2V05 |
| #65 | V2V06 |
| #66 | V2V06 |
| #67 | V2V06 |
| #68 | V2V05, V2V06 |
| #69 | V2V05 |
| #70 | V2V05 |
| #71 | V2V05 |
| #72 | V2V07 |
| #73 | V2V12 |

---

## Review Questions

- Does the granularity feel right, or should any large slices be split further?
- Are the dependencies correct, especially around V2V04/V2V05/V2V07?
- Should V2F02 and V2V09 stay separate, or merge the SDK/client and exchange-fill work?
- Are any items incorrectly treated as AFK rather than HITL?
