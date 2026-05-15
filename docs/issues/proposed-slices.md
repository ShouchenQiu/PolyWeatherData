# Proposed Implementation Slices

Source of truth: `docs/grill-tasks.md`
ADRs: `docs/adr/0001` through `docs/adr/0008`

Total: 5 foundation + 1 HITL + 15 vertical = **21 slices**

---

## Foundation Slices

Foundation slices deliver shared primitives that multiple vertical slices depend on.
Each is independently verifiable before any dependent slice starts.

---

### F01 -- Settings Cleanup + DB Orders Table

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Remove 5 dead fields from `config/settings.py`; change `mc_n_samp` default to 10000;
  add all 13 new settings fields (TradingLoop cadence, staleness, HRRR priority, risk,
  cold-start, feature flags, train/val/test windows). Add `orders` table to `db/schema.py`
  with full lifecycle columns and NULL semantics for optional timestamps.
- **Acceptance criteria**:
  - [ ] All 5 REMOVE fields are gone from `settings.py` and no import site references them
  - [ ] All 13 new fields have correct types and defaults
  - [ ] `db/schema.py` creates `orders` table with correct columns and constraints on fresh DB
  - [ ] `orders.status` domain enforced as CHECK constraint or documented comment
  - [ ] No existing test references removed settings fields
- **Tests**:
  - `config/settings.py`: assert defaults match spec; assert removed fields raise AttributeError
  - `db/schema.py`: create in-memory SQLite, verify `orders` table columns and NULL defaults
- **Source refs**:
  - `grill-tasks.md:29` (`config/settings.py` -- all REMOVE/CHANGE/ADD items, L30-50)
  - `grill-tasks.md:60` (`db/schema.py` -- orders table, L61-68)
- **Out of scope**:
  PositionTracker lifecycle methods (F03); cutoff removal from loop code (V09, V12);
  any module that reads the new settings fields (each downstream slice picks them up)

---

### F02 -- AlertBus

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Create `core/alerts.py` with `AlertEvent` dataclass and `AlertBus`.
  `AlertEvent` fields: `alert_type`, `station: str | None`, `message`, `occurred_at`,
  `context: dict[str, Any]`. `AlertBus.subscribe` registers async handlers;
  `AlertBus.emit` schedules each handler as an `asyncio.create_task`, wrapping each
  in try/except that logs exceptions at ERROR and does not propagate them.
- **Acceptance criteria**:
  - [ ] `AlertEvent` is a dataclass with all specified fields; `context` defaults to an empty dict via `field(default_factory=dict)`
  - [ ] `AlertBus.subscribe` registers a handler and `emit` invokes it via `create_task`
  - [ ] Exception raised inside a handler does not propagate to the caller or other handlers
  - [ ] Exception is logged at ERROR level with the event's `alert_type` in the message
  - [ ] `emit` raises `RuntimeError` if called outside a running event loop (asyncio contract)
- **Tests**:
  - Handler is called with correct `AlertEvent` when `emit` fires
  - Second handler still fires when first handler raises an exception
  Emitting `"data_stale"`, `"order_failure"`, `"risk_rejected"`, `"book_fetch_failure"` types all accepted
- **Source refs**:
  - `grill-tasks.md:237` (`core/alerts.py [NEW]` -- AlertEvent + AlertBus, L238-251)
- **Out of scope**:
  All subscribers (UI in V13, TradingLoop emitters in F05/V11, CLOB emitters in V07);
  Monitor bodies (Deferred)

---

### F03 -- PositionTracker Order Lifecycle

- **Type**: AFK
- **Blocked by**: F01 (orders table must exist in schema)
- **What to build**:
  Extend `core/positions.py` with 8 new methods covering the full order lifecycle:
  `record_order`, `mark_filled`, `mark_closed`, `mark_cancelled`, `mark_settled`,
  `open_positions_by_token`, `total_exposure`, `exposure_by_condition`.
  Timestamps default to current UTC when not provided.
  `cancelled` rows are excluded from open counts and exposure totals.
- **Acceptance criteria**:
  - [ ] `record_order` writes a `pending` row; all nullable columns are NULL
  - [ ] `mark_filled` transitions status to `filled` and sets `fill_price` + `filled_at`
  - [ ] `mark_closed` transitions status to `closed` and sets `closed_at`
  - [ ] `mark_cancelled` transitions status to `cancelled`; order immediately absent from `open_positions_by_token()`
  - [ ] `mark_settled` transitions status to `settled` and writes `settled_payout`
  - [ ] `open_positions_by_token()` counts only `pending | filled` rows per `token_id`
  - [ ] `total_exposure()` sums `size_usd` for `pending | filled` rows only
  - [ ] `exposure_by_condition(condition_id)` sums `size_usd` for matching `pending | filled` rows
- **Tests**:
  - Full lifecycle: `pending -> filled -> closed -> settled`
  - Cancellation path: `pending -> cancelled`; order absent from counts immediately
  - `open_positions_by_token()` with mixed-status rows for same token_id
  - `total_exposure()` excludes cancelled and settled rows
- **Source refs**:
  - `grill-tasks.md:218` (`core/positions.py` -- all 8 ADD items, L219-235)
- **Out of scope**:
  Anything that calls `mark_settled` (V12); RiskPipeline integration (V06);
  orders table creation (F01)

---

### F04 -- ProbabilityStore

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Create `core/probability_store.py` with `ProbabilityStore`.
  Store key: `(condition_id: str, day_offset: int)` where `day_offset in {0, 1, 2}`.
  Value per key: `dict[tuple[float, float], BucketPayload]` where bucket key is
  `(lo_f, hi_f)` and `BucketPayload` contains `mc_prob`, `quantiles_by_fxx`, and
  `last_tick_utc` (shared across all buckets for the same key).
  `asyncio.Lock` guards all writes.
- **Acceptance criteria**:
  - [ ] Read after write returns correct payload
  - [ ] `day_offset` outside `{0, 1, 2}` raises `ValueError`
  - [ ] `asyncio.Lock` is acquired on write; concurrent write does not corrupt state
  - [ ] `last_tick_utc` is shared across all buckets for the same `(condition_id, day_offset)` key
  - [ ] Bucket key is `(lo_f, hi_f)` float tuple, NOT part of the store key
- **Tests**:
  - Write and read back full `BucketPayload` for two different `day_offset` values
  - Concurrent write test with `asyncio.gather`
  - `day_offset` validation
- **Source refs**:
  - `grill-tasks.md:202` (`core/probability_store.py [NEW]`, L203-213) [ADR-0004]
- **Out of scope**:
  ForecastLoop writes (V10); TradingLoop reads (V11/F05)

---

### F05 -- TradingLoop Skeleton

- **Type**: AFK
- **Blocked by**: F01, F02, F04
- **What to build**:
  Create `loops/trading_loop.py` with the `TradingLoop` class: constructor wires
  cadence from `settings.trading_loop_interval_s`, holds a `ProbabilityStore` reference
  and an `AlertBus` reference. Tick method reads `last_tick_utc` from `ProbabilityStore`
  and applies the staleness gate: when `now - last_tick_utc > staleness_threshold_min`,
  emits `AlertEvent(alert_type="data_stale", ...)` via `AlertBus` and skips order
  submission. Does NOT yet connect to MultiStrategy, RiskPipeline, or any exchange.
- **Acceptance criteria**:
  - [ ] `TradingLoop` can be instantiated with a fake `ProbabilityStore` and `AlertBus`
  - [ ] Staleness gate fires and emits `data_stale` alert when age exceeds threshold
  - [ ] No order submission code exists in this slice (stub raises NotImplementedError or is absent)
  - [ ] Cadence reads from `settings.trading_loop_interval_s`
- **Tests**:
  - Staleness gate: inject stale `last_tick_utc`, verify `data_stale` alert emitted
  - Non-stale data: verify alert not emitted
  - Cadence value sourced from settings
- **Source refs**:
  - `grill-tasks.md:173` (`loops/trading_loop.py [NEW]`) -- class skeleton + cadence (L176), staleness gate (L178), data_stale alert (L180) [ADR-0004, ADR-0006]
- **Out of scope**:
  MultiStrategy/RiskPipeline/exchange wiring (V11); YES/NO token handling (V11);
  `risk_rejected` alert (V11); full order submission (V11)

---

## HITL Slice

---

### H01 -- Station Grid Lookup (6 New Stations)

- **Type**: HITL
- **Blocked by**: None (human action; code work can start any time)
- **What to build**:
  Add Toronto, Atlanta, Chicago, Seattle, Dallas, and Miami to `config/stations.py`
  with correct HRRR grid `row` and `col` values. Once row/col values are confirmed
  via external HRRR grid lookup, `_grib_source.py` (V02) picks them up automatically
  from the STATIONS dict -- no additional wiring slice needed.
- **Acceptance criteria**:
  - [ ] All 6 stations have valid `hrrr_grid_row_zarr` and `hrrr_grid_col_zarr` values confirmed against actual HRRR grid
  - [ ] Station entries follow existing dict structure in `config/stations.py`
- **HITL reason**: Row/col values require running a HRRR grid-point lookup script
  against real HRRR data or a known reference grid. Not derivable from code alone.
- **Source refs**:
  - `grill-tasks.md:52` (`config/stations.py` -- ADD 6 stations, L53-55)
- **Out of scope**:
  Wiring new stations into _grib_source (automatic once STATIONS dict is populated)

---

## Vertical Slices

Each vertical slice is end-to-end verifiable on its own.

---

### V01 -- Sniper Manager Rename

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Rename `snipers/manager.py` to `snipers/sniper_manager.py`.
  Update all import sites: `from snipers.manager import` becomes
  `from snipers.sniper_manager import`.
- **Acceptance criteria**:
  - [ ] `snipers/manager.py` no longer exists
  - [ ] All import sites updated; no `ImportError` on startup
  - [ ] Existing sniper tests pass unchanged
- **Tests**: Existing tests suffice; verify no `ImportError` after rename.
- **Source refs**:
  - `grill-tasks.md:363` (`snipers/sniper_manager.py [RENAME]`, L364-368)
- **Out of scope**: No logic changes; rename only

---

### V02 -- HRRR Multi-Station Pipeline + HrrrFill + Parquet Enforcement

- **Type**: AFK
- **Blocked by**: F01 (settings: `hrrr_source_priority`, `cold_start_hours`)
- **What to build**:
  Create `data/forecast/hrrr_fill.py` with `HrrrFill`: splices non-MAJOR 18h HRRR
  to 48h using the nearest prior MAJOR-hour run aligned by `valid_time`; falls back
  to the MAJOR-hour before that for tail gaps; logs WARNING on fallback [ADR-0005].
  Overhaul `_grib_source.py`: remove hardcoded `_klga_idx`; change `fetch` to return
  `dict[station_id, pd.DataFrame]` decoding all STATIONS in one eccodes pass;
  wire `HrrrFill` for non-MAJOR init_times; add auto-switch fallback on download/decode
  failure; add gap-detection; add structured logging for all failure modes.
  Change `hrrr_zarr.py` to batch-download only (remove from production fallback chain).
  Enforce f01-only in `parquet_store.py` [ADR-0003].
- **Acceptance criteria**:
  - [ ] `_grib_source.fetch()` returns data for all configured STATIONS in one call
  - [ ] Non-MAJOR init_time triggers `HrrrFill`; tail-gap fallback logs WARNING
  - [ ] Source fallback switches automatically on download failure (no manual intervention)
  - [ ] `parquet_store.py` drops any fxx > 1 on write
  - [ ] `hrrr_zarr.py` remains importable for scripts/batch use, but is not included in the production fallback selection
  - [ ] New stations from H01 automatically appear in output without code change
- **Tests**:
  - `HrrrFill`: correct splice alignment; tail-gap WARNING emitted; no valid_time gaps in output
  - `_grib_source`: multi-station dict returned; fallback triggered on injected download error
  - `parquet_store.py`: fxx=2 row silently dropped on write
- **Source refs**:
  - `grill-tasks.md:72` (`data/forecast/hrrr_fill.py [NEW]`, L73-78) [ADR-0005]
  - `grill-tasks.md:79` (`data/forecast/hrrr_zarr.py` -- CHANGE to batch-only, L80-82)
  - `grill-tasks.md:83` (`data/forecast/_grib_source.py` -- all 7 items, L84-93)
  - `grill-tasks.md:114` (`data/storage/parquet_store.py` -- ENFORCE f01-only, L115) [ADR-0003]
- **Out of scope**: Cleaner feature engineering (V03); new station data (H01)

---

### V03 -- Cleaner + DataManager + build_encoder_obs_window

- **Type**: AFK
- **Blocked by**: F01 (`cold_start_hours`), V02 (Cleaner reads parquet that _grib_source writes)
- **What to build**:
  Wire `data/storage/cleaner.py` into the data flow (currently isolated).
  Cleaner owns all feature engineering: K->C (HRRR temp), F->C (IEM temp),
  wind decomposition, log1p smoothing, `temp_error = IEM_obs - f01_HRRR`.
  Drop cold-start rows (first `cold_start_hours` rows per station/dataset).
  Save processed output back to ParquetStore replacing raw columns.
  In `data/manager.py`: remove any duplicate feature transforms; thread station
  parameter through all fetch calls; add `build_encoder_obs_window(station, init_time)
  -> pd.DataFrame` that reads the last 72 hourly f01 HRRR + IEM obs from ParquetStore,
  merges on `valid_time`, and raises if fewer than `cold_start_hours` rows available
  for either series.
- **Acceptance criteria**:
  - [ ] Cleaner is called in the data flow before data reaches model layer
  - [ ] K->C and F->C conversions correct to 2 decimal places
  - [ ] Wind decomposition produces `u_wind` and `v_wind` columns
  - [ ] `temp_error` column = IEM_obs - f01_HRRR at matching `valid_time`
  - [ ] Cold-start rows (first 72h) absent from output
  - [ ] `build_encoder_obs_window` returns exactly 72 merged rows when data is sufficient
  - [ ] `build_encoder_obs_window` raises when fewer than 72 rows available
  - [ ] No feature transform code remains in `data/manager.py`
- **Tests**:
  - Cleaner: K->C, F->C, wind decomp, log1p, cold-start drop, output shape
  - `build_encoder_obs_window`: 72-row case; under-72-row raises
- **Source refs**:
  - `grill-tasks.md:107` (`data/storage/cleaner.py` -- WIRE, OWN, DROP, SAVE, L108-112)
  - `grill-tasks.md:117` (`data/manager.py` -- REMOVE, ADD station param, ADD build_encoder_obs_window, L118-124)
- **Out of scope**: Model pipeline consuming the window (V05b); IEM obs (V04, independent)

---

### V04 -- IEM + NOAA Obs Hardening

- **Type**: AFK
- **Blocked by**: F01 (settings)
- **What to build**:
  Change `data/obs/iem.py` `fetch()` to retrieve only the `:51` reading for hour H-1
  (the most accurate representative value); do not re-download past 72h each tick.
  Fix fallback: switch automatically when data is incomplete, not only when all-NaN.
  Add fine-grained warnings (e.g. high-precision temp unavailable).
  Add METAR parse robustness to `data/obs/noaa_file.py` for multi-field numeric
  records and incomplete records; add fine-grained warnings.
- **Acceptance criteria**:
  - [ ] `IemSource.fetch` requests only `(now_utc - 1h).replace(minute=51)`
  - [ ] Fallback triggers on incomplete data, not only all-NaN response
  - [ ] Warning logged when high-precision temp unavailable
  - [ ] `noaa_file.py` does not crash on incomplete METAR records
- **Tests**:
  - `IemSource.fetch`: correct `:51` timestamp requested; fallback on partial data
  - `noaa_file.py`: incomplete METAR record parsed without exception; warning emitted
- **Source refs**:
  - `grill-tasks.md:95` (`data/obs/iem.py` -- CHANGE, FIX, ADD, L96-102)
  - `grill-tasks.md:103` (`data/obs/noaa_file.py` -- ADD robustness, ADD warnings, L104-106)
- **Out of scope**: How IEM data flows to Cleaner (V03, independent); settlement IEM read (V12)

---

### V05a -- MC + Registry Calibration

- **Type**: AFK
- **Blocked by**: F01 (`mc_n_samp` default change)
- **What to build**:
  Overhaul `model/mc.py`: remove AR(1) covariance and `bias_corr_c` [ADR-0002];
  change `n_samp` default to 10000; add `fit_cov(residuals: np.ndarray) -> None`;
  fix `bucket_probs` to return `p10_f`, `p50_f`, `p90_f` computed across all samples;
  add `max_bias` correction (subtract per-station `max_bias` from E[max(samples)]
  before bucket mapping).
  Fix `model/store/registry.py`: replace `db._conn` direct access with DbClient
  public interface; add `store_max_bias`, `load_max_bias`, `calibrate_max_bias`.
- **Acceptance criteria**:
  - [ ] AR(1) and `bias_corr_c` code absent from `mc.py`
  - [ ] `fit_cov` accepts residuals array and stores empirical covariance
  - [ ] `bucket_probs` returns quantiles across full sample set
  - [ ] `max_bias` applied before bucket mapping; `load_max_bias` returns 0.0 if uncalibrated
  - [ ] `registry._db_latest_row` uses only DbClient public interface
  - [ ] `calibrate_max_bias` stores result; subsequent `load_max_bias` returns stored value
- **Tests**:
  - MC: `fit_cov` with known residuals; `bucket_probs` quantile correctness; `max_bias` shifts probabilities correctly
  - Registry: `store/load` round-trip; `calibrate_max_bias` end-to-end with fake val residuals
- **Source refs**:
  - `grill-tasks.md:129` (`model/mc.py` -- 2 REMOVE, CHANGE, ADD fit_cov, FIX, ADD max_bias, L130-136) [ADR-0002]
  - `grill-tasks.md:152` (`model/store/registry.py` -- FIX, 3 ADD, L153-161)
- **Out of scope**: TFT/pipeline changes (V05b); backtest (V05c)

---

### V05b -- TFT + ModelPipeline 72h Assembly

- **Type**: AFK
- **Blocked by**: F01, V03 (build_encoder_obs_window), V04 (IEM obs), V05a (MC ready)
- **What to build**:
  Change `model/tft.py` `MAX_ENCODER_LENGTH` 24->72 [ADR-0001]; remove `_prep()`
  (moved to Cleaner in V03); fix line 277 un-residualise (add f01 back to get
  absolute temperature).
  Fix the three pipeline bugs (`fetch_hrrr` station arg, encoder window -71..0,
  merge tolerance 20 min) and add the four pipeline features (`time_idx` column,
  seasonal features, superset column alignment, staleness timestamp) in
  `model/pipeline.py`.
- **Acceptance criteria**:
  - [ ] `MAX_ENCODER_LENGTH == 72` and encoder window uses -71..0
  - [ ] `_prep()` removed from tft.py; no NameError
  - [ ] Pipeline output is absolute temperature (not residual)
  - [ ] `fetch_hrrr` receives `station` argument
  - [ ] Merge tolerance is 20 min
  - [ ] `time_idx` column present in assembled frames
  - [ ] Seasonal features present: month, day-of-year, hour-of-day
  - [ ] pipeline output includes `last_tick_utc` staleness timestamp
- **Tests**:
  - `pipeline.run()`: returns non-residual output on synthetic data
  - Encoder window length = 72 rows
  - Staleness timestamp present in `pipeline.run()` return value
- **Source refs**:
  - `grill-tasks.md:138` (`model/tft.py` -- CHANGE encoder length, MOVE _prep, FIX un-residualise, L139-141) [ADR-0001]
  - `grill-tasks.md:143` (`model/pipeline.py` -- 3 FIX L144-146 + 4 ADD L147-150)
- **Out of scope**: Backtest call site (V05c); ForecastLoop scheduling (V10)

---

### V05c -- Backtest Compatibility

- **Type**: AFK
- **Blocked by**: V05b (ModelPipeline.run() signature finalised)
- **What to build**:
  Update `backtest/engine.py`: add cold-start guard (skip first `cold_start_hours`
  rows instead of padding with zeros); update `self.pipeline.run(station, init_utc, buckets)`
  call site to match the new `ModelPipeline.run()` signature from V05b.
- **Acceptance criteria**:
  - [ ] Backtest skips first `cold_start_hours` rows with no padding
  - [ ] `engine._process_init()` call site matches new pipeline signature
  - [ ] Existing backtest tests pass (or are updated in same pass if signature changed)
- **Tests**:
  - Cold-start guard: first 72 rows absent from backtest output
  - `_process_init()` call: no TypeError on patched pipeline
- **Source refs**:
  - `grill-tasks.md:339` (`backtest/engine.py` -- ADD cold-start guard, CHANGE call site, L340-345)
- **Out of scope**: Backtest replay fidelity, extended test coverage (not on critical path per Testing strategy)

---

### V06 -- Strategy + Risk Stack

- **Type**: AFK
- **Blocked by**: F01 (`max_open_per_token`), F03 (PositionTracker: `open_positions_by_token`)
- **What to build**:
  Change `core/strategy/base.py` `compute_signals` to accept `open_positions: dict[str, int]`;
  add `name: str` abstract property.
  Fix `core/strategy/edge.py` to evaluate NO tokens (NO edge = `(1 - mc_prob) - clob_price_no`);
  add `name = "edge"`.
  Create `core/strategy/multi.py` with `MultiStrategy`: accepts non-empty list of strategies
  (ValueError on empty), merges and deduplicates signals by token_id (first wins), `name = "multi"`.
  Change `core/risk/base.py` signatures to add `open_positions` param to both
  `approve_signal` and `compute_size`.
  Add duplicate-position prevention to `core/risk/fixed_fraction.py` via `open_positions` param.
  Create `core/risk/kelly.py` with `KellyCriterion` stub (NotImplementedError).
  Create `core/risk/pipeline.py` with `RiskPipeline`: AND approve, min-positive size,
  ValueError on empty.
- **Acceptance criteria**:
  - [ ] `MultiStrategy` ValueError on empty list; dedup by token_id; `name == "multi"`
  - [ ] EdgeStrategy generates BUY for NO token when NO edge > `min_edge`
  - [ ] `FixedFraction` rejects when `open_positions.get(token_id, 0) >= max_open_per_token`
  - [ ] `RiskPipeline.approve_signal`: one rejection blocks all; `compute_size` not called on rejected signal
  - [ ] `RiskPipeline.compute_size`: returns minimum positive value across sub-models
  - [ ] `KellyCriterion.approve_signal` and `.compute_size` raise `NotImplementedError`
  - [ ] All existing `BaseStrategy` / `BaseRiskModel` subclass call sites updated to new signatures
- **Tests**:
  - `MultiStrategy.compute_signals`: merge N strategies, dedup
  - `EdgeStrategy.compute_signals`: YES and NO token paths
  - `FixedFraction`: reject when at-limit; approve when below limit
  - `RiskPipeline.approve_signal`: AND logic; `RiskPipeline.compute_size`: min-positive
  - `KellyCriterion`: NotImplementedError with expected message
- **Source refs**:
  - `grill-tasks.md:253` (`core/strategy/base.py` -- CHANGE + ADD name, L254-258)
  - `grill-tasks.md:259` (`core/strategy/edge.py` -- FIX NO token + ADD name, L260-264)
  - `grill-tasks.md:265` (`core/strategy/multi.py [NEW]`, L266-270)
  - `grill-tasks.md:271` (`core/risk/base.py` -- 2 CHANGE, L272-275)
  - `grill-tasks.md:276` (`core/risk/fixed_fraction.py` -- ADD dup prevention, L277-281)
  - `grill-tasks.md:282` (`core/risk/kelly.py [NEW]`, L283-286)
  - `grill-tasks.md:287` (`core/risk/pipeline.py [NEW]`, L288-294)
- **Out of scope**: TradingLoop wiring (V11); AlertBus integration (V11)

---

### V07 -- CLOB Auth Split + Paper Exchange + bot.py

- **Type**: AFK
- **Blocked by**: F01 (settings), F02 (AlertBus -- order failure alert)
- **What to build**:
  Add `order_failure` alert emission to `core/exchange/clob.py` via `AlertBus`
  on submission exception, auth failure, or rejected-order response.
  Add `OrderbookFetcher` to `data/clob/orderbook.py` (wraps L1-only `ClobClient`,
  normalises bids/asks, exposes `get_book(token_id)`);
  use it in `core/exchange/paper.py`; emit `book_fetch_failure` alert when
  `get_book()` fails (the paper order still succeeds, filled at submitted price).
  Fix `core/bot.py` paper-mode data source wiring; change auth: live mode
  constructs full L2 `ClobClient`; paper mode constructs L1-only `ClobClient`,
  wraps as `OrderbookFetcher`, never loads L2 credentials [ADR-0008].
- **Acceptance criteria**:
  - [ ] Live mode: L2 `ClobClient` with ApiCreds wired to `ClobExchange`
  - [ ] Paper mode: L1-only `ClobClient` wrapped as `data.clob.orderbook.OrderbookFetcher`; no ApiCreds loaded
  - [ ] `order_failure` alert emitted on CLOB exception (`clob.py`)
  - [ ] `book_fetch_failure` alert emitted on `get_book()` failure (`paper.py`)
  - [ ] `bot.py` paper mode mounts real data sources to DataManager
  - [ ] Tests validate constructor wiring only (no real credentials required; AFK-safe)
- **Tests**:
  - `bot.py` live mode: ApiCreds passed to `ClobClient` constructor (mock ClobClient)
  - `bot.py` paper mode: no ApiCreds argument present (mock ClobClient)
  - `ClobExchange`: `order_failure` alert emitted on injected exception
  - `PaperExchange`: `book_fetch_failure` alert emitted on `get_book()` failure
- **Source refs**:
  - `grill-tasks.md:295` (`core/exchange/clob.py` -- ADD order_failure alert, L296-301)
  - `grill-tasks.md:302` (`data/clob/orderbook.py` -- OrderbookFetcher; `core/exchange/paper.py` -- ADD book_fetch_failure alert, L303-309)
  - `grill-tasks.md:310` (`core/bot.py` -- FIX paper mode + CHANGE auth, L311-315) [ADR-0008]
- **Out of scope**: TradingLoop calling these exchanges (V11); full credential rotation

---

### V08 -- Slug Parser Registry

- **Type**: AFK
- **Blocked by**: None -- can start immediately
- **What to build**:
  Refactor `events/discovery.py` `parse_slug`: replace ad-hoc if/elif with a
  module-level list of `(compiled_re, handler: Callable[[re.Match], tuple[float, float]])`
  entries. Parser iterates the list, returns first match result, logs WARNING and
  returns `None` on no match. Add registry entries for all known patterns:
  `or-above`, `or-higher`, `or-below`, `be-{N}f-or-below`, and observed variants.
- **Acceptance criteria**:
  - [ ] No if/elif slug branching remains in `parse_slug`
  - [ ] All known patterns return correct `(lo, hi)` tuple
  - [ ] Unmatched slug returns `None` and logs WARNING
  - [ ] New patterns can be added by appending to the list, no other code change
- **Tests**:
  - Each known pattern returns correct `(lo, hi)` tuple
  - Unknown slug returns `None` and WARNING logged
- **Source refs**:
  - `grill-tasks.md:193` (`events/discovery.py` -- CHANGE parse_slug to registry + ADD entries, L194-200)
- **Out of scope**: DiscoveryLoop scheduling (V09); new slug patterns beyond observed production

---

### V09 -- Discovery Loop Overhaul

- **Type**: AFK
- **Blocked by**: F01 (cutoff settings removed; ensures no dangling reference)
- **What to build**:
  Change `loops/discovery_loop.py` cadence from daily to hourly.
  Add deduplication: skip Events/Markets already present in DB.
  Remove any cutoff-hour logic.
- **Acceptance criteria**:
  - [ ] Loop fires hourly (not daily)
  - [ ] Event already in DB is not re-inserted or re-discovered
  - [ ] No reference to `cutoff_hour_utc` or `min_hours_before_cutoff` remains
- **Tests**:
  - Dedup: second discovery call with same event_id results in no DB write
  - No import or usage of removed settings fields
- **Source refs**:
  - `grill-tasks.md:188` (`loops/discovery_loop.py` -- CHANGE cadence, ADD dedup, REMOVE cutoff, L189-192)
- **Out of scope**: Slug parsing (V08); EventLifecycle cutoff removal (V12, independent)

---

### V10 -- ForecastLoop Alignment + ProbabilityStore Writes

- **Type**: AFK
- **Blocked by**: F01, F04 (ProbabilityStore), V05b (ModelPipeline ready)
- **What to build**:
  Split the TradingLoop portion out of `loops/forecast_loop.py` (TradingLoop is F05/V11).
  Change ForecastLoop cadence to wall-clock UTC hour alignment (not `sleep(3600)`).
  After each pipeline run, write output to `ProbabilityStore` for each active Event
  keyed by `(condition_id, day_offset)` where `day_offset in {0, 1, 2}`; write
  `mc_prob`, TFT `p10/p50/p90`, and `last_tick_utc` per bucket [ADR-0004].
- **Acceptance criteria**:
  - [ ] ForecastLoop fires within +-5 s of the wall-clock UTC hour
  - [ ] ProbabilityStore updated for each active Event after each run
  - [ ] `day_offset` values: today=0, tomorrow=1, day-after=2
  - [ ] `last_tick_utc` written once per run (shared across all buckets for the key)
  - [ ] TradingLoop code is NOT in forecast_loop.py after this slice
- **Tests**:
  - Wall-clock alignment: mock `asyncio.sleep`; verify next-tick calculation within 5 s
  - ProbabilityStore write: after one simulated run, verify all 3 day_offsets populated
- **Source refs**:
  - `grill-tasks.md:166` (`loops/forecast_loop.py` -- SPLIT, CHANGE cadence, ADD ProbStore write, L167-172) [ADR-0004]
- **Out of scope**: TradingLoop that reads ProbabilityStore (V11); ForecastLoop scheduling of model (V05b)

---

### V11 -- TradingLoop Full Wiring

- **Type**: AFK
- **Blocked by**: F01, F02, F03, F04, F05 (skeleton), V06 (Strategy+Risk), V07 (exchanges), V10 (ProbStore populated)
- **What to build**:
  Complete `loops/trading_loop.py` with full wiring: iterates active Events from
  the event DB, reads matching `ProbabilityStore` entries for each event; iterates
  YES and NO tokens; runs signals through the single
  `MultiStrategy` instance; pipes results through `RiskPipeline`; emits
  `risk_rejected` alert on rejection; on successful exchange submission calls
  `PositionTracker.record_order` with the returned order_id; on submission failure
  skips `record_order` and lets the exchange-layer exception propagate to the loop
  logger (`order_failure` alert is emitted by the exchange, not TradingLoop);
  enforces exactly
  one `MultiStrategy` and one `RiskPipeline` in constructor (TypeError if wrong
  type passed).
- **Acceptance criteria**:
  - [ ] TradingLoop holds exactly one `MultiStrategy` and one `RiskPipeline`
  - [ ] Both YES and NO tokens processed per active Event
  - [ ] `risk_rejected` alert emitted when `RiskPipeline.approve_signal` returns False
  - [ ] On successful submit: `record_order` called with exchange-returned order_id; on failure: no `record_order` call, exception logged (exchange layer owns `order_failure` alert)
  - [ ] Staleness gate (from F05) still active; stale = no order submission
  - [ ] Constructor raises TypeError when passed bare `BaseStrategy` or `BaseRiskModel`
- **Tests**:
  - Full tick: stale data -> no submission; fresh data -> MultiStrategy -> RiskPipeline -> exchange called
  - `risk_rejected` alert emitted with correct alert_type
  - YES and NO signal paths both produce exchange calls on approval
  - Successful submit: `record_order` called with exchange order_id after submit returns; failed submit: no `record_order` call, exception propagates to loop logger
- **Source refs**:
  - `grill-tasks.md:173` (`loops/trading_loop.py [NEW]`) -- active Events iteration (L174), YES/NO tokens (L177), risk_rejected alert (L183), MultiStrategy enforcement (L185) [ADR-0004, ADR-0006]
- **Out of scope**: UI staleness display (V13); SettlementLoop (V12); Monitor bodies (Deferred)

---

### V12 -- Event Lifecycle + Settlement Loop

- **Type**: AFK
- **Blocked by**: F03 (PositionTracker.mark_settled), V03+V04 (IEM obs in parquet for temp confirmation)
- **What to build**:
  Remove `_CUTOFF_HOUR_UTC` and `min_hours_before_cutoff` gating from
  `events/lifecycle.py`; change `advance()` so `"pending_settle"` entered only when
  `event_end_date < now_utc` (midnight boundary); add `set_settled(event_id)`.
  Create `loops/settlement_loop.py`: `confirm_settlement(event, station)` reads
  IEM ParquetStore for observed max temp on event date; compares to bucket boundaries;
  calls `mark_settled` for all `filled | closed` orders in `condition_id`; calls
  `EventLifecycle.set_settled(event_id)`.
  Add `SettlementRelayer` stub in `core/exchange/settlement.py`: single abstract
  method `resolve_on_chain(event_id) -> bool`; raises `NotImplementedError`
  referencing ADR-0007.
  Wire `SettlementLoop` to call `confirm_settlement` for each `"pending_settle"` Event
  [ADR-0007].
- **Acceptance criteria**:
  - [ ] `advance()` transitions to `pending_settle` on midnight boundary only
  - [ ] `set_settled()` writes `"settled"` to DB; NOT called by `advance()`
  - [ ] `confirm_settlement` identifies correct winning bucket from IEM max temp
  - [ ] Winning-bucket orders receive correct payout; losing-bucket = 0
  - [ ] `set_settled()` called exactly once per event after all orders marked
  - [ ] `SettlementRelayer.resolve_on_chain` raises NotImplementedError
- **Tests**:
  - Lifecycle: `advance()` at `event_end_date + 1s` enters `pending_settle`; no cutoff-hour check
  - Settlement: correct bucket from injected IEM temp; winning vs losing payout; `set_settled` called once
  - Relayer stub: NotImplementedError with ADR-0007 reference
- **Source refs**:
  - `grill-tasks.md:328` (`events/lifecycle.py` -- REMOVE cutoff gating, CHANGE advance(), ADD set_settled(), L329-334) [ADR-0007]
  - `grill-tasks.md:349` (`loops/settlement_loop.py [NEW]` -- 4 ADD + 1 WIRE, L350-362) [ADR-0007]
- **Out of scope**: On-chain SettlementRelayer implementation (Deferred); Telegram payout notification (Deferred)

---

### V13 -- UI AlertBus Integration

- **Type**: AFK
- **Blocked by**: F02 (AlertBus), F05 (TradingLoop Skeleton emits data_stale)
- **What to build**:
  Subscribe `ui/app.py` to `AlertBus` for `"data_stale"` events; show banner or
  status indicator in UI when alert fires.
  Register Telegram stub subscriber with `AlertBus` at startup: handler is a no-op
  that logs at DEBUG level only (`"Telegram not yet implemented"`).
- **Acceptance criteria**:
  - [ ] UI shows staleness indicator when `"data_stale"` alert is received
  - [ ] Telegram stub registered at startup; fires on any alert; no exception raised
  - [ ] Telegram stub logs at DEBUG level with correct message
  - [ ] Full Telegram routing not implemented (stub only)
- **Tests**:
  - Inject `AlertEvent(alert_type="data_stale", ...)` into AlertBus; verify UI state updates
  - Telegram stub: handler called; DEBUG log emitted; no exception
- **Source refs**:
  - `grill-tasks.md:317` (`ui/app.py` -- ADD staleness display, ADD Telegram stub, L318-323)
- **Out of scope**: Full Telegram routing, notification dedup, recovery alerts, dashboard (all Deferred)

---

## Coverage Matrix

| Source item (grill-tasks.md) | Slice | Status |
|---|---|---|
| settings.py -- REMOVE `klga_grid_row`, `klga_grid_col` | F01 | Covered |
| settings.py -- REMOVE `mc_rho`, `mc_bias_corr_c` | F01 | Covered |
| settings.py -- REMOVE `cutoff_hour_utc`, `min_hours_before_cutoff` | F01 | Covered |
| settings.py -- REMOVE `forecast_interval_min` | F01 | Covered |
| settings.py -- REMOVE `clob_window_min` | F01 | Covered |
| settings.py -- CHANGE `mc_n_samp` 5000 -> 10000 | F01 | Covered |
| settings.py -- ADD `trading_loop_interval_s` | F01 | Covered |
| settings.py -- ADD `staleness_threshold_min` | F01 | Covered |
| settings.py -- ADD `staleness_edge_bump` | F01 | Covered |
| settings.py -- ADD `hrrr_source_priority` | F01 | Covered |
| settings.py -- ADD `max_open_per_token` | F01 | Covered |
| settings.py -- ADD `cold_start_hours` | F01 | Covered |
| settings.py -- ADD `enable_sniper` | F01 | Covered |
| settings.py -- ADD `train_start` | F01 | Covered |
| settings.py -- ADD `train_end` | F01 | Covered |
| settings.py -- ADD `val_start` | F01 | Covered |
| settings.py -- ADD `val_end` | F01 | Covered |
| settings.py -- ADD `test_start` | F01 | Covered |
| settings.py -- ADD `test_end` | F01 | Covered |
| stations.py -- ADD 6 stations (row/col TBD) | H01 | Blocked/HITL |
| db/schema.py -- ADD `orders` table | F01 | Covered |
| hrrr_fill.py [NEW] -- ADD HrrrFill splice | V02 | Covered |
| hrrr_fill.py [NEW] -- ADD tail-gap fallback [ADR-0005] | V02 | Covered |
| hrrr_zarr.py -- CHANGE to batch-only | V02 | Covered |
| _grib_source.py -- REMOVE hardcoded `_klga_idx` | V02 | Covered |
| _grib_source.py -- CHANGE fetch() -> dict[station_id, DataFrame] | V02 | Covered |
| _grib_source.py -- ADD batch multi-station index lookup | V02 | Covered |
| _grib_source.py -- WIRE HrrrFill for non-MAJOR | V02 | Covered |
| _grib_source.py -- ADD auto-switch fallback | V02 | Covered |
| _grib_source.py -- ADD error handling + logging | V02 | Covered |
| _grib_source.py -- ADD gap-detection | V02 | Covered |
| iem.py -- CHANGE :51-only fetch | V04 | Covered |
| iem.py -- FIX fallback on incomplete data | V04 | Covered |
| iem.py -- ADD fine-grained warnings | V04 | Covered |
| noaa_file.py -- ADD METAR parse robustness | V04 | Covered |
| noaa_file.py -- ADD fine-grained warnings | V04 | Covered |
| cleaner.py -- WIRE into data flow | V03 | Covered |
| cleaner.py -- OWN feature engineering (K->C, F->C, wind, log1p, temp_error) | V03 | Covered |
| cleaner.py -- DROP cold-start rows | V03 | Covered |
| cleaner.py -- SAVE to ParquetStore | V03 | Covered |
| parquet_store.py -- ENFORCE f01-only [ADR-0003] | V02 | Covered |
| manager.py -- REMOVE dup feature transforms | V03 | Covered |
| manager.py -- ADD station param threading | V03 | Covered |
| manager.py -- ADD build_encoder_obs_window | V03 | Covered |
| mc.py -- REMOVE AR(1) covariance [ADR-0002] | V05a | Covered |
| mc.py -- REMOVE `bias_corr_c` [ADR-0002] | V05a | Covered |
| mc.py -- CHANGE n_samp default to 10000 | V05a | Covered |
| mc.py -- ADD fit_cov | V05a | Covered |
| mc.py -- FIX bucket_probs quantiles | V05a | Covered |
| mc.py -- ADD max_bias apply | V05a | Covered |
| tft.py -- CHANGE MAX_ENCODER_LENGTH 24->72 [ADR-0001] | V05b | Covered |
| tft.py -- MOVE _prep() to cleaner.py | V05b | Covered |
| tft.py -- FIX un-residualise line 277 | V05b | Covered |
| pipeline.py -- FIX fetch_hrrr station arg | V05b | Covered |
| pipeline.py -- FIX encoder window -71..0 | V05b | Covered |
| pipeline.py -- FIX assemble_frames merge tolerance 20 min | V05b | Covered |
| pipeline.py -- ADD time_idx column | V05b | Covered |
| pipeline.py -- ADD seasonal/periodic features | V05b | Covered |
| pipeline.py -- ADD superset column alignment | V05b | Covered |
| pipeline.py -- ADD staleness timestamp to pipeline output | V05b | Covered |
| registry.py -- FIX _db_latest_row DbClient interface | V05a | Covered |
| registry.py -- ADD store_max_bias | V05a | Covered |
| registry.py -- ADD load_max_bias | V05a | Covered |
| registry.py -- ADD calibrate_max_bias | V05a | Covered |
| forecast_loop.py -- SPLIT (TradingLoop out) | V10 | Covered |
| forecast_loop.py -- CHANGE wall-clock UTC alignment | V10 | Covered |
| forecast_loop.py -- ADD ProbabilityStore write [ADR-0004] | V10 | Covered |
| trading_loop.py [NEW]:173 -- class skeleton + ProbStore/AlertBus refs | F05 | Covered |
| trading_loop.py [NEW]:176 -- CHANGE cadence from settings | F05 | Covered |
| trading_loop.py [NEW]:178 -- ADD staleness gate [ADR-0006] | F05 | Covered |
| trading_loop.py [NEW]:180 -- ADD data_stale alert emission | F05 | Covered |
| trading_loop.py [NEW]:174 -- ADD active Events iteration (full behavior) | V11 | Covered |
| trading_loop.py [NEW]:177 -- ADD YES and NO token handling | V11 | Covered |
| trading_loop.py [NEW]:183 -- ADD risk_rejected alert | V11 | Covered |
| trading_loop.py [NEW]:185 -- ENFORCE MultiStrategy [ADR-0004] | V11 | Covered |
| discovery_loop.py -- CHANGE daily -> hourly | V09 | Covered |
| discovery_loop.py -- ADD deduplication | V09 | Covered |
| discovery_loop.py -- REMOVE cutoff-hour logic | V09 | Covered |
| events/discovery.py -- CHANGE parse_slug to registry | V08 | Covered |
| events/discovery.py -- ADD registry entries (all known patterns) | V08 | Covered |
| probability_store.py [NEW] -- ADD ProbabilityStore [ADR-0004] | F04 | Covered |
| positions.py -- ADD record_order | F03 | Covered |
| positions.py -- ADD mark_filled | F03 | Covered |
| positions.py -- ADD mark_closed | F03 | Covered |
| positions.py -- ADD mark_cancelled | F03 | Covered |
| positions.py -- ADD mark_settled | F03 | Covered |
| positions.py -- ADD open_positions_by_token | F03 | Covered |
| positions.py -- ADD total_exposure | F03 | Covered |
| positions.py -- ADD exposure_by_condition | F03 | Covered |
| alerts.py [NEW] -- ADD AlertEvent dataclass | F02 | Covered |
| alerts.py [NEW] -- ADD AlertBus (subscribe/emit/exception-per-task) | F02 | Covered |
| strategy/base.py -- CHANGE compute_signals (open_positions) | V06 | Covered |
| strategy/base.py -- ADD name abstract property | V06 | Covered |
| strategy/edge.py -- FIX NO token evaluation | V06 | Covered |
| strategy/edge.py -- ADD name property | V06 | Covered |
| strategy/multi.py [NEW] -- ADD MultiStrategy | V06 | Covered |
| risk/base.py -- CHANGE approve_signal (open_positions) | V06 | Covered |
| risk/base.py -- CHANGE compute_size (open_positions) | V06 | Covered |
| risk/fixed_fraction.py -- ADD dup-position prevention | V06 | Covered |
| risk/kelly.py [NEW] -- ADD KellyCriterion stub | V06 | Covered |
| risk/pipeline.py [NEW] -- ADD RiskPipeline | V06 | Covered |
| exchange/clob.py -- ADD order_failure alert | V07 | Covered |
| data/clob/orderbook.py -- ADD OrderbookFetcher normalising wrapper | V07 | Covered |
| exchange/paper.py -- CHANGE to use data.clob.orderbook.OrderbookFetcher | V07 | Covered |
| exchange/paper.py -- ADD book_fetch_failure alert | V07 | Covered |
| bot.py -- FIX paper mode data sources | V07 | Covered |
| bot.py -- CHANGE live/paper auth wire-up [ADR-0008] | V07 | Covered |
| ui/app.py -- ADD staleness alert display | V13 | Covered |
| ui/app.py -- ADD Telegram stub subscriber | V13 | Covered |
| lifecycle.py -- REMOVE cutoff gating | V12 | Covered |
| lifecycle.py -- CHANGE advance() midnight-only | V12 | Covered |
| lifecycle.py -- ADD set_settled() [ADR-0007] | V12 | Covered |
| backtest/engine.py -- ADD cold-start guard | V05c | Covered |
| backtest/engine.py -- CHANGE pipeline.run() call site | V05c | Covered |
| settlement_loop.py [NEW] -- ADD confirm_settlement [ADR-0007] | V12 | Covered |
| settlement_loop.py [NEW] -- ADD mark_settled call for all orders | V12 | Covered |
| settlement_loop.py [NEW] -- ADD set_settled call | V12 | Covered |
| settlement_loop.py [NEW] -- ADD SettlementRelayer stub [ADR-0007] | V12 | Covered |
| settlement_loop.py [NEW] -- WIRE SettlementLoop | V12 | Covered |
| snipers/sniper_manager.py [RENAME] -- MOVE manager.py | V01 | Covered |
| snipers/sniper_manager.py [RENAME] -- CHANGE all import sites | V01 | Covered |
| Testing strategy -- Cleaner tests | V03 | Covered |
| Testing strategy -- build_encoder_obs_window tests | V03 | Covered |
| Testing strategy -- IemSource.fetch tests | V04 | Covered |
| Testing strategy -- ProbabilityStore tests | F04 | Covered |
| Testing strategy -- ForecastLoop alignment tests | V10 | Covered |
| Testing strategy -- TradingLoop staleness + cadence tests | F05 | Covered |
| Testing strategy -- TradingLoop full wiring tests | V11 | Covered |
| Testing strategy -- MultiStrategy tests | V06 | Covered |
| Testing strategy -- RiskPipeline tests | V06 | Covered |
| Testing strategy -- FixedFraction dup-position tests | V06 | Covered |
| Testing strategy -- EdgeStrategy NO token tests | V06 | Covered |
| Testing strategy -- PositionTracker lifecycle tests | F03 | Covered |
| Testing strategy -- parse_slug registry tests | V08 | Covered |
| Testing strategy -- SettlementLoop bucket/payout tests | V12 | Covered |
| Deferred: DataHealthMonitor | -- | Deferred |
| Deferred: ModelHealthMonitor | -- | Deferred |
| Deferred: SystemHealthMonitor | -- | Deferred |
| Deferred: Full Telegram routing / dedup / recovery / dashboard | -- | Deferred |

**Uncovered items: none**

---

## Dependency Graph (execution order)

```
Can start immediately (no deps):
  F01, F02, F04, V01, V08

After F01:
  F03, F05(*), V02, V04, V05a, V09
  (* F05 also needs F02, F04)

After F01 + F02 + F04:
  F05

After F01 + V02:
  V03

After F01 + F03:
  V06

After F01 + F02:
  V07

After V03 + V04 + V05a:
  V05b

After V05b:
  V05c
  V10 (also needs F01, F04)

After F03 + V03 + V04:
  V12

After F01 + F02 + F03 + F04 + F05 + V06 + V07 + V10:
  V11

After F02 + F05:
  V13

H01: any time (HITL; unblocks new stations in V02 automatically)
```

---

## Deferred Register

The following items from `docs/grill-tasks.md` Deferred section are explicitly
out of scope for this overhaul and must not be implemented in any slice above:

- `DataHealthMonitor`: beyond staleness gate; drift / completeness monitoring
- `ModelHealthMonitor`: prob sanity checks, calibration drift, bucket coverage
- `SystemHealthMonitor`: loop heartbeat, process health, resource monitoring
- Full Telegram routing, notification dedup, recovery alerts, dashboard
