# V2 Overhaul Decisions (2ndMajorOverhaul_05_30)

## Layer 1: db/ (V2)

### db/schema.py
- ADD `is_paper INTEGER NOT NULL DEFAULT 0` column to `orders` table; paper run
  writes 1, live run writes 0; PnL module filters by this column for per-mode analysis
  (#8)
- ADD `executable_price REAL` column to `orders` table: the VWAP quote computed at
  order submission time (from `compute_quote`); distinct from `fill_price` (actual
  or paper-simulated settlement price) and `price` (submitted limit price)
  (#5)
- ADD `filled_shares REAL` column to `orders` table: number of shares implied by
  `size_usd / fill_price`; written by `PositionTracker.mark_filled()`; used by PnL
  module for mark-to-market unrealized PnL computation (#9 / #10)
- ADD `signed_shares REAL` column to `orders` table: BUY fills write positive
  shares, SELL fills write negative shares; position and settlement logic must use
  signed shares rather than counting orders (#9)
- ADD `cash_flow_usd REAL` column to `orders` table: BUY fills write negative cash
  outflow, SELL fills write positive cash inflow; PnL must be computed from cash
  flows plus settlement/mark values, not from `settled_payout - size_usd` for every
  side (#9 / #10)
- CREATE TABLE `signals`: new table for tick-level decision log; columns:
  `signal_id TEXT PRIMARY KEY`, `yes_token_id TEXT`, `no_token_id TEXT`,
  `direction TEXT`, `mc_prob REAL`, `reference_price REAL`, `edge REAL`,
  `final_token_id TEXT`, `final_side TEXT`, `executable_price REAL`,
  `tick_utc TEXT`, `approved INTEGER`, `reject_reason TEXT`, `computed_size REAL`,
  `order_id TEXT`; written by TradingLoop as each directional signal progresses
  through arbitration -> risk -> size -> submit (#12)
- CREATE TABLE `redeem_tasks`: one row per `condition_id`; columns:
  `condition_id TEXT PRIMARY KEY`, `status TEXT`, `attempt_count INTEGER`,
  `last_error TEXT`, `created_at TEXT`, `updated_at TEXT`, `redeemed_at TEXT`;
  status domain: `pending | redeemed | failed | skipped_paper`. `RedeemLoop`
  retries `pending|failed` rows and marks the condition redeemed once
  `ctf_redeem_positions_call(condition_id)` succeeds (#73)
- ADD `discovery_status TEXT NOT NULL DEFAULT 'parsed'` to `markets`; allowed values
  include `parsed`, `unrecognized_slug`, and `invalid_market`. Forecast/trading/
  settlement queries only consume `parsed` markets with non-null bucket bounds
  (#15 / #18)

## Layer 1: config/

### config/settings.py
- REMOVE `polymarket_api_key`, `polymarket_api_secret`, `polymarket_passphrase`,
  `polymarket_funder`, `polymarket_proxy_address` (all five V1 credential fields)
  (#1) [ADR-0009]
- ADD `polymarket_private_key: str` (reads env POLYMARKET_PRIVATE_KEY; live mode only)
  (#1) [ADR-0009]
- ADD `polymarket_wallet_address: str` (reads env POLYMARKET_WALLET_ADDRESS; passed as
  `wallet` arg to `SecureClient.create`; empty string means use signer address)
  (#1) [ADR-0009]

---

## Layer 2: data/ (V2)

### data/forecast/hrrr_fill.py
- ADD `is_extrapolated: bool` column to the output DataFrame: rows filled from
  the nearest MAJOR run via valid_time alignment get `is_extrapolated=False`; rows
  whose tail gap cannot be covered by any MAJOR run and are filled by short-horizon
  smoothing extrapolation get `is_extrapolated=True`; MAJOR run output also gets
  `is_extrapolated=False` on all rows so downstream always finds this column (#28)
- ADD tail-gap smoothing: when the nearest MAJOR run does not cover the last N fxx
  hours of the non-MAJOR window, linearly extrapolate from the last available aligned
  value for each column; mark those rows `is_extrapolated=True`; cap extrapolation at
  3 hours beyond the last aligned value (beyond that: raise `RuntimeError`) (#28)
- FIX `_vt_index()` duplicate valid_time handling: sort input DataFrame by valid_time
  ascending and drop_duplicates(keep="last") before indexing; emit `WARNING` log if
  duplicates are found, including count and init_time (#34)
- FIX valid_time rounding: apply `.dt.floor("s")` to valid_time before alignment to
  ensure sub-second differences (e.g. 03:00:00 vs 03:00:00.000001) do not prevent
  matching; update comment to match (#35)
- FIX output column union: derive output columns as the union of non_major_df.columns
  and nearest_major_df.columns; columns present only in the major run are included
  (filled from major) rather than silently dropped (#33)

### data/forecast/_grib_source.py
- FIX `.idx` matching: replace substring search with structured field parse (split
  on `:`) and exact match on VAR + LEVEL + forecast descriptor; reject rows where
  VAR is a prefix of the target (e.g. reject `MAXTMP` when target is `TMP`) (#29)
- FIX duplicate colname match: if more than one `.idx` row matches a target column,
  raise `ValueError` with the matched rows; never write a colname result
  non-deterministically (#29)
- FIX Range response validation: accept only HTTP 206 as success for Range requests;
  if server returns 200, validate `Content-Range` header and verify returned byte
  count matches `end - start`; log ERROR and return `None` on mismatch (#30)
- FIX last-message end offset: replace hardcoded `off + 5_000_000` with open-ended
  Range (`bytes={start}-`) for the final `.idx` entry; avoids truncating large GRIB
  messages (#31)
- ADD failure log context: on the final retry failure in `_fetch_range()`, emit
  `WARNING` with `grib_url`, `start`, `end`, `status_code`, and exception message;
  single transient retries may remain silent (#32)

### data/forecast/hrrr_s3.py
- ADD `is_extrapolated=False` default column to all returned DataFrames so that
  MAJOR run output is consistent with `HrrrFill` output schema (#28)
- CHANGE base URL source: read `settings.hrrr_s3_bucket` instead of hardcoded
  constant; constructor accepts optional `base_url` override for testing (#36)

### data/forecast/hrrr_nomads.py
- ADD `is_extrapolated=False` default column to all returned DataFrames (#28)
- CHANGE base URL source: read `settings.hrrr_nomads_base` instead of hardcoded
  constant; constructor accepts optional `base_url` override for testing (#36)
- ADD unit test `test_hrrr_nomads_build_url`: verify URL construction for MAJOR
  and non-MAJOR init_times, matching `hrrr_s3` test pattern (#37)

### data/forecast/hrrr_zarr.py
- ADD `is_extrapolated=False` default column to all returned DataFrames for MAJOR
  init; non-MAJOR path delegates to `HrrrFill` which sets the column (#28)
- ADD per-variable failure logging: on `except` in variable read loop, emit
  `WARNING` with `init_time`, `station_id`, col, zarr_path, url; if the failed
  variable is `TMP_2m`, raise `RuntimeError` (fail fast) instead of filling NaN (#38)
- FIX `n_fcst_steps` probe failure: raise `RuntimeError` if `TMP_2m` shape
  exploration fails; do not fall back to a hardcoded default (#39)
- WIRE `HrrrFill` for non-MAJOR init: after downloading raw zarr data, call
  `HrrrFill.fill()` (same as GRIB sources) so non-MAJOR 18h windows are extended
  to 48h with `is_extrapolated` flag; aligns behaviour with `_grib_source.py` (#40)
- FIX station grid index fallback: remove fallback to KLGA defaults; raise
  `ValueError` if `station_config` is missing `hrrr_grid_row_zarr` or
  `hrrr_grid_col_zarr` (#41)
- ADD grid bounds check: after opening the zarr dataset, assert `row < shape[0]`
  and `col < shape[1]` for the station grid index; raise `ValueError` with
  station_id, row, col, and actual shape on failure (#42)
- ADD `DATA_COLS` / `ZARR_VARS` consistency assertion: at module load time, assert
  `set(DATA_COLS) == set(ZARR_VARS.keys())`; raise `AssertionError` with diff if
  they diverge (#43)
- CHANGE base URL source: read `settings.hrrr_zarr_bucket` instead of hardcoded
  constant; constructor accepts optional `base_url` override for testing (#44)

### data/storage/parquet_store.py
- FIX dedup key time normalization: before comparing existing vs new keys, apply
  `pd.to_datetime(..., utc=True)` to all timestamp columns in the dedup key tuple;
  prevents logical duplicates from being written when timestamps differ only in
  string representation or sub-second precision (#45)
- FIX read() directory side-effect: split `_station_dir()` into `_station_path()`
  (returns path, no mkdir) and `_ensure_station_dir()` (creates dir); `read()` uses
  `_station_path()`; `write()` calls `_ensure_station_dir()` first (#46)
- ADD explicit schema validation: in `_pick_dedup_key()`, if the DataFrame is
  neither HRRR nor has a `valid_time` column, raise `ValueError("ParquetStore
  requires either init_time+fxx (HRRR) or valid_time (obs)")` (#47)
- ADD TODO comment: `write()` is not safe for concurrent multi-process access; if
  backfill scripts and live bot ever run in parallel, replace `to_parquet()` with
  write-to-tmp + `os.replace()` and add a file lock (#48)

### data/obs/iem.py
- FIX `:51` valid_time representation: in `clean_iem()`, after rounding to the
  nearest hour, overwrite `valid_time` with the rounded value so downstream
  reads using hourly boundaries correctly include the observation (#49)
- FIX NOAA tick fallback window: when `now_utc` is provided, `NoaaFileSource.fetch()`
  must apply the same H-1 :51 tick window as `IemSource`; pass `now_utc` through
  to NOAA fallback path and use it to build the fetch window (#50)
- FIX `skyc1` missing/unknown fill: in `clean_iem()`, apply `ffill().fillna(0)` to
  `skyc1` after parsing; treat unknown codes as 0 (clear) (#51)
- FIX wind direction VRB/missing: in `clean_iem()`, set `drct = NaN` when raw value
  is `"VRB"` or missing before computing U/V components; do not treat VRB as 0 deg;
  same fix required in NOAA fallback path (#52)
- FIX U/V NaN fill order: move U/V `ffill()/bfill()` into `clean_iem()` before
  `build_features_iem()` derives `obs_WS`/`obs_WD_*`; `build_features_iem()` must
  not call `fillna(0)` on U/V directly (#53)

### data/obs/noaa_file.py
- CHANGE scope: restrict `NoaaFileSource` to now_utc tick fallback only; add guard
  that raises `ValueError` if caller requests a window wider than 2 hours; prevents
  NOAA from being silently used as a range fallback where it returns 0-1 rows (#55)
- FIX wind direction VRB/missing: same fix as `iem.py` (#52) -- in the NOAA METAR
  parser, set `drct = NaN` when raw value is `"VRB"` or missing before computing
  U/V components; do not treat VRB as 0 degrees (#52)

### data/storage/cleaner.py
- FIX sort order: in `clean_iem()`, sort by `valid_time` ascending before applying
  `ffill()`/`bfill()`; ensures directional fills are time-ordered (#57)
- CHANGE cold-start drop semantics: `cold_start_hours` is exclusively a data-pipeline
  burn-in period (Parquet/pipeline stabilization, not encoder prep). Do not call
  `Cleaner.drop_cold_start()` on each live single-tick append, because that would
  discard every incremental row. Batch/offline materialization may discard the first
  `cold_start_hours` rows once per station/dataset; live inference uses
  `DataManager.inference_readiness()` to enforce `raw_data_start + cold_start_hours +
  MAX_ENCODER_LENGTH` before the first prediction (#3)

### data/manager.py
- CHANGE `build_encoder_obs_window(station, init_time)`: read exactly the fixed
  `encoder_hours` rows prior to `init_time` from Parquet (callers pass
  `MAX_ENCODER_LENGTH`, currently 72; i.e.
  `[init_time - 72h, init_time - 1h]`); do NOT read extra rows and drop them;
  raise a precise error (`InsufficientEncoderData`) if the window is not fully
  covered by Cleaner-processed rows (#27)
- ADD `inference_readiness(station, init_time, encoder_hours=72) -> ReadinessStatus`:
  returns `ReadinessStatus(ready, reason, first_eligible_init)`;
  `reason` is one of `"cold_start" | "incomplete_hrrr_window" |
  "incomplete_obs_window" | "missing_required_columns" | "ok"`;
  derives the earliest Cleaner-retained HRRR/IEM row per station, computes
  `first_eligible_init = first_clean_row + encoder_hours`, and checks
  Cleaner-processed Parquet row coverage for both HRRR f01 and IEM obs series over
  `[init_time - encoder_hours * 1h, init_time - 1h]`. Do not add
  `cold_start_hours` again when the earliest row already comes from cleaned Parquet;
  that would double-count the burn-in (#3)
- ADD `is_inference_ready(station, init_time, encoder_hours=72) -> bool`: thin
  wrapper over `inference_readiness` returning `status.ready`; for BacktestEngine
  and other callers that only need a boolean gate (#3)
- ADD required-column guard in `build_encoder_obs_window()`: after merging HRRR and
  obs frames, assert presence of `valid_time`, `TMP_2m` (HRRR), and
  `raw_tmpc_tenths` (obs); raise `RuntimeError` with the set of missing columns if
  any are absent; also assert `temp_error` is computed and non-null (#58)
- ADD `.copy()` after store reads: immediately after `ParquetStore.read()` calls in
  `build_encoder_obs_window()`, call `.copy()` on the returned DataFrame before any
  in-place mutations to avoid shared-reference side effects (#59)
- CHANGE `_obs_is_complete()`: extend completeness check beyond temperature to
  include `obs_WS` (or U/V components), `skyc1`, and `p01i`; all four must be
  non-null in at least one row for the result to be considered complete; a single
  NOAA row with only temperature must not pass this gate; this is the shared
  completeness predicate called for both IEM and NOAA fallback paths (#54 / #56)

### data/clob/orderbook.py
- CHANGE `get_book(token_id)` implementation: replace `ClobClient.get_order_book(token_id)`
  with `PublicClient.get_order_book(token_id=token_id)`; return normalized dict with
  `bids` (descending by price), `asks` (ascending by price), `best_bid`, `best_ask`,
  `mid_price`; apply defensive sort regardless of SDK ordering  [ADR-0009] [ADR-0011]
- FIX bids ordering assumption: new SDK returns bids ascending (lowest first); any
  code reading `bids[0]` as best bid must use `bids[-1]` instead  [ADR-0009]
- FIX missing `best_bid` / `best_ask` fields: `SniperLoop` reads `best_ask` from the
  fetcher but the current fetcher never sets it; this is a hidden bug  [ADR-0011]

---

## Layer 3: model/

### model/features.py  [NEW]
- ADD `HRRR_KNOWN_REALS: list[str]`: canonical cleaned HRRR column names available
  in encoder and decoder (e.g. `TMP_2m`, `DPT_2m`, `WS_10m`,
  `log1p_GUST_sfc`); single source of truth for training and inference; no
  renamed/pipeline-internal names
- ADD `OBS_UNKNOWN_REALS: list[str]`: canonical obs-only columns available in the
  encoder but not decoder (e.g. `raw_tmpc_tenths`, `raw_dwpc_tenths`, `obs_WS`)
- ADD `TARGET = "temp_error"` and helper constants:
  `REQUIRED_ENCODER_SOURCE_COLUMNS = HRRR_KNOWN_REALS + OBS_UNKNOWN_REALS + [TARGET]`
  and `REQUIRED_DECODER_SOURCE_COLUMNS = HRRR_KNOWN_REALS`
- ADD `MODEL_GENERATED_COLUMNS`: `time_idx`, `local_idx`, `month`, `day_of_year`,
  `hour_of_day`, `sin_hour`, `cos_hour`, `sin_doy`, `cos_doy`, `init_time_str`,
  and `Node_ID`; these are computed by `_add_model_features()` and are not part of
  source-column validation (#21 / #26)
- NOTE: `model/features.py` must not import from any other project package; column
  rename is not introduced in V2. Upstream source modules and Cleaner already
  produce the canonical names used by training (`TMP_2m`, `raw_tmpc_tenths`, etc.).

### model/tft.py
- CHANGE column list sources: replace hardcoded `KNOWN_REALS`/`UNKNOWN_REALS` lists
  with imports from `model/features.py` (#26)
- ADD `predict()` input sort: sort encoder_df and decoder_df by `time_idx` ascending
  before passing to `TimeSeriesDataSet`; fail if `time_idx` column is missing (#64)
- ADD encoder/decoder row count assertions: raise `ValueError` if encoder row count
  != `MAX_ENCODER_LENGTH` or decoder row count != 48 (#62)
- ADD model output shape assertion: raise `ValueError` if prediction tensor shape
  does not match `(1, 48, n_quantiles)` (#62)
- FIX `TMP_2m` fill: remove `.fillna(0)` for `TMP_2m` in decoder; raise `ValueError`
  if any `TMP_2m` value is NaN (#61)
- ADD `fxx` completeness check in decoder: assert `set(decoder_df.fxx) == set(range(1,49))`
  before building `TimeSeriesDataSet`; raise `ValueError` on mismatch (#60)

### model/pipeline.py
- ADD `is_extrapolated` filter: when selecting fxx rows for the daily-high target
  computation, exclude rows where `is_extrapolated=True` before passing to
  `McSampler.bucket_probs()`; log count of excluded rows at DEBUG level; if the
  target-day fxx window becomes empty after this exclusion, raise `ValueError`
  so ForecastLoop skips the event instead of computing a max from extrapolated
  tail data (#28)
- CHANGE `_add_model_features()`: raise `ValueError` when `valid_time` column is
  missing from encoder or decoder frame; do not fill calendar features with 0 (#71)
- ADD pre-predict column validation: after `_add_model_features()` and before
  `tft.predict()`, assert all required source columns from `model/features.py` are present;
  raise `ValueError` with the set of missing columns; do NOT fill missing required
  columns with 0. Time-derived/generated columns are validated by
  `_add_model_features()` itself (#21 / #63)
- CHANGE `pipeline.run()` signature: accept `station_config: StationConfig` and
  `target_date`; `station_config` is looked up by `event["station_id"]`;
  `target_date` comes from the Event's `event_end_date`, while `day_offset`
  remains only the ProbabilityStore key. Remove reliance on `self.station` for
  multi-station correctness (#22 / #24 / #4) [ADR-0010]
- CHANGE `init_time` return value: always return a `pd.Timestamp` with UTC tzinfo;
  reject or convert naive timestamps at entry (#23 / #69)
- ADD decoder completeness gate: if `hrrr_df` filtered to decoder fxx range is empty
  or missing `TMP_2m`, raise `RuntimeError` (fail fast); do not construct a synthetic
  decoder with NaN HRRR columns (#25)
- ADD decoder sort: sort decoder rows by `fxx` ascending after filtering from
  `hrrr_df`; before calling `_add_model_features()` (#70)
- CHANGE column list source: replace hardcoded lists with imports from
  `model/features.py` (#26)
- REMOVE `tft=None` fallback branch from `ModelPipeline.__init__`: `ModelPipeline`
  no longer accepts `tft=None`; raise `TypeError` immediately if `tft` does not
  implement the `predict(encoder_df, decoder_df)` protocol. Production callers
  load a real `TftModel`; unit tests may pass lightweight stubs that implement
  the same protocol (#68)
- ADD encoder/decoder length pre-check: before calling `tft.predict()`, assert
  encoder row count == `MAX_ENCODER_LENGTH` (imported from `model/tft.py`) and
  decoder row count == 48; raise `ValueError` with actual vs expected counts (#69)
- ADD quantile sanity check: after `tft.predict()` returns quantile dict, assert
  fxx keys are exactly 1..48, values are finite temperatures, and each row is ordered
  `p10 <= p50 <= p90`; do not constrain temperature quantiles to [0, 1] (#69)
- FIX sigma formula: guard against negative or zero sigma from anomalous quantiles;
  if `q90 - q10 <= 0`, raise `ValueError` rather than producing a negative std (#69)
- CHANGE daily-high fxx selection: replace "future 48h global max" with per-event
  local-calendar-day window; derive target fxx hours as those whose valid_time falls
  within `[target_local_day 00:00, target_local_day 24:00)` in the station's local
  timezone (from `StationConfig.timezone`), converted to UTC; pass only those fxx
  rows to McSampler for the daily-max computation (#24) [ADR-0010]

### scripts/train_tft.py
- CHANGE column list source: replace hardcoded feature lists with imports from
  `model/features.py` (#26)

### model/store/registry.py
- ADD `load_covariance(version: str | None = None) -> np.ndarray | None`: look for
  a sibling `covariance.npy` beside the active TFT checkpoint; return the matrix if
  present, otherwise log WARNING "covariance.npy not found -- MC will use diagonal
  (uncorrelated) assumption" and return `None`. The registry does not mutate
  `McSampler`; `core/bot.py` wires the returned matrix into `mc.set_cov()` (#65)

### model/mc.py
- ADD `set_cov(cov_matrix: np.ndarray) -> None` for production wiring; `fit_cov()`
  remains for offline training use only. `core/bot.py` loads covariance via
  `ModelRegistry.load_covariance()` and calls `mc.set_cov()` when present (#65)
- ADD input validation in `fit_cov()`: raise `ValueError` if residuals array is
  empty or has fewer rows than the number of stations (#66)
- ADD guard in `_mc_sample()`: assert `mu_vec` and `sigma_vec` are finite and
  non-empty; raise `ValueError` on violation (#66)
- FIX `bucket_probs()` zero-division: guard against empty sample array and raise
  `ValueError` instead of dividing by 0 (#66)
- ADD bucket validation in `bucket_probs()`: assert buckets are well formed and
  non-overlapping; gaps are allowed because active market feeds may omit buckets,
  but overlaps raise `ValueError` with a description (#66)

---

## Layer 4: loops/ (V2)

### events/lifecycle.py
- ADD periodic `advance()` call over locally stored non-terminal Events, not only
  over the events returned by the live Gamma poll. `DiscoveryLoop` should first
  sync live discovery results, then query DB for all `status NOT IN ('settled',
  'cancelled')`, call `EventLifecycle.advance(event)`, and write status changes via
  `db.update_event_status(condition_id, new_status)`. This prevents the `live=True`
  API filter from hiding just-expired events that must move to `pending_settle`
  (#16 / #17)
- ADD transition logging: each status change emits one INFO log:
  `"lifecycle: {condition_id} {old_status} -> {new_status}"`; no AlertBus event
  (lifecycle transitions are expected; not actionable) (#16)
- REMOVE `cutoff` state from `EventLifecycle` state machine: valid states are
  `discovered | active | pending_settle | settled`; advance() must not produce
  `cutoff` or `closed` (#16)

### config/settings.py
- ADD `discovery_loop_interval_s: int = 300` if not already present: used by
  DiscoveryLoop sleep; lifecycle advance() runs on the same cadence (#16)

### config/stations.py
- ADD `slug_aliases: list[str]` field to `StationConfig` dataclass: list of slug
  substrings that unambiguously identify this station (e.g. `["klga", "new-york-lga",
  "laguardia"]`); used by `_infer_station()` in `events/discovery.py` (#13)

### events/discovery.py
- CHANGE market discovery HTTP call: replace raw `requests.get(gamma_api_url/events)`
  with `PublicClient.list_events(live=True, tag_slug="weather")` typed paginator
  [ADR-0009]
- ADD active/live filter to the SDK call (`live=True` or the SDK's equivalent active
  filter) so closed markets are not returned from the API (#17)
- CHANGE `_infer_station()`: replace fragile substring match with explicit alias lookup;
  each `StationConfig` in `config/stations.py` gains a `slug_aliases: list[str]` field;
  `_infer_station()` iterates aliases and returns the first match; raises `ValueError`
  (no warning + continue) if no alias matches (#13)
- FIX `_process_markets()` zip truncation: before zipping tokens with bucket keys,
  assert `len(tokens) == len(expected_outcomes)`; log ERROR and skip the market if
  lengths differ; do not silently drop tokens (#14)
- CHANGE unrecognized-slug behaviour: when `parse_slug()` returns None/invalid,
  write the market to DB with `discovery_status = "unrecognized_slug"` and emit an
  AlertBus `discovery_warn` event containing the raw slug; do NOT skip the upsert (#15)
- ADD downstream hard gate: `get_tradeable_events()` / `get_active_events()` query
  filters markets to `discovery_status = 'parsed'`, `bucket_lo_f IS NOT NULL`, and
  `bucket_hi_f IS NOT NULL`; no forecast/trading/settlement loop ever sees
  unrecognized-slug rows (#15 / #18)
- ADD discovery summary log at end of each poll cycle: one INFO line with counts of
  new events upserted, updated, skipped (already-known), and unrecognized-slug warns
  (#19)
- CHANGE slug filter: tighten `parse_slug()` regex so it requires both a station
  token (ICAO or alias) AND a temperature-range pattern; slugs matching only one
  component are treated as unrecognized (#20)

---

## Layer 5: core/ (V2)

### core/probability_store.py
- CHANGE `BucketPayload` structure: lift `quantiles_by_fxx` and `last_tick_utc` out of
  per-bucket payload; replace with `ProbabilitySnapshot(buckets: dict[tuple[float, float],
  float], quantiles_by_fxx: dict[int, dict[str, float]], last_tick_utc: datetime)`;
  `buckets` maps (lo_f, hi_f) -> mc_prob only; `quantiles_by_fxx` and `last_tick_utc`
  are key-level fields (one per condition_id/day_offset, not per bucket) (#2)

### core/pricing/quote.py  [NEW]
- ADD `compute_quote(book, side, size_usd) -> Quote`: walks L2 depth; BUY consumes
  asks in ascending price order, SELL consumes bids in descending price order;
  returns `Quote(vwap_price, max_fillable_usd, filled_shares, slippage_bps,
  best_bid, best_ask, mid_price, complete_fill: bool)` (#5) [ADR-0011]
- ADD `Quote` dataclass with the above fields (#5) [ADR-0011]
- NOTE: this module must not import from any other project package (leaf module)

### config/settings.py
- ADD `quote_size_usd: float = 50.0` (proxy order size used by EdgeStrategy when
  computing executable VWAP edge at signal time; shared default with RiskPipeline
  sizing)  [ADR-0011]

### core/types.py
- ADD `DirectionalSignal` dataclass for strategy output: fields include
  `yes_token_id`, `no_token_id`, `direction` (`"bullish"` or `"bearish"`),
  `model_prob`, `reference_price`, `edge`, bucket bounds, and `condition_id`.
  It contains no executable `side` because SELL-YES vs BUY-NO is decided later
  by TradingLoop (#6)
- ADD `OrderIntent` dataclass for execution/risk input: fields include
  `token_id`, `side`, `size_usd | None`, `executable_price`, `edge`, bucket bounds,
  and `condition_id`. `TradingLoop` converts each `DirectionalSignal` into an
  `OrderIntent` after inventory arbitration and L2 quote checks (#6 / ADR-0011)

### core/strategy/edge.py
- CHANGE edge computation: replace `mid_price` lookup with
  `compute_quote(book, side="BUY", size_usd=settings.quote_size_usd).vwap_price`;
  skip token when `quote.complete_fill=False`  [ADR-0011]
- CHANGE `books` parameter contract: callers must pass full L2 book dict (with
  `bids`/`asks` levels), not a flat `{"mid_price": float}` dict  [ADR-0011]
- CHANGE signal semantics: `EdgeStrategy` produces **directional mispricing signals**
  only; it does NOT decide SELL YES vs BUY NO; for a bucket where `(1-mc_prob)` is
  mispriced, emit a single `DirectionalSignal(direction="bearish", ...)` with both
  YES and NO token_ids attached; execution arbitration is TradingLoop's responsibility
- REMOVE generation of explicit SELL/BUY-NO split inside `EdgeStrategy`; the strategy
  must not inspect `open_positions` to decide execution direction

### core/exchange/clob.py
- CHANGE order submission: replace `ClobClient.create_order(OrderArgs) + post_order()`
  with `SecureClient.place_limit_order(token_id=, price=, size=, side=)` returning
  `AcceptedOrder | RejectedOrder`  [ADR-0009]
- REMOVE pricing decisions from exchange adapter: `ClobExchange` receives an already
  validated limit price from TradingLoop and submits it; it does not fetch books,
  re-quote, or evaluate edge  [ADR-0011]
- CHANGE order cancellation: replace old cancel API with `SecureClient.cancel_orders()`
  [ADR-0009]
- REMOVE import of `py_clob_client` entirely  [ADR-0009]

### core/exchange/paper.py
- CHANGE `_get_fill_price()`: replace `mid_price` lookup with
  `compute_quote(book, side, size_usd).vwap_price`; reject order (return None) when
  `complete_fill=False`; BUY fills at asks VWAP, SELL fills at bids VWAP  [ADR-0011]
- REMOVE internal `_orders` and `_positions` memory dicts: `PaperExchange` no longer
  owns persistence; it only computes fill price and returns order_id; `TradingLoop`
  calls `PositionTracker.record_order(... is_paper=True)` as it does for live orders

### core/positions.py
- CHANGE `record_order()` signature: add `is_paper: bool = False` and
  `executable_price: float | None = None` parameters; writes both to DB
- CHANGE `mark_filled()`: compute and write `filled_shares = size_usd / fill_price`
  to DB `orders.filled_shares` column
- CHANGE `mark_filled()`: also write `signed_shares` and `cash_flow_usd`; BUY fills
  use positive shares and negative cash flow, SELL fills use negative shares and
  positive cash flow (#9 / #10)
- ADD `position_size_usd(token_id: str) -> float`: returns sum of `size_usd` for all
  `filled` rows with `side="BUY"` minus sum for `side="SELL"` for the given token_id;
  used for exposure reporting; do not use this as the primary SELL inventory unit
- ADD `position_shares(token_id: str) -> float`: returns net signed shares for the
  token; used by SELL sizing so exits are share-based rather than cost-basis-based
  (#9)

### core/pnl.py  [NEW]
- ADD `PnLCalculator`: pure-computation class; no side effects; reads from DB only
- ADD `realized_pnl(condition_id: str | None = None, is_paper: bool | None = None)
  -> float`: sum realized cash flows plus settlement payouts for settled net-long
  shares; must respect `signed_shares` so SELL rows are not treated as new long
  positions; `condition_id=None` means all events; `is_paper=None` means both modes
  (#10)
- ADD `unrealized_pnl(mark_prices: dict[str, float], is_paper: bool | None = None)
  -> float`: sum `mark_price * net_signed_shares + cash_flow_usd` for open net
  positions; `mark_prices` is `{token_id: mid_or_mark_price}`; mark-to-market
  semantics may use midpoint and does NOT walk L2 depth (#10)
- ADD `total_pnl(mark_prices, is_paper) -> float`: `realized_pnl() +
  unrealized_pnl(mark_prices)`
- ADD `pnl_by_event(mark_prices, is_paper) -> dict[str, float]`: per `condition_id` breakdown
  of total PnL; used by dashboard
- NOTE: `PnLCalculator` must not import from `loops/`, `model/`, or `data/`; it only
  imports from `db/` and `core/types.py`

### core/risk/pipeline.py
- CHANGE BUY admission gate: remove the order-count-based `max_open_per_token`
  check; replace with a USD-exposure check only: allow the order if
  `current_exposure + proposed_size_usd <= settings.max_total_usd`; this permits
  legitimate scale-in (second BUY on same token when exposure headroom remains) (#7)
- REMOVE `max_open_per_token` enforcement from `RiskPipeline.approve_signal()`; the
  field should no longer block any BUY (#7)

### config/settings.py
- REMOVE `max_open_per_token` field (or mark deprecated with a comment "replaced by
  max_total_usd gate in RiskPipeline"); do not read it anywhere after this change
  (#7)

### core/risk/fixed_fraction.py
- CHANGE `compute_size()`: when `OrderIntent.side == "SELL"`, return
  `settings.max_exit_fraction * positions.position_shares(intent.token_id) *
  intent.executable_price` instead of the normal BUY sizing; if net shares are zero
  or negative, return 0.0 (caller skips the order). This makes a 100% exit sell the
  current share inventory rather than the original cost basis (#9)

### config/settings.py
- ADD `max_exit_fraction: float = 1.0` (fraction of current position to exit on a
  SELL signal; 1.0 = full close; configurable for partial-exit strategies)

### core/bot.py
- FIX `McSampler` construction: pass `settings.mc_n_samp` (int) not the full
  `Settings` object as `n_samp` (#67)
- ADD `NoaaFileSource` construction: instantiate `NoaaFileSource()` and pass to
  `IemSource` as the fallback obs source; ensures NOAA tick fallback is active in
  production (#67)
- ADD TFT load guard: `Bot.__init__` must load a real `TftModel` from disk; raise
  `RuntimeError("TFT model file not found at {path}")` if the checkpoint is absent;
  `tft=None` is not a valid production state (#68)
- WIRE covariance: after constructing `McSampler`, call
  `ModelRegistry.load_covariance()` and `mc.set_cov(cov)` when a matrix is present
  (#65)
- CHANGE live mode wiring: construct `SecureClient.create(private_key=settings.
  polymarket_private_key, wallet=settings.polymarket_wallet_address or None)`;
  pass to `ClobExchange` (#1) [ADR-0009]
- CHANGE paper mode wiring: construct `PublicClient()` (no credentials); pass to
  `OrderbookFetcher` and `PaperExchange`; never load private key in paper mode
  (#1 / #8) [ADR-0009]
- WIRE `RedeemLoop`: instantiate `RedeemLoop(db, secure_client, settings)` only in
  live mode after `SecureClient` exists; add it to the asyncio task group alongside
  other loops. Paper mode does not construct `RedeemLoop`; any pre-existing paper
  task rows may be marked `skipped_paper` by maintenance scripts, not by the main
  paper bot (#73)

---

## Layer 7: backtest/ (V2)

### backtest/engine.py
- ADD inference readiness skip: before processing each historical init, call
  `data_manager.inference_readiness(station, init_time, encoder_hours=MAX_ENCODER_LENGTH)`
  or the boolean wrapper; skip rows where `ready=False` and record the skip reason
  in backtest logs/metrics. This replaces `_CUTOFF_EST_HOUR` / `MIN_HOURS` style
  time gates for model readiness (#3)
- CHANGE price used for edge and PnL: when historical L2 book data is available,
  call `compute_quote(book, side, size_usd)` from `core/pricing/quote.py`;
  when historical row has only `mid_price_yes`, store as `legacy_mid_price` (not
  `clob_price`); do not pass `legacy_mid_price` through `compute_quote`  [ADR-0011]

---

## Layer 4: loops/ (V2 continued)

### loops/trading_loop.py  [NEW - V2 addition to existing file]
- ADD bearish-signal arbitration: when a bearish `DirectionalSignal` is received, check
  `open_positions_by_token(yes_token_id)`; if open YES position exists -> generate
  SELL YES `OrderIntent`; otherwise -> generate BUY NO `OrderIntent`; both paths
  must pass `compute_quote` depth check before proceeding to `RiskPipeline`
  [ADR-0011]
- ADD re-quote before submission: after `RiskPipeline.compute_size()` resolves actual
  `size_usd`, fetch a fresh book and call `compute_quote(book, side, size_usd)`;
  skip order if `quote.complete_fill=False` or the re-quoted edge no longer clears
  `min_edge`  [ADR-0011]
- FIX per-tick risk state staleness: after each successful `place_order()` +
  `record_order()` call within a tick, immediately update the local `open_positions`
  dict and `current_exposure` float in memory (do not re-query DB); this ensures the
  second signal in the same tick sees the updated exposure and open-position count
- ADD signal write to DB after `compute_signals()`: for each signal call
  `db.record_signal(...)` with directional fields (`yes_token_id`, `no_token_id`,
  `direction`, `mc_prob`, `reference_price`, `edge`, `tick_utc`) before arbitration
  and risk; this writes a pending row to `signals` table (#12)
- ADD risk decision update: after `approve_signal()`, update the signal row with
  `final_token_id`, `final_side`, `executable_price`, `approved`, `reject_reason`,
  and `computed_size` (#12)
- ADD order linkage: after successful `place_order()`, update the signal row with
  `order_id` (#12)
- ADD tick-level log: at end of each `tick()`, emit one structured INFO log
  summarising: events checked, signals generated, approved, submitted, order_ids (#12)
- NOTE: order_submitted / order_filled / order_cancelled / exposure_snapshot /
  pnl_snapshot are written to DB (signals + orders tables) only; they do NOT emit
  AlertBus events; AlertBus is reserved for actionable human-intervention alerts
  (e.g. stale-data halt, settlement failure, discovery_warn) (#12)

### loops/forecast_loop.py
- ADD `inference_readiness` gate: at the start of each per-event tick, call
  `data_manager.inference_readiness(station, init_time, encoder_hours=MAX_ENCODER_LENGTH)`
  (inject `DataManager` into `ForecastLoop` or expose it explicitly from
  `ModelPipeline`); if `ready=False`, log one structured INFO skip message with
  `reason` and `first_eligible_init`; do not call `pipeline.run()` for this event;
  continue to next event (#3)
- CHANGE pipeline call: derive `station_config` from `event["station_id"]` and
  `target_date` from `event["event_end_date"]`; pass both to
  `ModelPipeline.run(station_config, init_utc, target_date, buckets)` while using
  `day_offset` only as the ProbabilityStore key (#22 / ADR-0010)
- ADD invalid bucket filter: before calling `mc.bucket_probs()`, filter out any
  bucket where `bucket_lo_f` or `bucket_hi_f` is None; log WARNING with
  condition_id and count of skipped buckets (#72)
- ADD empty probs guard: if `bucket_map` is empty after filtering, log WARNING
  and skip `ProbabilityStore.write()` for this event/tick entirely (#72)
- ADD top-level exception guard in `run()`: wrap the outer `tick()` call (not the
  per-event inner try) in `except Exception`; log ERROR with traceback and
  `continue` to next sleep cycle; prevents the coroutine from dying on transient
  DB or data errors (#72)

### loops/settlement_loop.py
- ADD redeem task enqueue: after marking local order payouts, if any settled order
  for the condition has winning payout (`settled_payout > 0`) and the bot is in
  live mode, upsert one `redeem_tasks(condition_id, status="pending")` row. In
  paper mode, upsert `status="skipped_paper"` or skip task creation explicitly;
  do not leave a pending redeem task that no loop will process. Do not call the SDK
  redeem method directly from `SettlementLoop` (#73)

### loops/redeem_loop.py  [NEW]
- ADD `RedeemLoop`: polls `redeem_tasks` for rows where
  `status IN ("pending", "failed")`; calls
  `SecureClient.ctf_redeem_positions_call(condition_id)` once per condition_id;
  on success writes `status="redeemed"` and `redeemed_at`; on failure writes
  `status="failed"`, increments `attempt_count`, records `last_error`, logs WARNING,
  and retries on the next poll cycle (#73)
- ADD paper guard: skip `ctf_redeem_positions_call` entirely when running in
  paper mode (`settings.trading_mode == "paper"`); this is a defensive guard only,
  because the main paper bot does not construct `RedeemLoop`. If reached, mark task
  `skipped_paper` and log INFO "paper mode: redeem skipped for condition_id={}"
  (#73)
- ADD poll interval: `settings.redeem_loop_interval_s: int = 3600` (default 1h);
  RedeemLoop sleeps between polls (#73)

### config/settings.py
- ADD `redeem_loop_interval_s: int = 3600` (#73)

### loops/sniper_loop.py
- FIX `approve_signal` call: add `open_positions` dict parameter to match V2
  `BaseRiskModel.approve_signal` signature; read `open_positions_by_token()` from
  `PositionTracker` once per event loop iteration (same pattern as TradingLoop)
- FIX `compute_size` call: add `open_positions` parameter to match V2 signature
- FIX `best_ask` read: replace `book.get("best_ask") or book.get("mid_price")`
  with `compute_quote(book, side="BUY", size_usd=size).vwap_price` from
  `core/pricing/quote.py`; SniperLoop is exempt from staleness gate (it fires
  on real observed temperature, which supersedes forecast staleness)
- FIX `record_order()` call: add `is_paper=(settings.trading_mode == "paper")`
  and `executable_price=quote.vwap_price` keyword arguments to match V2
  `PositionTracker.record_order()` signature (#11)
- REMOVE `get_active_events` fallback: always use `get_tradeable_events`; raise
  `AttributeError` loudly if `db` does not have the method
