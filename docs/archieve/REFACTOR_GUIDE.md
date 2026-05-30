# PolyWeatherBot Refactor Guide

> **Usage**: This document is the step-by-step execution handbook for the
> VS Code Copilot agent.  Work through each Phase in order.  Each Phase
> is a self-contained unit; complete all tasks within a Phase before
> moving to the next.  After every Phase run `pytest tests/ -q` and
> confirm zero regressions before continuing.
>
> **Rule**: Read every file in full before editing it.  Never patch a
> function without reading the whole module first.

---

## Conventions

| Tag | Meaning |
|-----|---------|
| `[EDIT]` | Modify an existing file |
| `[NEW]` | Create a new file |
| `[TEST]` | Write / update a pytest test |
| `[VERIFY]` | Run a shell command and confirm output |
| `->` | Depends on a prior task |

All paths are workspace-relative unless marked absolute.

---

## Phase 1 ！ Model Core Fixes  (mc.py + tft.py + pipeline.py)

These are the most foundational correctness bugs; every downstream module
depends on correct model output.

### Task 1.1  [EDIT]  `model/mc.py` ！ Replace AR(1) with empirical covariance

**Problem**: `ar1_mc_sample` builds a synthetic AR(1) covariance matrix from
a single `rho` scalar.  This ignores the real correlation structure across 48
forecast hours. The `bias_corr_c` parameter comes from a stale prior run and
should not be hard-coded. The default `n_samp=5000` is too low.

**Changes required**:
1. Remove `ar1_mc_sample`, `rho`, and `bias_corr_c` from `McSampler.__init__`
   and from `McSampler.from_settings`.
2. Add `n_samp` default = **10000**.
3. Add a new method `fit_cov(residuals: np.ndarray) -> None` that accepts an
   `(N_history, n_hours)` array of historical residuals and stores the
   empirical covariance matrix `self._cov` (shape `(n_hours, n_hours)`).
4. Replace `ar1_mc_sample` with `sample(mu_vec, cov=None)`:
   - If `cov` is None, use `self._cov`; raise `RuntimeError` if neither is set.
   - Draw `n_samp` multivariate-normal samples from `N(mu_vec, cov)`.
   - Return `(n_samp, n_hours)` array in the same units as `mu_vec`.
5. Update `sample_daily_high`:
   - Remove bias correction step entirely.
   - Call `self.sample(mu_vec)` instead of the old AR(1) call.
   - C->F conversion stays.
6. Update `bucket_probs` to return `p10`, `p50`, `p90` computed over all
   `(n_samp,)` daily-high samples together (not split per bucket).
   Return signature:
   ```python
   {
       "probs": {(lo_f, hi_f): float, ...},
       "p10_f": float,
       "p50_f": float,
       "p90_f": float,
   }
   ```
7. Remove `mc_rho` and `mc_bias_corr_c` from `config/settings.py`; add
   `mc_n_samp: int = 10000`.
8. Remove any Settings fields that no longer exist from `Settings.from_env`.

**Test** (`tests/test_model/test_mc.py`):
- `test_fit_cov_shapes`: fit on random (100, 48) residuals; confirm
  `_cov.shape == (48, 48)` and is positive semi-definite.
- `test_sample_shape`: after fit, `sample(mu_vec).shape == (10000, 48)`.
- `test_sample_daily_high_range`: output shape `(10000,)`; all values in
  reasonable Fahrenheit range (-40 to 150).
- `test_bucket_probs_keys`: result dict has keys `"probs"`, `"p10_f"`,
  `"p50_f"`, `"p90_f"`.

---

### Task 1.2  [EDIT]  `model/tft.py` ！ Fix encoder length, move data prep, fix residual inversion

**Problem**:
- `MAX_ENCODER_LENGTH = 24` but the design requires 72 hours.
- `_prep()` does feature engineering that belongs in the data layer
  (`data/storage/cleaner.py`).
- `predict()` currently returns residuals instead of temperature values.

**Changes required**:
1. Change `MAX_ENCODER_LENGTH = 72`.
2. Delete the module-level `_prep` function and `_wind_components` helper
   from this file entirely.
3. In `predict(encoder_df, decoder_df)`:
   - Remove any call to `_prep`.
   - The input DataFrames arrive already feature-engineered from
     `ModelPipeline._assemble_frames`.
   - After running inference, invert the residual: for each `fxx`, add back
     the corresponding `TMP_2m` value from `decoder_df` so the output is
     actual temperature in degC, not residual.
   - Return `{fxx: {"p10": float, "p50": float, "p90": float}}` in degC.
4. Add `local_idx` computation inside `predict` (based on `fxx` column of
   the input DataFrames, not hard-coded offsets).
5. Add `time_idx` column support: if the input DataFrames do not have a
   `time_idx` column, compute it as `fxx + 71` for encoder rows (fxx in
   -71..0) and `fxx + 71` for decoder rows (fxx 1..48), yielding global
   indices 0..119.

**Test** (`tests/test_model/test_tft.py`):
- `test_max_encoder_length`: assert `MAX_ENCODER_LENGTH == 72`.
- `test_prep_not_present`: confirm `_prep` is not defined in the module
  (`not hasattr(tft_module, "_prep")`).
- `test_predict_returns_celsius_not_residual`: mock the torch model to
  return zero residuals; confirm `predict()` output equals the `TMP_2m`
  values from decoder_df.

---

### Task 1.3  [EDIT]  `model/pipeline.py` ！ Fix window lengths, column superset, staleness

**Problems**:
- `fetch_hrrr` called without `station` argument (line ~79).
- Encoder window is 24 hours; must be 72 hours.
- `_assemble_frames` tolerance is 30 min; reduce to 20 min.
- Encoder rows must include both IEM observations and the per-hour HRRR
  f01 values (not just IEM).
- No time-cycle features (sin/cos for hour, day-of-year, month, year).
- Column superset not built before `pd.concat`; relies on automatic
  alignment that can reorder columns and introduce NaN.
- No `time_idx` field.
- No staleness tracking: 5-minute tick does not know how old the last
  pipeline result is.

**Changes required**:
1. In `run()`, pass `station` to `self.data_mgr.fetch_hrrr(init_time, station)`.
2. Change encoder window: `enc_start = init_time - pd.Timedelta(hours=71)`.
3. In `run()`, change IEM fetch to match new encoder window:
   `iem_df = self.data_mgr.fetch_iem(station, enc_start, init_time)`.
4. Add `_build_time_features(vt: pd.Timestamp) -> dict` static method that
   returns `sin_hour`, `cos_hour`, `sin_doy`, `cos_doy`, `sin_month`,
   `cos_month` from `valid_time`.  Use period lengths 24, 365.25, 12.
5. In `_assemble_frames`:
   a. Encoder covers offsets -71 to 0 (72 rows).
   b. For each encoder row at offset `k`, look up in `hrrr_hist_df` (new
      parameter: HRRR f01 values for past hours, see ′2.3) the row where
      `fxx == 1` and `init_time == vt - 1h`; merge those HRRR columns as
      the "known_reals" for that encoder step.
   c. Merge IEM row by `valid_time` with tolerance 20 min (not 30).
   d. Add time features from `_build_time_features(vt)` to every row.
   e. Add `time_idx = offset + 71` (so encoder 0-indexed: 0..71).
   f. Build a **column superset** before constructing DataFrames:
      - Collect all column names from enc_rows and dec_rows dicts.
      - Create both DataFrames with `pd.DataFrame(rows, columns=superset)`.
      - This guarantees identical columns and deterministic order.
   g. Decoder rows (fxx 1..48): `time_idx = 71 + fxx`.
6. Add `run()` return key `"computed_at": pd.Timestamp.utcnow()` so the
   caller can compute staleness.
7. Update `run()` to pass the new `bucket_probs` return shape through
   unchanged (it now has `p10_f`, `p50_f`, `p90_f`).

**Note** (flag for later): The f01-per-hour historical HRRR data path is
resolved in Phase 3 (DataManager / ParquetStore).  For now, add an optional
`hrrr_hist_df` parameter to `_assemble_frames` (default empty DataFrame) and
log a warning if it is empty.

**Test** (`tests/test_model/test_pipeline.py`):
- `test_fetch_hrrr_passes_station`: mock `data_mgr.fetch_hrrr`; assert it
  is called with both `init_time` and `station`.
- `test_encoder_length`: mock data sources; assert `encoder_df` has 72 rows.
- `test_time_idx_range`: encoder `time_idx` in 0..71, decoder in 72..119.
- `test_column_superset_no_nan_from_alignment`: ensure no column present in
  one frame but absent in the other.
- `test_staleness_key_present`: `result["computed_at"]` is a `pd.Timestamp`.

---

### Task 1.4  [EDIT]  `model/store/registry.py` ！ Remove private DB access

**Problem**: `_db_latest_row()` accesses `self.db._conn` directly, bypassing
the `DbClient` public interface.

**Change**: Replace `self.db._conn` with a proper public method call.  Add
`get_model_versions() -> list[dict]` to `db/schema.py` `DbClient` if it does
not already exist.  Update `_db_latest_row` to call `self.db.get_model_versions()`.

**Test** (`tests/test_model/test_registry.py` or existing):
- Ensure `ModelRegistry` methods work without accessing `_conn`.

---

## Phase 2 ！ Data Observation Fixes  (obs/)

### Task 2.1  [EDIT]  `data/obs/iem.py` ！ Time-point fetch, dual-source, logging

**Problems**:
- Downloads a time-range for each call; does not cache prior-hour rows for
  the model's historical encoder window.
- Second source activates only when all data is NaN; should activate earlier
  to fill gaps.
- `extract_temp_c` does not warn when high-precision T-group is absent.

**Changes required**:
1. Add a parameter `allow_partial: bool = True` to `fetch()`.  When True,
   return partial data with warnings instead of failing silently.
2. Change fallback logic: if the primary source returns **any** rows with
   NaN in critical columns (`raw_tmpc_tenths`, `valid_time`), log a warning
   and attempt a partial merge with the secondary source for those missing
   rows specifically.
3. In `_parse_csv` / the row extraction path: after parsing each row, if
   T-group (`_RE_T_GROUP`) is not found, emit `logger.warning` noting the
   station and time.
4. Document clearly that `fetch()` is for time-range downloads; callers
   needing the historical encoder window should call multiple fetches and
   cache in `ParquetStore` (handled in Phase 3).
5. Add a `last_observation_time` property that returns the timestamp of the
   most recently fetched row, or None.

**Test** (`tests/test_data/test_iem.py`):
- `test_partial_fallback_triggers_warning`: mock primary to return rows with
  NaN temp; assert secondary is called and `logger.warning` fires.
- `test_t_group_missing_logs_warning`: feed CSV with no T-group; assert
  warning is logged.
- `test_last_observation_time`: after fetch, property returns a Timestamp.

---

### Task 2.2  [EDIT]  `data/obs/noaa_file.py` ！ Robustness, parse errors, logging

**Problems**:
- Returns at most one row (current observation only); no historical data.
- Parser is fragile with multiple numeric segments in METAR.
- Insufficient alerting on parse failure.

**Changes required**:
1. Wrap each regex match in `_parse_txt` with a try/except; log a warning
   per field that fails to parse, then continue.
2. When `_parse_txt` returns `None`, log at `logger.error` (not just silent
   empty return).
3. Add a docstring note: "This source provides at most one row (current
   observation).  It cannot fill historical gaps."
4. Log `logger.warning` when the observation timestamp is outside the
   requested `[start, end]` window (currently silently returns empty).

**Test** (`tests/test_data/test_noaa_file.py`):
- `test_parse_error_logs_warning`: feed malformed METAR string; assert
  `logger.warning` is captured.
- `test_out_of_window_logs_warning`: observation time outside window; assert
  warning is logged.

---

## Phase 3 ！ Data Forecast Fixes  (_grib_source.py, cleaner.py, parquet_store.py, manager.py)

### Task 3.1  [EDIT]  `data/forecast/_grib_source.py` ！ Multi-station, 48-hour fill, error handling

**Problems**:
- `_klga_idx` is named for a single station; multi-station support is absent.
- Only 4 HRRR cycles per day have 48-hour range; others max at 18h.
- `_fetch_fxx` and `_fetch_range` silently return NaN / None on failure.
- No source-switching on download failure.
- No handling for missing init_time cycles (holes in the time series).

**Changes required**:
1. Replace `self._klga_idx` with `self._grid_idx_cache: dict[str, int]`
   keyed by `station_id`.  Update `_find_grid_index` to look up / populate
   this dict.
2. Move the HRRR 48-hour fill logic into a new standalone class
   `HrrrGapFiller` in `data/forecast/_gap_filler.py`:
   - `fill(init_time, station, store, short_source, long_source) -> pd.DataFrame`
   - For init hours not in `{0, 6, 12, 18}`, the method fetches fxx=1..18
     from the current init and then the nearest MAJOR-hour cycle for
     fxx=19..48, splicing them together.
   - Saves the result to `store` with a `is_filled: bool` column.
3. In `_fetch_fxx`:
   - On download failure (None from `_fetch_idx` or `_fetch_range`), emit
     `logger.error` with station, init_time, fxx details.
   - Do **not** silently return a row of NaN; raise a descriptive
     `DataFetchError` (custom exception in `data/__init__.py`).
4. In `fetch()`, catch `DataFetchError` per-fxx and record the failed fxx
   list; at the end, if any fxx failed, log a consolidated warning.
5. Add completeness check: after assembling all fxx rows, verify the
   expected fxx values are all present; log which are missing.

**Test** (`tests/test_data/test_hrrr_s3.py` or new file):
- `test_grid_idx_cached_per_station`: two different station configs yield
  two separate cache entries.
- `test_gap_filler_splices_correctly`: mock short (fxx 1..18) and long
  (fxx 1..48) sources; verify result has 48 rows.
- `test_fetch_fxx_raises_on_failure`: mock `_fetch_idx` to return None;
  assert `DataFetchError` is raised.

---

### Task 3.2  [EDIT]  `data/storage/cleaner.py` ！ Add feature engineering, absorb _prep logic

**Problem**: `Cleaner` is isolated from the data flow and does not include
the feature-engineering steps currently scattered in `model/tft.py::_prep`.

**Changes required**:
1. Add `feature_engineer_hrrr(df: pd.DataFrame) -> pd.DataFrame` static
   method that applies (in order):
   a. Wind U/V decomposition for 10m, 80m, 500mb -> `WS_*`, `WD_sin_*`,
      `WD_cos_*` columns.
   b. `log1p_GUST_sfc = log1p(GUST_sfc)` (clip negatives to 0 first).
   c. `log1p_PRATE = log1p(PRATE)` (same guard).
   d. Cast all numeric output columns to `float32`.
2. Add `compute_temp_error(hrrr_df, obs_df) -> pd.Series` static method:
   - Merge on `valid_time` (tolerance 20 min).
   - Return `raw_tmpc_tenths - TMP_2m` (degC residual = target column).
3. Add `add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame`:
   - Appends `sin_hour`, `cos_hour`, `sin_doy`, `cos_doy`, `sin_month`,
     `cos_month` derived from `df[time_col]`.
4. Ensure `Cleaner` is called from `DataManager` after every raw fetch
   (see Task 3.4).

**Test** (`tests/test_data/test_cleaner.py`):
- `test_feature_engineer_wind_columns`: input with U/V columns; verify WS,
  sin, cos columns added and values are float32.
- `test_compute_temp_error_units`: HRRR TMP_2m=20C, IEM=21C -> error=1.
- `test_add_time_features_keys`: output has all six sin/cos columns.

---

### Task 3.3  [EDIT]  `data/storage/parquet_store.py` ！ HRRR stores only f01

**Problem**: HRRR store currently saves all fxx.  The design requires saving
only `fxx=1` (the per-hour f01 used as historical encoder input) for the
observation/encoder path.  The decoder HRRR data (fxx 1..48) is fetched
fresh each cycle and not persisted separately.

**Changes required**:
1. Add `append_hrrr_f01(station_id, df) -> int` method that filters
   `df[df["fxx"] == 1]` before calling `_append_to_quarter`.
2. Keep the existing `append` method for obs data unchanged.
3. Update docstring to clarify the two storage policies.

**Test** (`tests/test_data/test_parquet_store.py`):
- `test_append_hrrr_f01_only_stores_fxx1`: input df has fxx=[0,1,2]; after
  `append_hrrr_f01`, reading back yields only fxx=1 rows.

---

### Task 3.4  [EDIT]  `data/manager.py` ！ Wire cleaner, temperature units, cold-start guard

**Problems**:
- `Cleaner` is not called after fetching raw data.
- Kelvin->Celsius and F->C conversions are duplicated in multiple places.
- No cold-start guard: early rows have `temp_error 「 IEM_obs` (nonsensical).
- `fetch_hrrr` saves all fxx; should call `append_hrrr_f01`.
- Paper mode has no data sources wired.

**Changes required**:
1. After each successful `src.fetch()` in `fetch_hrrr`, call
   `Cleaner.clean_hrrr(df)` and `Cleaner.feature_engineer_hrrr(df)` before
   storing and returning.
2. After each successful `src.fetch()` in `fetch_iem`, call
   `Cleaner.clean_iem(df)`.
3. Replace `self._forecast_store.append(...)` with
   `self._forecast_store.append_hrrr_f01(...)` for HRRR data.
4. Add a `cold_start_hours: int = 72` constructor parameter.  Expose a
   `drop_cold_start(df: pd.DataFrame, time_col: str) -> pd.DataFrame`
   method that drops rows where `time_col < min(df[time_col]) + cold_start_hours`.
5. Add a `fetch_hrrr_encoder_history(station, init_time) -> pd.DataFrame`
   method that reads the f01 ParquetStore for the window
   `[init_time-71h, init_time]` and returns it for the pipeline encoder.
6. **Paper mode wiring**: add a `configure_paper_mode(settings) -> None`
   method that instantiates and attaches all data sources with their default
   parameters, so `BotManager` can call it during paper-mode startup.

**Test** (`tests/test_data/test_manager.py`):
- `test_cleaner_called_after_fetch_hrrr`: mock `Cleaner.clean_hrrr`; assert
  it is called.
- `test_cold_start_drop`: DataFrame with 100 hourly rows; `cold_start_hours=72`
  should drop first 72 rows.
- `test_fetch_hrrr_encoder_history_reads_f01_store`: mock parquet store;
  assert correct time window is requested.

---

## Phase 4 ！ Event & Loop Fixes

### Task 4.1  [EDIT]  `events/discovery.py` ！ Robust slug parsing, dedup

**Problems**:
- `parse_slug` does not handle `or-below` patterns (returns NaN).
- No deduplication check against the database before inserting.

**Changes required**:
1. Add pattern to `parse_slug`:
   - `Xf-or-below` / `Xforbelow` / `be-Xf-or-below` -> `(-inf, X)`.
   - Add test cases to cover the existing gap.
2. In `EventDiscovery.discover()`, before upserting any market, call
   `self.db.market_exists(token_id)` (add to `DbClient` if absent).  Skip
   insert for existing records; log at DEBUG level.

**Test** (`tests/test_events/test_discovery.py`):
- `test_parse_slug_or_below`: `"60f-or-below"` -> `(-inf, 60.0)`.
- `test_parse_slug_be_or_below`: `"be-55f-or-below"` -> `(-inf, 55.0)`.
- `test_dedup_skips_existing`: mock `db.market_exists` to return True; assert
  DB insert is not called.

---

### Task 4.2  [EDIT]  `loops/discovery_loop.py` ！ Change interval to 1 hour

**Problem**: Default `interval_seconds = 86_400` (24h); should be 3600 (1h).

**Change**: Set `interval_seconds: float = 3_600.0`.

**Test** (`tests/test_loops/` ！ no separate test needed; verify constant in
existing `test_discovery.py`):
- `test_discovery_loop_default_interval`: instantiate `DiscoveryLoop` with
  no args; assert `_interval_seconds == 3600`.

---

### Task 4.3  [EDIT]  `events/lifecycle.py` ！ Review cutoff, settled semantics

**Problems**:
- `_CUTOFF_HOUR_UTC = 21` (16:00 EST) is hard-coded and described as
  "arbitrary".
- `settled` state is claimed to be set externally but there is no clear
  mechanism.

**Changes required**:
1. Make `_CUTOFF_HOUR_UTC` a `Settings` field `event_cutoff_hour_utc: int = 21`
   with an env-var override.  Pass it into `EventLifecycle.__init__` from
   `Settings`.
2. Add a docstring clarification: `settled` is set by `Settler` after
   successful on-chain settlement or manual verification.  `advance()` never
   sets `settled`.
3. No behavior change required for this task.

**Test** (`tests/test_events/test_lifecycle.py`):
- `test_cutoff_hour_configurable`: construct with `cutoff_hour_utc=20`; assert
  cutoff fires at 20:00 UTC not 21:00.

---

### Task 4.4  [EDIT]  `loops/forecast_loop.py` ！ YES+NO signals, run not tick, staleness

**Problems**:
- Only YES-outcome markets are included in `buckets` and `books`.
- The method is named `tick` while all other loops use `run`.
- 5-minute tick does not check staleness of the last pipeline result.

**Changes required**:
1. In the market-fetching section, collect **both** YES and NO token_ids and
   pass both to the orderbook fetch.
2. Update signal generation: `strategy.compute_signals` already receives all
   markets; ensure it can return signals for both YES and NO sides.
3. Rename `tick` -> `run` (keep `tick` as a deprecated alias calling `run`
   with a deprecation warning).
4. Add `max_staleness_minutes: int = 65` constructor parameter.  At the top
   of `run()`, check `self._last_result_at`; if the last pipeline result is
   older than `max_staleness_minutes`, log a warning and increase the
   strategy's `min_edge` threshold by a configurable `staleness_edge_bump`
   (default 0.02) to reduce risk when data is stale.
5. Store `self._last_result_at: pd.Timestamp | None = None` and update it
   after each successful `pipeline.run()`.

**Test** (`tests/test_loops/test_forecast_loop.py`):
- `test_no_token_included_in_books`: mock DB to return a NO-outcome market;
  assert `orderbook.get_book` is called for the NO token.
- `test_run_alias_for_tick`: calling `run()` and `tick()` both delegate to
  the same logic.
- `test_staleness_bumps_edge`: set `_last_result_at` to 90 minutes ago;
  assert `min_edge` is increased.

---

## Phase 5 ！ Risk & Strategy Fixes

### Task 5.1  [EDIT]  `core/risk/fixed_fraction.py` ！ Per-token duplicate guard

**Problem**: No per-token/per-market repeat-order protection; only total
exposure is checked.

**Changes required**:
1. Add `max_open_per_token: int = 1` constructor parameter.
2. `approve_signal` receives an additional optional `open_positions: list[dict]`
   parameter (default `[]`).  Count open positions with matching `token_id`;
   reject if count >= `max_open_per_token`.
3. Update all call sites in `loops/forecast_loop.py` to pass
   `self.positions.get_open_positions()` (add this method to
   `PositionTracker` if absent).

**Test** (`tests/test_core/test_risk.py`):
- `test_duplicate_token_rejected`: one position already open for token A;
  second signal for token A is rejected.
- `test_different_token_approved`: existing position for token A; new signal
  for token B is approved.

---

### Task 5.2  [EDIT]  `core/exchange/clob.py` ！ Add authentication check

**Problem**: No credential validation at construction time.

**Change**: In `__init__`, call `self._client.get_ok()` or equivalent health
check; if it raises, raise `RuntimeError("CLOB authentication failed: ...")`.
Wrap in try/except and include the original exception message.

**Test** (`tests/test_core/test_exchange.py`):
- `test_clob_auth_failure_raises`: mock `clob_client.get_ok` to raise; assert
  `RuntimeError` is raised at construction.

---

### Task 5.3  [EDIT]  `core/exchange/paper.py` ！ Use real CLOB book for fills

**Problem**: When `book_fetcher` is None, fills happen at submitted price;
this misrepresents real-world conditions.

**Change**: Make `book_fetcher` a required argument (remove default `None`).
Raise `ValueError` if `book_fetcher` is None and log a clear message.

**Test** (`tests/test_core/test_exchange.py`):
- `test_paper_requires_book_fetcher`: constructing `PaperExchange(None)`
  raises `ValueError`.

---

## Phase 6 ！ Backtest Cold-Start Fix

### Task 6.1  [EDIT]  `backtest/engine.py` ！ Cold-start skip

**Problem**: No mechanism to skip the initial period where the model has
insufficient encoder history.

**Changes required**:
1. Add `cold_start_hours: int = 72` constructor parameter.
2. In `run()`, for each `init_time`, skip it if
   `init_time < pd.Timestamp(start_date, tz="UTC") + pd.Timedelta(hours=cold_start_hours)`.
3. Log a one-time INFO message reporting how many init_times were skipped.

**Test** (`tests/test_backtest/test_engine.py`):
- `test_cold_start_skips_first_72h`: run with start_date, verify no result
  rows with init_utc within the first 72 hours.

---

## Phase 7 ！ Settings Cleanup

### Task 7.1  [EDIT]  `config/settings.py` ！ Remove stale fields, add new ones

Consolidate all changes from prior phases:

| Remove | Add |
|--------|-----|
| `mc_rho` | `mc_n_samp: int = 10000` |
| `mc_bias_corr_c` | `event_cutoff_hour_utc: int = 21` |
|  | `cold_start_hours: int = 72` |
|  | `max_staleness_minutes: int = 65` |
|  | `staleness_edge_bump: float = 0.02` |
|  | `max_open_per_token: int = 1` |

Read `config/settings.py` fully before editing; preserve all other fields.
Update `Settings.from_env()` accordingly.

**Test** (`tests/test_config/test_settings.py`):
- `test_removed_fields_absent`: assert `Settings` does not have `mc_rho`
  or `mc_bias_corr_c` attributes.
- `test_new_fields_present`: assert all new fields exist with correct defaults.

---

## Phase 8 ！ Integration Regression Tests

After all prior phases are complete:

### Task 8.1  [TEST]  End-to-end paper-mode smoke test

File: `tests/test_integration/test_paper_smoke.py`

1. Instantiate `Settings` with paper-mode defaults.
2. Wire up `DataManager.configure_paper_mode(settings)`.
3. Instantiate `ModelPipeline` with mocked `TftModel` and `McSampler`.
4. Run `ForecastLoop.run()` with a mocked DB returning one active event.
5. Assert at least one order is placed (or zero orders with a log if no edge).

### Task 8.2  [VERIFY]  Full pytest suite green

```bash
cd /home/Code/PolyWeatherBot
source .venv/bin/activate
pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: all tests pass; zero errors.

---

## Appendix: Cross-cutting Rules

1. All new Python files must contain only ASCII characters.
2. After every edit, run `pytest tests/ -q` and fix failures before moving on.
3. Do not rename symbols, reorder functions, or reformat files outside the
   scope of each task.
4. Every new public method must have at least one call site or test; no dead
   stubs.
5. If a task reveals a second independent bug, document it in this file under
   a new Task entry rather than fixing it silently.
6. `config/` must not import from any other project package.
7. `data/` sources must not import from `core/`, `model/`, or `loops/`.
