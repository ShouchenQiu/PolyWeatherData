# V2 Problem Audit

This file is the intake list for the next planning round.  Keep entries short:
problem first, agreed direction second, candidate edits last.

Do not generate v2 slices, implementation workflow, or ADRs until this audit
stabilises.  Do not rewrite completed v1 plan files in place.

## Format rules

```
### PNN -- Short problem title
- PROBLEM: what is inconsistent, wrong, missing, or ambiguous
- DIRECTION: confirmed direction from grill Q&A
- CANDIDATE: likely files/behaviours to change later (Layer N: path/)
- OPEN-BLOCK: question that must be closed before this problem can be sliced
- OPEN-DEFER: question that can be resolved inside the implementing slice
- SPAWNS: PNN problems that emerged from investigating this one
```

Canonical labels:
| Label | Meaning |
|-------|--------|
| PROBLEM | Confirmed issue or ambiguity |
| DIRECTION | Agreed design direction |
| CANDIDATE | Possible future implementation/doc edit; include layer tag in parens |
| OPEN-BLOCK | Must be resolved before this problem can be sliced |
| OPEN-DEFER | Can be decided by the implementing slice; does not block slicing |
| SPAWNS | Cross-reference to problems spawned by investigating this one |

---

## Time Semantics

### P01 -- Encoder window and cold-start semantics are mixed
- PROBLEM: `DataManager.build_encoder_obs_window()` documents fixed
  `[init_time - n_hours, init_time - 1h]`, but implementation reads
  `n_hours + cold_start_hours` and drops the first `cold_start_hours` rows.
- PROBLEM: `CONTEXT.md` says encoder window is `-71..0`, while the agreed live
  model input is `-72..-1`.
- PROBLEM: `CONTEXT.md` says cold-start rows are dropped by `Cleaner`, but agreed
  semantics treat cold-start as data-readiness/skip behavior.
- DIRECTION: encoder window is exactly 72 rows covering `-72..-1` relative to
  `init_time`; do not include the `init_time` hour.
- DIRECTION: cold-start means there is not enough prior data to construct the
  fixed encoder window; it should not widen every live encoder read.
- DIRECTION: live runs may let incomplete encoder-window construction fail the
  current ForecastLoop tick; backtests keep the explicit `cold_start_hours`
  init-time skip.
- CANDIDATE: change `data/manager.py` to read only
  `[init_time - n_hours, init_time - 1h]` and remove its `Settings` dependency.
  (Layer 2: data/)
- CANDIDATE: remove `Cleaner.drop_cold_start()` if no production caller remains.
  (Layer 2: data/)
- CANDIDATE: update `tests/test_data/test_manager_window.py` to assert exact read
  bounds and fixed 72-row behavior. (tests/)
- CANDIDATE: update `CONTEXT.md` terms for `encoder window` and `cold-start`.
  (docs/)
- OPEN-BLOCK: check whether `ForecastLoop`, `ModelPipeline`, `BacktestEngine`, and
  tests contain any other time-boundary mismatches; findings may spawn new problems.
- OPEN-DEFER: decide whether ForecastLoop should emit a data-readiness alert when
  encoder-window construction fails, or whether logging is enough.
- SPAWNS: TBD -- pending resolution of OPEN-BLOCK above.

### P02 -- Generic timestamp names hide different business times
- PROBLEM: `signals.timestamp` means the time a trading signal was generated or
  recorded, but the generic column name does not say which lifecycle event it
  represents.
- PROBLEM: `BacktestEngine` expects historical CLOB rows with `timestamp_utc`,
  which means the CLOB snapshot sampling time, not a signal time.
- PROBLEM: UI code aliases `signals.timestamp AS signal_time`, which confirms
  readers already need a more specific term at query time.
- DIRECTION: avoid bare `timestamp` for domain data; name time fields after the
  event they record.
- DIRECTION: use `signal_generated_at` for the time a trading signal is produced.
- DIRECTION: use `clob_snapshot_at_utc` for the sampling time of a historical CLOB
  price snapshot.
- DIRECTION: treat `signals.timestamp` as a compatibility surface first; prefer
  query/API aliases before planning a SQLite migration.
- DIRECTION: allow `BacktestEngine` to move toward `clob_snapshot_at_utc` sooner,
  while temporarily accepting legacy `timestamp_utc` input.
- CANDIDATE: rename or alias `signals.timestamp` to `signal_generated_at`.
  (Layer 1b: db/)
- CANDIDATE: rename or alias backtest CLOB input `timestamp_utc` to
  `clob_snapshot_at_utc`. (backtest/)
- CANDIDATE: update `CONTEXT.md` with the chosen terms once canonical names are
  confirmed. (docs/)
- OPEN-DEFER: decide whether a real SQLite migration is worth doing once other
  v2 schema issues are known.

### P03 -- Sniper sources use naive datetimes inconsistent with the rest of the codebase
- PROBLEM: `snipers/noaa.py` uses `datetime.now()` (naive local time) for all sniper
  window boundary comparisons; the windows (xx:50:45--xx:56:00) are UTC-semantic, so
  this is only accidentally correct when the host OS timezone is UTC.
- PROBLEM: Both `snipers/noaa.py` and `snipers/awc.py` pass `datetime.utcnow()` (naive
  UTC, deprecated since Python 3.12) as the `observed_at` argument to `on_observation`;
  every other module uses `datetime.now(timezone.utc)` exclusively.
- PROBLEM: `snipers/awc.py` also uses `datetime.now()` (naive local) inside
  `_in_sniper_window()` for the same UTC-semantic window check.
- DIRECTION: replace every `datetime.now()` and `datetime.utcnow()` call in both
  sniper sources with `datetime.now(timezone.utc)`; window helpers must accept and
  return UTC-aware datetimes.
- DIRECTION: `SniperManager` queue items already carry the timestamp opaquely;
  no change to its interface is needed beyond the source fix.
- CANDIDATE: replace naive datetime calls in `snipers/noaa.py`. (snipers/)
- CANDIDATE: replace naive datetime calls in `snipers/awc.py`. (snipers/)
- OPEN-DEFER: `SniperLoop` currently uses the observation timestamp only for
  `logger.debug`; confirm no comparison against aware datetimes is added before
  this slice lands.

---

## Model / Store Layout

### P04 -- quantiles_by_fxx and last_tick_utc are duplicated across every BucketPayload
- PROBLEM: `BucketPayload.quantiles_by_fxx` is the TFT decoder-hour quantile output
  (48 hours ¡Á p10/p50/p90); it is identical for every bucket in the same
  (condition_id, day_offset) write ¡ª ForecastLoop passes the same `dict(quantiles)`
  to every bucket in the comprehension.
- PROBLEM: `BucketPayload.last_tick_utc` is also identical across all buckets in one
  write call (enforced by `_validate_last_tick_utc_consistent`), yet it is stored
  redundantly in every payload and then re-extracted via a fragile
  `next(iter(buckets.values())).last_tick_utc` hack in `read()`.
- PROBLEM: `TradingLoop` tests construct `BucketPayload(mc_prob=..., quantiles_by_fxx={}, ...)`
  with empty dicts, confirming that production callers never read `quantiles_by_fxx`
  from the bucket level.
- DIRECTION: `BucketPayload` becomes a single-field value object: `mc_prob: float` only.
- DIRECTION: `quantiles_by_fxx` and `last_tick_utc` both move to `ProbabilitySnapshot`
  (key level); `write()` receives them as explicit top-level parameters alongside
  `buckets: dict[_BucketKey, float]`.
- DIRECTION: `_validate_last_tick_utc_consistent` is deleted; `write()` owns
  `last_tick_utc` directly so no cross-bucket consistency check is needed.
- CANDIDATE: refactor `core/probability_store.py`: simplify `BucketPayload`,
  update `ProbabilitySnapshot`, update `write()` signature, remove
  `_validate_last_tick_utc_consistent`, fix `read()`. (core/)
- CANDIDATE: update `loops/forecast_loop.py` call site to pass `last_tick_utc` and
  `quantiles_by_fxx` as `write()` arguments, not inside each bucket. (loops/)
- CANDIDATE: update `tests/test_core/test_probability_store.py` and
  `tests/test_loops/test_forecast_loop.py` and `tests/test_loops/test_trading_loop.py`
  to remove `quantiles_by_fxx` from `BucketPayload` construction. (tests/)
