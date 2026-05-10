18 slices total: 4 foundation, 1 HITL, 13 vertical.

---

### FOUNDATION SLICES

---

**F01 — Settings Cleanup + DB Orders Table**

- Type: AFK
- Blocked by: none
- Source refs:
    - [settings.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — all 5 REMOVE, 1 CHANGE (mc_n_samp), all 13 ADD
    - [schema.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD orders table
- Out of scope: cutoff removal from loop code (that is V09); PositionTracker methods (F03)

---

**F02 — AlertBus**

- Type: AFK
- Blocked by: none
- Source refs:
    - `core/alerts.py [NEW]` — AlertEvent dataclass + AlertBus (subscribe/emit/exception-per-task)
- Out of scope: subscribers (UI in V13, TradingLoop emitters in V11); Monitor bodies (Deferred)

---

**F03 — PositionTracker Order Lifecycle**

- Type: AFK
- Blocked by: F01 (orders table must exist)
- Source refs:
    - [positions.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — all 8 ADDs (record_order, mark_filled, mark_closed, mark_cancelled, mark_settled, open_positions_by_token, total_exposure, exposure_by_condition)
- Out of scope: anything that calls mark_settled (that is V12); RiskPipeline integration (V06)

---

**F04 — ProbabilityStore**

- Type: AFK
- Blocked by: none
- Source refs:
    - `core/probability_store.py [NEW]` — full implementation, asyncio Lock, key/value/payload shape, day_offset domain [ADR-0004]
- Out of scope: ForecastLoop writes (V10); TradingLoop reads (V11)

---

### HITL SLICE

---

**H01 — Station Grid Lookup (6 new stations)**

- Type: HITL
- Blocked by: none (human action; can start any time)
- Source refs:
    - [stations.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD Toronto, Atlanta, Chicago, Seattle, Dallas, Miami with HRRR row/col
- Out of scope: wiring the new stations into _grib_source (V02 picks them up automatically from STATIONS dict once H01 lands)
- HITL reason: row/col values require running a HRRR grid-lookup script against actual HRRR data; not derivable from code alone

---

### VERTICAL SLICES

---

**V01 — Sniper Manager Rename**

- Type: AFK
- Blocked by: none
- Source refs:
    - `snipers/sniper_manager.py [RENAME]` — MOVE + all import sites
- Out of scope: nothing else

---

**V02 — HRRR Multi-Station Pipeline + HrrrFill + Parquet Enforcement**

- Type: AFK
- Blocked by: F01 (settings: `hrrr_source_priority`, `cold_start_hours`)
- Source refs:
    - `data/forecast/hrrr_fill.py [NEW]` — HrrrFill + tail-gap fallback [ADR-0005]
    - [_grib_source.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — all 7 items (REMOVE klga_idx, CHANGE fetch sig, ADD batch multi-station, WIRE HrrrFill, ADD fallback, ADD error handling, ADD gap-detection)
    - [hrrr_zarr.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE to batch-only
    - [parquet_store.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ENFORCE f01-only [ADR-0003]
- Out of scope: Cleaner feature engineering (V03); new stations (H01 feeds in automatically)

---

**V03 — Cleaner + DataManager + build_encoder_obs_window**

- Type: AFK
- Blocked by: F01 (settings: `cold_start_hours`), V02 (Cleaner reads parquet that _grib_source writes)
- Source refs:
    - [cleaner.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — WIRE, OWN feature engineering (K->C, F->C, wind decomp, log1p, temp_error), cold-start drop, SAVE to parquet
    - [manager.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — REMOVE dup transforms, ADD station param threading, ADD `build_encoder_obs_window`
- Out of scope: model pipeline consuming the window (V05); IEM obs hardening (V04, independent path)

---

**V04 — IEM + NOAA Obs Hardening**

- Type: AFK
- Blocked by: F01 (settings)
- Source refs:
    - [iem.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE :51-only fetch, FIX fallback, ADD fine-grained warnings
    - [noaa_file.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD parse robustness, ADD fine-grained warnings
- Out of scope: how IEM data reaches the Cleaner (V03); settlement IEM read (V12 uses parquet directly)

---

**V05 — Model Pipeline 72h + MC Overhaul + Registry + Backtest Update**

- Type: AFK
- Blocked by: F01, F04 (ProbabilityStore — pipeline writes staleness timestamp to it), V03 (build_encoder_obs_window), V04 (IEM obs in parquet)
- Source refs:
    - [mc.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — REMOVE AR(1)/bias_corr_c [ADR-0002], CHANGE n_samp, ADD fit_cov, FIX bucket_probs, ADD max_bias apply
    - [tft.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE encoder 24->72 [ADR-0001], MOVE `_prep` (source moved to V03/cleaner), FIX un-residualise
    - [pipeline.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — all 6 FIX + 4 ADD (time_idx, seasonal features, superset alignment, staleness timestamp)
    - [registry.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — FIX `_db_latest_row`, ADD store/load/calibrate_max_bias
    - [engine.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD cold-start guard, CHANGE pipeline.run() call site
- Out of scope: ForecastLoop scheduling (V10); TradingLoop consuming output (V11); Monitor health checks (Deferred)

---

**V06 — Strategy + Risk Stack**

- Type: AFK
- Blocked by: F01 (settings: `max_open_per_token`), F03 (PositionTracker: `open_positions_by_token`)
- Source refs:
    - [base.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE `compute_signals` sig (add `open_positions`), ADD `name` abstract property
    - [edge.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — FIX NO token edge, ADD `name`
    - `core/strategy/multi.py [NEW]` — MultiStrategy (MVP impl, dedup by token_id, ValueError on empty)
    - [base.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE `approve_signal` + `compute_size` sigs (add `open_positions`)
    - [fixed_fraction.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD duplicate-position prevention via `open_positions` param
    - `core/risk/kelly.py [NEW]` — KellyCriterion stub (NotImplementedError)
    - `core/risk/pipeline.py [NEW]` — RiskPipeline (AND approve, min-positive size, ValueError on empty)
- Out of scope: TradingLoop wiring of MultiStrategy and RiskPipeline (V11); AlertBus integration (V11)

---

**V07 — CLOB Auth Split + Paper Exchange + bot.py**

- Type: AFK
- Blocked by: F01 (settings), F02 (AlertBus — order failure alert)
- Source refs:
    - [clob.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD order_failure alert via AlertBus
    - [paper.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE to real CLOB book via OrderbookFetcher, ADD OrderbookFetcher (L1-only), ADD order_failure alert
    - [bot.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — FIX paper mode data sources, CHANGE live/paper auth wire-up [ADR-0008]
- Out of scope: TradingLoop calling these exchanges (V11)

---

**V08 — Slug Parser Registry**

- Type: AFK
- Blocked by: none
- Source refs:
    - [discovery.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE parse_slug to module-level registry, ADD entries for all known patterns (or-above, or-higher, or-below, be-{N}f-or-below, variants)
- Out of scope: DiscoveryLoop scheduling (V09); new slug formats beyond what production has observed (future extension)

---

**V09 — Discovery Loop Overhaul**

- Type: AFK
- Blocked by: F01 (cutoff settings are REMOVED; ensures no dangling reference)
- Source refs:
    - [discovery_loop.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — CHANGE hourly, ADD dedup (skip existing DB entries), REMOVE cutoff-hour logic
- Out of scope: slug parsing (V08, independent); EventLifecycle cutoff removal (V12)

---

**V10 — ForecastLoop Alignment + ProbabilityStore Writes**

- Type: AFK
- Blocked by: F01, F04 (ProbabilityStore), V05 (model pipeline writes probabilities)
- Source refs:
    - [forecast_loop.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — SPLIT (TradingLoop moves to V11), CHANGE wall-clock UTC alignment, ADD write to ProbabilityStore per active Event keyed by (condition_id, day_offset in {0,1,2}) [ADR-0004]
- Out of scope: TradingLoop that reads ProbabilityStore (V11); ForecastLoop scheduling of model (already in V05)

---

**V11 — TradingLoop + Staleness Gate**

- Type: AFK
- Blocked by: F01, F02 (AlertBus), F03 (PositionTracker), F04 (ProbabilityStore), V06 (Strategy+Risk), V07 (CLOB/Paper exchanges), V10 (ForecastLoop populates ProbabilityStore)
- Source refs:
    - `loops/trading_loop.py [NEW]` — all 7 items: TradingLoop, cadence, YES/NO tokens, staleness gate [ADR-0006], data_stale alert, risk_rejected alert, ENFORCE MultiStrategy [ADR-0004]
- Out of scope: UI staleness display (V13); SettlementLoop (V12); Monitor bodies (Deferred)

---

**V12 — Event Lifecycle + Settlement Loop**

- Type: AFK
- Blocked by: F03 (PositionTracker.mark_settled), V03+V04 (IEM obs in parquet for temp confirmation)
- Source refs:
    - [lifecycle.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — REMOVE `_CUTOFF_HOUR_UTC` + gating, CHANGE `advance()` (midnight-only), ADD `set_settled()` [ADR-0007]
    - `loops/settlement_loop.py [NEW]` — ADD `confirm_settlement`, ADD mark_settled for all filled/closed orders, ADD `set_settled` call, ADD SettlementRelayer stub in [settlement.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) [ADR-0007]
- Out of scope: on-chain SettlementRelayer implementation (Deferred per ADR-0007); full Telegram payout notification (Deferred)

---

**V13 — UI AlertBus Integration**

- Type: AFK
- Blocked by: F02 (AlertBus), V11 (TradingLoop emits data_stale)
- Source refs:
    - [app.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) — ADD staleness alert display (subscribe to AlertBus for "data_stale"), ADD Telegram stub subscriber (no-op DEBUG log)
- Out of scope: full Telegram routing, notification dedup, recovery alerts, dashboard (all Deferred)

---

## Coverage Matrix (Summary)

|File / heading|Source items|Slice|Status|
|---|---|---|---|
|[settings.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|5 REMOVE + 1 CHANGE + 13 ADD = 19|F01|Covered|
|[stations.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ADD (6 stations, row/col TBD)|H01|Blocked/HITL|
|[schema.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ADD (orders table)|F01|Covered|
|`data/forecast/hrrr_fill.py`|2 ADD|V02|Covered|
|[hrrr_zarr.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE|V02|Covered|
|[_grib_source.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 REMOVE + 1 CHANGE + 3 ADD + 1 WIRE + 1 ADD(gap) = 7|V02|Covered|
|[iem.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE + 1 FIX + 1 ADD = 3|V04|Covered|
|[noaa_file.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|2 ADD|V04|Covered|
|[cleaner.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 WIRE + 1 OWN + 1 DROP + 1 SAVE = 4|V03|Covered|
|[parquet_store.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ENFORCE|V02|Covered|
|[manager.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 REMOVE + 1 ADD(station param) + 1 ADD(build_encoder_obs_window) = 3|V03|Covered|
|[mc.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|2 REMOVE + 1 CHANGE + 1 ADD(fit_cov) + 1 FIX + 1 ADD(max_bias) = 6|V05|Covered|
|[tft.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE + 1 MOVE + 1 FIX = 3|V05|Covered|
|[pipeline.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|3 FIX + 1 FIX(merge tol) + 4 ADD = 7 (all 6+4 in doc)|V05|Covered|
|[registry.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 FIX + 3 ADD = 4|V05|Covered|
|[forecast_loop.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 SPLIT + 1 CHANGE + 1 ADD(ProbStore write) = 3|V10|Covered|
|`loops/trading_loop.py [NEW]`|7 ADD/CHANGE/ENFORCE items|V11|Covered|
|[discovery_loop.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE + 1 ADD + 1 REMOVE = 3|V09|Covered|
|[discovery.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE (registry) + 1 ADD (entries) = 2|V08|Covered|
|`core/probability_store.py [NEW]`|1 ADD (full impl)|F04|Covered|
|[positions.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|8 ADD|F03|Covered|
|`core/alerts.py [NEW]`|2 ADD (AlertEvent + AlertBus)|F02|Covered|
|[base.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE + 1 ADD = 2|V06|Covered|
|[edge.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 FIX + 1 ADD = 2|V06|Covered|
|`core/strategy/multi.py [NEW]`|1 ADD (MultiStrategy)|V06|Covered|
|[base.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|2 CHANGE|V06|Covered|
|[fixed_fraction.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ADD (dup prevention)|V06|Covered|
|`core/risk/kelly.py [NEW]`|1 ADD (stub)|V06|Covered|
|`core/risk/pipeline.py [NEW]`|1 ADD (RiskPipeline)|V06|Covered|
|[clob.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ADD (order_failure alert)|V07|Covered|
|[paper.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 CHANGE + 1 ADD(OBFetcher) + 1 ADD(alert) = 3|V07|Covered|
|[bot.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 FIX + 1 CHANGE = 2|V07|Covered|
|[app.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|2 ADD (staleness display, Telegram stub)|V13|Covered|
|[lifecycle.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 REMOVE + 1 CHANGE + 1 ADD = 3|V12|Covered|
|[engine.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html)|1 ADD + 1 CHANGE = 2|V05|Covered|
|`loops/settlement_loop.py [NEW]`|4 ADD + 1 WIRE = 5|V12|Covered|
|`snipers/sniper_manager.py [RENAME]`|1 MOVE + 1 CHANGE (import sites) = 2|V01|Covered|
|Testing strategy -- all targets|distributed across all slices|each slice|Covered|
|DataHealthMonitor|—|—|Deferred|
|ModelHealthMonitor|—|—|Deferred|
|SystemHealthMonitor|—|—|Deferred|
|Full Telegram routing/dedup/recovery/dashboard|—|—|Deferred|

**Uncovered items: none**

## Questions for you

1. **Granularity** — V05 (Model Pipeline) is the thickest slice: mc.py + tft.py + pipeline.py + registry.py + [engine.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) all in one. This is because they're tightly coupled (pipeline calls both mc and tft, and registry is wired into pipeline). Do you want it split into V05a (mc + registry) and V05b (tft + pipeline + backtest), or keep together?
    
2. **V02 vs H01 dependency** — V02 (HRRR multi-station) is written to work dynamically from the STATIONS dict, so it is not blocked by H01 (new stations). New stations become active automatically once H01 lands. Is that the right read, or do you want a thin "add-stations" wiring slice after H01?
    
3. **V09 + V12 ordering** — The cutoff removal in [lifecycle.py](vscode-file://vscode-app/c:/Users/yorkj/AppData/Local/Programs/Microsoft%20VS%20Code/8b640eef5a/resources/app/out/vs/code/electron-browser/workbench/workbench.html) is in V12, but the cutoff removal in `discovery_loop.py` is in V09. V09 does not depend on V12 (the loop cutoff removal is independent of the lifecycle cutoff removal). Correct?
    
4. **HITL/AFK marks** — Do you agree that H01 is the only true HITL slice? Everything else is AFK.
    
5. **V11 dependencies** — V11 (TradingLoop) is the deepest dependency node: blocked by F01-F04, V06, V07, V10. This means it can only land after ~12 other slices. Is that acceptable, or should we find a way to stub earlier?
请用中文依次回答。