# PolyWeatherBot Context

Automated weather-event trading system: ingests HRRR forecasts and IEM observations,
feeds a TFT+MC model, and submits orders on Polymarket weather binary markets.

## Language

### Weather

**init_time**: The UTC hour at which a HRRR model run is initialised; every data row is
keyed to one init_time.
_Avoid_: run time, model time, forecast time

**fxx**: Forecast hour offset from init_time (f01 = 1 h ahead, f48 = 48 h ahead).
_Avoid_: lead time (ambiguous)

**f01**: The 1-hour-ahead HRRR value for a given init_time; the only fxx value persisted
per init_time in ParquetStore.
_Avoid_: first-hour forecast

**MAJOR hour**: One of the four HRRR init_times per day (00z, 06z, 12z, 18z) that produce
a full 48-hour forecast horizon; non-MAJOR init_times produce only 18 h.
_Avoid_: full run, extended run

**HRRR fill**: Extending a non-MAJOR init_time's 18-hour forecast to 48 hours by
splicing valid_times from the nearest prior MAJOR-hour run, then persisting the
last available values across only the final uncovered 1..5 hours. Earlier MAJOR
runs are not consulted because they end earlier; following MAJOR runs are
forbidden future information.
_Avoid_: padding, interpolation

**trusted decoder horizon**: The sourced prefix of a 48-hour decoder window,
whose length is 48 minus the non-MAJOR init's hour offset from its prior MAJOR.
_Avoid_: model horizon, available horizon

**tail fallback**: The final 1..5 decoder hours backed by persisted HRRR inputs,
excluded from TFT loss and trading calculations, and presented using the last
trusted TFT quantiles.
_Avoid_: TFT prediction, extrapolated forecast

**encoder window**: Past 72 hours of (f01, IEM obs) pairs fed to the TFT encoder,
indexed -71..0 relative to current init_time.
_Avoid_: history window, lookback

**decoder window**: Future 48 hours of HRRR fxx values fed to the TFT decoder,
indexed 1..48 relative to current init_time.
_Avoid_: forecast window, prediction horizon

**cold-start**: The first 72 hours of any dataset where the encoder window cannot be
fully populated; these rows are dropped by Cleaner before model input or backtest.
_Avoid_: warm-up, burn-in

**staleness**: The age in minutes of the most recent ForecastLoop output; when stale,
TradingLoop widens its edge threshold or halts.
_Avoid_: latency, delay

### Market

**Event**: A Polymarket weather-event record covering a specific metric for a specific
day and location, identified by a slug.
_Avoid_: market (overloaded)

**Market**: A YES/NO binary sub-market within an Event, corresponding to a single
temperature bucket (e.g. "above 90 deg F").
_Avoid_: event (overloaded), contract

**Token**: The YES or NO side of a Market; the atomic unit bought/sold on the CLOB.
_Avoid_: contract, side

**Bucket**: One temperature-range bin defined by a Market's slug; the model outputs one
probability per bucket.
_Avoid_: bin, range

**slug**: The URL-safe string identifier of a Market, parsed to extract metric, location,
date, and bucket bounds.
_Avoid_: market id

**edge**: The difference between the model's estimated probability for a Token and its
fee-adjusted executable price for the intended order size; the primary trading signal.
Mid-price is display evidence, not an executable edge input.
_Avoid_: alpha, signal (edge has a precise meaning here)

**conservative executable edge**: The lower confidence bound of the calibrated Token
probability minus its fee-adjusted executable price. Diagnostic replay may enter only
when this value clears its frozen threshold.
_Avoid_: raw edge, midpoint edge

**normalized synthetic midpoint**: At one complete 5-minute Event tick, the vector of
past-public-trade synthetic YES midpoints divided by its cross-Bucket sum. The raw sum
is retained as overround evidence. This is a public-price benchmark for signal
evaluation, not a historical L2 quote or an executable price.
_Avoid_: market probability, executable midpoint

**Forecast Signal Study**: A strategy-free `oof_development` comparison of raw
ForecastCache Bucket probabilities with the normalized synthetic midpoint on identical
complete Event/tick vectors. It reports proper scores, calibration, sharpness, and
paired event-day uncertainty, but has no promotion gate and never consumes holdout.
_Avoid_: Edge Study, strategy backtest

**signal advantage slice**: A development-only, ex-ante subset of Forecast Signal
Study vectors used to test where model probabilities improve on the normalized
synthetic midpoint. Distribution scores use all eligible ticks with Event-equal
weighting; Token evidence uses only the first eligible tick per Event. Its theoretical
midpoint excess excludes execution, fees, capacity, and fill risk.
_Avoid_: strategy, tradable edge, profitable segment

**frozen forecast distribution validation**: A non-search validation artifact that
freezes signal advantage slice thresholds and families from `oof_development`, then
scores them on an independent forecast split such as `model_test`. It validates
forecast distribution advantage with proper scores and true-Bucket probability
evidence; it does not recalibrate, select rules, or establish executable edge.
_Avoid_: slice discovery, strategy validation, holdout optimisation

**Event settlement boundary**: The end of the station-local target calendar day;
Edge Study lead time and decision eligibility use this boundary, independently of
the later CLOB `closed_at`.
_Avoid_: market close, event_end_at

**CLOB**: The Polymarket central limit order book; the live venue for order submission,
cancellation, and fill tracking.
_Avoid_: exchange, book

**Relayer path**: The on-chain settlement route via Polymarket's Relayer contract;
distinct from direct on-chain execution.
_Avoid_: on-chain settlement (ambiguous)

### Model

**TFT**: TemporalFusionTransformer; takes encoder+decoder windows and outputs residual
quantiles (p10/p50/p90) for each of the 48 decoder hours.
_Avoid_: transformer, the model

**residual (temp_error)**: The target variable: IEM observed temperature minus f01 HRRR
forecast for the same hour; what TFT is trained to predict.
_Avoid_: error, delta

**empirical error correlation**: The 48x48 correlation matrix estimated from
standardized validation errors. McSampler combines it with each forecast's calibrated
dynamic sigma.
_Avoid_: AR(1), fixed covariance

**calibrated model contract**: A contract-v2 TFT artifact with the baseline observation
and HRRR inputs plus a checkpoint-bound MC calibration bundle.
_Avoid_: residual-input model, B model

**McSampler**: The Monte Carlo module that draws n_samp joint residual samples from
TFT quantiles using empirical error correlation, then maps samples to bucket
probabilities via temperature reconstruction.
_Avoid_: MC, sampler

**ForecastCache bucket probability**: The point-in-time YES settlement probability for
one `(init_time, target_date, condition_id)`, obtained from a versioned daily-high
distribution source under the Event's integer-Fahrenheit resolution rule; it becomes
actionable only at the snapshot's `available_at`.
_Avoid_: model price, 48-hour maximum probability

**daily-high PMF pilot**: An offline point-in-time model artifact that assigns
probability directly to each integer-Fahrenheit settlement high using April
chronological OOF training and frozen May diagnosis, without consuming June or serving
production probabilities.
_Avoid_: production model, MC recalibration

**future-max Gaussian pilot**: An offline, low-capacity alternative that predicts a
truncated-normal distribution for the still-unobserved local-day maximum, then composes
the final daily high exactly as `max(observed_high, future_max)`. It is parallel to the
PMF pilot, preserves ForecastCache bucket semantics, and never serves production
probabilities.
_Avoid_: final-high truncation, production Gaussian

**historical cross-fit TFT trajectory**: A point-in-time hourly prediction produced by
a TFT trained and early-stopped only on earlier calendar ranges. It supplies offline
distribution-model training features but is neither a production checkpoint nor a
market-probability source.
_Avoid_: historical in-sample prediction, promoted TFT

**point-in-time partial-day daily high**: A target local-day distribution whose elapsed
hours come from observations visible by `init_time` and whose later hours come from
non-extrapolated MC forecasts over aligned slots `(local D 00:00, local D+1 00:00]`,
emitted only when their union covers the complete day.
_Avoid_: remaining-hours maximum, partial maximum

**max_bias**: Per-station scalar correction for the upward bias introduced when taking
max() over McSampler samples (Jensen's inequality effect). Measured on the validation
set as E[max(samples)] ? mean(observed daily max). Stored in the registry and subtracted
from the MC expected maximum before bucket assignment. Replaces the hardcoded
`bias_corr_c` from the original Herbie codebase.
_Avoid_: bias_corr_c, correction factor

**p10/p50/p90**: 10th, 50th, and 90th percentile quantiles output by TFT for each of
the 48 decoder hours; inputs to McSampler.

**time_idx**: Global integer index on the time axis used by the TFT DataLoader to
locate rows on the full timeline and delimit encoder/decoder boundaries.
_Avoid_: row index, step

### Ops

**training factory**: The offline workflow that prepares historical weather
data, audits readiness, trains TFT experiments, and packages candidate evidence.
_Avoid_: remote training, training script

**training workspace**: An isolated `runs/<RUN_ID>` state root containing
prepared data, audit evidence, checkpoints, candidate artifacts, and stage
manifests for one training run.
_Avoid_: run folder, remote run

**experiment template**: A version-controlled schema-v1 JSON specification that
fixes one station, audited data splits, TFT defaults, search space, and execution
policy before a study is created.
_Avoid_: run config, Optuna config

**training study**: A managed collection of smoke, calibration, and optimisation
executions that all reference one ready training workspace and one immutable
experiment-template revision.
_Avoid_: run, training workspace

**backtest study**: A governed trading-research lifecycle bound to immutable forecast,
synthetic-quote, CLOB, and resolution-truth hashes plus one split role. Development uses
only `oof_development`; trading research cannot use `model_test`; a frozen study may
consume `post_test_holdout` once.
_Avoid_: training study, ad-hoc replay

**diagnostic replay**: A signal-quality replay with an independent fixed risk per Event,
at most one first qualifying Token entry per Event, and settlement-only exit. It excludes
adaptive sizing, adding, and reversal so money management cannot mask signal quality.
_Avoid_: portfolio backtest, strategy optimisation

**trial snapshot**: The immutable resolved configuration and evidence for one
managed training execution, including its exact hyperparameters, data fingerprint,
environment, status, metrics, timings, and artifacts.
_Avoid_: experiment config, checkpoint metadata

**test lock**: The training-factory rule that smoke, calibration, and optimisation
executions cannot load or score the test split; one approved final evaluation may
unlock it without returning its result to the search process.
_Avoid_: test flag, holdout option

**search pause**: A durable training-study request that lets the current Optuna
trial finish, prevents another trial from being claimed, and leaves the study
resumable from the same catalog and Optuna storage.
_Avoid_: cancellation, process suspension

**candidate model package**: A reviewed-but-inactive model artifact archive that
can be imported into a production model store without changing its active model.
_Avoid_: production model, deployed model

**promotion**: The explicit production operation that selects one imported
model version by writing the active-version state and an audit record.
_Avoid_: import, publish

**decoder cache**: Training-only persisted HRRR fxx 1..48 windows used to build
TFT decoder samples and never read by production inference.
_Avoid_: forecast store, production cache

**decoder baseline report**: Training-workspace evidence that aligns cached HRRR
decoder cells with canonical IEM observations and records raw-HRRR error metrics
per cell, init, forecast hour, and UTC init month. Performance outliers are
diagnostic and do not invalidate otherwise complete training data. Production
inference never reads this report.
_Avoid_: model evaluation, decoder cache, production metric

**ForecastLoop tick**: The 1-hour recurring cycle: fetch HRRR+IEM, run Cleaner, run
TFT+McSampler, write bucket probabilities to shared state.
_Avoid_: hourly loop, prediction loop

**TradingLoop tick**: The 5-minute recurring cycle (interval configurable via
`trading_loop_interval_s`): for each active Event, read its bucket probabilities from
ProbabilityStore, check staleness, run strategy+risk, submit orders to CLOB.
Runs continuously until all active Events are settled; never has a cutoff hour.
_Avoid_: main loop

**day_offset**: Integer offset from `init_utc.date()` (UTC) to the target Event's
`event_end_date`; values in {0, 1, 2} meaning today/tomorrow/day-after. Used as part
of the ProbabilityStore key `(condition_id, day_offset)`. Controls which local-calendar-day
boundary the pipeline uses to filter fxx hours for the daily-high computation: the
target window is `[target_local_day 00:00, target_local_day 24:00)` in the station's
local timezone, converted to UTC and mapped to fxx offsets from init_time.
_Avoid_: day index, target day index

**ProbabilityStore**: The in-process shared-memory object written by ForecastLoop and
read by TradingLoop. Holds per `(condition_id, day_offset)` key: a `ProbabilitySnapshot`
containing the per-bucket `mc_prob` float map, the `quantiles_by_fxx` dict (shared
across all buckets for the same key), and `last_tick_utc` for staleness computation.
_Avoid_: probability cache, result store

**DiscoveryLoop**: The hourly recurring cycle that fetches all currently-active
Polymarket weather Events and persists any new ones to the database. Has no cutoff
hour -- it runs as long as the bot runs; if no new Events exist, it is a no-op.
_Avoid_: event loop (conflicts with asyncio term)

**PaperExchange**: A paper-trading exchange backed by live CLOB order book data (not
synthetic mid-prices).
_Avoid_: mock exchange, simulator

**ParquetStore partition**: A quarterly time-partitioned Parquet file storing f01 HRRR
values and IEM observations.
_Avoid_: parquet file, storage file

## Data & Signal Flow

- HRRR/IEM source -> **Cleaner** -> **ParquetStore partition** -> **training
  workspace** -> training dataset -> **candidate model package** -> import ->
  **promotion**.
- Importing a **candidate model package** never changes production active state;
  only **promotion** may select it.
- A lower-trust training machine may produce candidate artifacts and evidence,
  but only `john_linux` may import or promote them.

- One **init_time** may cover up to 48 **fxx** offsets; only **f01** is stored per
  init_time in **ParquetStore**.
- **encoder window**: 72 rows (-71..0), each = (f01, IEM obs); owned by **Cleaner**
  after all feature engineering is applied.
- **decoder window**: 48 rows (1..48), each = (HRRR fxx, IEM target).
- **Cleaner** owns all weather-domain and meteorological feature engineering (wind
  components, log1p smoothing, temp_error column); no other module performs
  meteorological transforms. TFT model-adapter features (time_idx, Node_ID, calendar
  encodings such as sin_hour/cos_hour/sin_doy/cos_doy) are added by
  `ModelPipeline._add_model_features` immediately before inference.
- One `(init_time, target_date)` -> one McSampler daily-high distribution -> one
  **ForecastCache bucket probability** per Market; `available_at`, not `init_time`,
  controls when trading research may consume it.
- **Forecast Signal Study** runs before strategy research. It requires all Event
  Buckets at one quote tick, joins every Bucket to one latest-visible Forecast
  snapshot, and compares `model_raw` with the **normalized synthetic midpoint** using
  Event-equal scoring. Events without a valid overlap remain explicit coverage
  exclusions; stale prices and future forecasts are never filled to improve coverage.
- A **signal advantage slice** may nominate a strategy candidate only from
  `oof_development`. Adaptive thresholds are fitted outside contiguous held-out
  event-day folds, and the result remains development evidence until a separately
  frozen executable strategy is evaluated.
- Edge Study observations require quote ticks before both the CLOB close and the
  **Event settlement boundary**; lead-time strata measure time to settlement, not
  time to the venue's later close.
- An in-progress Event uses a **point-in-time partial-day daily high**; uncovered
  right-tail hours or extrapolated decoder cells suppress the snapshot.
- Offline distribution pilots may write a parallel ForecastCache only; neither the
  production `ModelPipeline` nor `ProbabilityStore` may select them without an explicit
  promotion decision and separate holdout evidence.
- **ForecastLoop** -> **TFT** -> **McSampler** -> bucket probabilities -> **TradingLoop**
- **DiscoveryLoop** owns **Event** / **Market** / **Token** records; **TradingLoop**
  references them by slug only. At any given time 2-3 Events may be concurrently active.
- **TradingLoop** iterates over all active Events each tick; it runs until every
  active Event is settled - there is no cutoff hour.
- **cold-start period**: the first `cold_start_hours` (72 h) of any dataset/station are
  a data-pipeline burn-in window; **Cleaner** drops these rows permanently at write time.
  Distinct from encoder_length (also 72 h): the two constraints are additive, so the
  first eligible `init_time` for a new dataset is `data_start + 144 h`.
- **first eligible init**: `data_start + cold_start_hours + encoder_length`; the earliest
  `init_time` at which `DataManager.inference_readiness()` returns `ready=True`.
- **staleness** is computed by **TradingLoop** as `now - ForecastLoop.last_tick_utc`;
  no other component tracks it.

## Layer Exceptions

### Training factory and CLI adapters

`training/` may import `data/`, `config/`, `db/`, `model.features`, and
`model.contract`. Production `model/`, `core/`, `loops/`, and `backtest/`
modules must not import `training/`. Files in `scripts/` are command-line
adapters and do not own training or model-store implementations.

### `core/bot.py` -- BotManager wiring root

`BotManager` in `core/bot.py` is the top-level wiring root: it constructs and
connects all subsystems (data layer, exchange, loops, snipers, DB) in a single
`_build()` method.  As a result it imports from `data/`, `loops/`, `model/`,
`snipers/`, and `db/` -- all of which nominally sit "above" `core/` in the
standard layer hierarchy.

This is an intentional architectural exception.  `BotManager` is conceptually
outside all layers; it lives in `core/` solely for historical reasons.  Do not
apply the standard `core/` import restriction when reviewing `bot.py`.  If the
wiring root is ever extracted to a dedicated top-level `bot.py` module alongside
`main.py`, update this note.


## Flagged ambiguities

- "market" used to mean both the Polymarket top-level event page and a single YES/NO
  bucket contract -- resolved: **Event** vs **Market**.
- "model time" used to mean init_time -- resolved: use **init_time** exclusively.
- "encoder length" set to 24 in the original code (tft_step7 bug) while the design
  requires 72 -- resolved: **encoder window = 72 hours**.
