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

**HRRR fill**: Extending a non-MAJOR init_time's 18-hour forecast to 48 hours by splicing
in fxx values from the nearest prior MAJOR-hour run.
_Avoid_: padding, interpolation

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

**edge**: The difference between the model's estimated probability for a Token and the
market's implied probability (mid_price); the primary trading signal.
_Avoid_: alpha, signal (edge has a precise meaning here)

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

**empirical covariance**: The 48x48 covariance matrix estimated from historical
residuals across decoder time steps; used by McSampler to capture inter-hour correlations.
_Avoid_: AR(1), correlation matrix

**McSampler**: The Monte Carlo module that draws n_samp joint residual samples from
TFT quantiles using the empirical covariance matrix, then maps samples to bucket
probabilities via temperature reconstruction.
_Avoid_: MC, sampler

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

**ForecastLoop tick**: The 1-hour recurring cycle: fetch HRRR+IEM, run Cleaner, run
TFT+McSampler, write bucket probabilities to shared state.
_Avoid_: hourly loop, prediction loop

**TradingLoop tick**: The 5-minute recurring cycle (interval configurable via
`trading_loop_interval_s`): for each active Event, read its bucket probabilities from
ProbabilityStore, check staleness, run strategy+risk, submit orders to CLOB.
Runs continuously until all active Events are settled; never has a cutoff hour.
_Avoid_: main loop

**ProbabilityStore**: The in-process shared-memory object written by ForecastLoop and
read by TradingLoop. Holds, per condition_id ¡Á day_offset ¡Á bucket: the MC bucket
probability, plus the TFT p10/p50/p90 quantiles for each decoder hour, plus
`last_tick_utc` for staleness computation.
_Avoid_: probability cache, result store

**DiscoveryLoop**: The hourly recurring cycle that fetches all currently-active
Polymarket weather Events and persists any new ones to the database. Has no cutoff
hour ¡ª it runs as long as the bot runs; if no new Events exist, it is a no-op.
_Avoid_: event loop (conflicts with asyncio term)

**PaperExchange**: A paper-trading exchange backed by live CLOB order book data (not
synthetic mid-prices).
_Avoid_: mock exchange, simulator

**ParquetStore partition**: A quarterly time-partitioned Parquet file storing f01 HRRR
values and IEM observations.
_Avoid_: parquet file, storage file

## Data & Signal Flow

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
- **ForecastLoop** -> **TFT** -> **McSampler** -> bucket probabilities -> **TradingLoop**
- **DiscoveryLoop** owns **Event** / **Market** / **Token** records; **TradingLoop**
  references them by slug only. At any given time 2-3 Events may be concurrently active.
- **TradingLoop** iterates over all active Events each tick; it runs until every
  active Event is settled - there is no cutoff hour.
- **cold-start** rows (first 72 h of any dataset) are dropped by **Cleaner** before
  any model input or backtest entry.
- **staleness** is computed by **TradingLoop** as `now - ForecastLoop.last_tick_utc`;
  no other component tracks it.

## Layer Exceptions

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
  bucket contract ¡ª resolved: **Event** vs **Market**.
- "model time" used to mean init_time ¡ª resolved: use **init_time** exclusively.
- "encoder length" set to 24 in the original code (tft_step7 bug) while the design
  requires 72 ¡ª resolved: **encoder window = 72 hours**.
