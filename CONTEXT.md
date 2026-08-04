# PolyWeatherData Context

Polymarket weather-market mispricing analysis: compares CLOB market probabilities
against IEM settlement truth across time, stations, and lead-time strata.  No model
inference, no HRRR download, no order submission.

## Language

### Weather

**station-local calendar day**: A calendar day defined in the station's local timezone.
The daily high temperature is the maximum IEM observation within that local day.
_Avoid_: UTC day, event day

**lead time**: The wall-clock time remaining from a given CLOB snapshot tick to the
station-local event settlement boundary.  Binned into bands (e.g. &lt;1h, 1-6h, 6-24h,
1-3d, 3-7d) for stratum-level analysis.
_Avoid_: time-to-event, remaining time

### Market

**Event**: A Polymarket weather-event record covering a specific metric for a specific
day and location, identified by a slug.
_Avoid_: market (overloaded)

**Market**: A YES/NO binary sub-market within an Event, corresponding to a single
temperature bucket (e.g. "above 90 deg F").
_Avoid_: event (overloaded), contract

**Token**: The YES or NO side of a Market; the atomic unit bought/sold on the CLOB.
_Avoid_: contract, side

**Bucket**: One temperature-range bin defined by a Market's slug.
_Avoid_: bin, range

**slug**: The URL-safe string identifier of a Market, parsed to extract metric, location,
date, and bucket bounds.
_Avoid_: market id

**normalized synthetic midpoint**: At one complete 5-minute Event tick, the vector of
past-public-trade synthetic YES midpoints divided by its cross-Bucket sum. The raw sum
is retained as overround evidence.  This is the primary market-probability source for
mispricing analysis.
_Avoid_: executable midpoint, L2 price

**Event settlement boundary**: The end of the station-local target calendar day;
lead time is measured to this boundary, independently of the later CLOB `closed_at`.
_Avoid_: market close, event_end_at

**market probability**: The CLOB-implied YES probability for one Bucket at one
point-in-time tick, derived from the normalized synthetic midpoint vector or from
bid/ask depth.  It is a public-price benchmark, not a model estimate.  Sourced from
Polymarket Gamma API historical data at 5-minute snapshot granularity; stored in
ParquetStore partitioned by station.
_Avoid_: model probability, true probability, executable price

**settlement truth**: The binary (0/1) post-event outcome for one Bucket: 1.0 if the
station-local daily high temperature falls in the Bucket's [lo, hi) range, 0.0
otherwise.  Primary source is IEM ASOS daily-high (broad station coverage, long
history); Weather.com is a supplementary source for recent stations.  Never imputed.
_Avoid_: ground truth, actual outcome, resolved value

**mispricing**: A persistent deviation between market probability and the aggregated
settlement-truth rate within a declared time/space/lead-time stratum.  Measured via
calibration gap = E[settlement_truth] - E[market_probability] per stratum, with
paired bootstrap significance testing and rolling-window stability diagnostics.
Generaliseable to new Events without model retraining.
_Avoid_: edge, alpha, model error

**calibration gap**: The stratum-level difference between realised settlement-truth
rate and mean market probability.  Positive gap = market underpricing; negative gap =
market overpricing.  Reported with bootstrap 95% CI.
_Avoid_: predictor error, mispricing (calibration gap is the specific estimator)

**stratum**: A declared analysis slice defined by (station, lead_time_band,
calendar_period).  Each stratum must have a minimum sample size (e.g. &gt;= 30 Events)
to be included in significance testing.
_Avoid_: slice, cohort, segment
