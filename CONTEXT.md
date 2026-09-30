# PolyWeatherData Context

Polymarket daily maximum-temperature market calibration: measures whether CLOB
taker prices behave like probabilities, and how the favorite-longshot bias varies
with the horizon to the daily temperature peak.  **No model inference, no order
submission** — this is a market-calibration study, not a trading system.

The paper pipeline lives in `pipeline/` and is read-only with respect to the raw
artifacts. Reproducing the paper requires only `pipeline/`; `data/`, `config/`,
and `scripts/` exist to rebuild the raw artifacts from the public API.

## Language

### Weather

**daily temperature peak**: The moment of the daily maximum temperature for a
station-local calendar day, obtained from an observation source. All horizons are
measured backwards from this timestamp (`daily_high_argmax.parquet`). This is the
anchor for the whole analysis — *not* the market's settlement boundary.
_Avoid_: event day, settlement time, UTC day

**horizon h**: Hours before the daily temperature peak, from the fixed grid
`[48, 42, 36, 30, 24, 18, 12, 9, 6, 3, 1]`. Smaller h is closer to the peak.
_Avoid_: lead time, time-to-event, remaining time

**station-local calendar day**: A calendar day defined in the station's local timezone.
_Avoid_: UTC day, event day

### Market

**Event**: One city-date maximum-temperature market, identified by a slug and
carrying a `target_date`. It holds 7–11 mutually exclusive Buckets.
_Avoid_: market (overloaded)

**Market / condition**: A YES/NO binary sub-market within an Event, corresponding
to a single temperature bucket. `trades` joins on its `condition_id`.
_Avoid_: event (overloaded), contract

**Bucket**: One temperature-range bin defined by a Market's slug; bounds are parsed
into `bucket_lo_f` / `bucket_hi_f` (open-ended buckets carry $\pm\infty$).
_Avoid_: bin, range

**slug**: The URL-safe string identifier of a Market, parsed by
`data/clob/slug.py::parse_slug` to extract metric, location, date, and bucket bounds.
_Avoid_: market id

### Price and calibration

**horizon price**: At horizon $h$, a Bucket's price is the **mean of all taker
trade prices** in the $h \pm 0.5$ hour window before the daily temperature peak
(`WIN_HALF_H = 0.5`). This single definition governs every reported number.
_Avoid_: midpoint, L2 price, snapshot price

**range bias** $D_h(a,b)$: $\operatorname{mean}(y - p)$ over Buckets whose price
lies in $[a,b]$ at horizon $h$, where $y$ is the realised outcome. Negative means
price exceeds the observed win rate (overpricing). Favorites are $0.75 \le p \le
0.95$; longshots are $0 \le p \le 0.10$.
_Avoid_: mispricing (use the precise estimator name), calibration gap

**weighting scheme**: Two are reported. **event-weighted** averages within an
Event then equally across Events; **station-equal** averages per city then equally
across the ten cities. Never mix them in one table.
_Avoid_: unweighted, pooled

**event-level score**: Brier, Log Loss, and RPS computed on the normalized vector
$q_k = p_k / \sum_j p_j$ over an Event's Buckets. Only **full-coverage** Events
(every Bucket priced at that horizon, at least 7 Buckets) qualify.
_Avoid_: bucket-level score, per-contract error

**recalibration**: A cubic B-spline on $z = \operatorname{logit}(p)$,
$\operatorname{logit}(\hat p) = w_0 + \sum_{j=1}^{6} w_j B_j(z)$, with 2 inner
knots at the 0.35 / 0.65 quantiles of training logit prices (7 parameters total).
Fitted on training only, applied once to July.
_Avoid_: isotonic, Platt scaling

**time split**: Chronological by `target_date` — train 2025-12-05…2026-05-31,
validation June 2026, test July 2026. Never random-split.
_Avoid_: train/test split, k-fold

**Buy-NO**: A diagnostic that takes the opposite side of a range bias. Fee follows
the Polymarket schedule $F = C f r (1-r)$ with $f = 0.05$, $C = 10/r$ shares,
$r = 1-p$ the NO price, \$10 notional. Excludes depth, spread, and slippage, so it
does not establish executable profits.
_Avoid_: alpha, tradable edge, strategy return
