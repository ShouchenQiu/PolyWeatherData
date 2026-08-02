# KLGA historical transaction calibration

## Decision intent

This study tests whether resolved Token outcome rates match historical public
transaction prices, and where any mismatch concentrates by price and time to the
station-local target-day boundary. It is a development-only calibration study. It
does not consume `post_test_holdout` and cannot establish executable edge without
historical L2 ask/bid depth.

The approved questions are preserved in `question_ledger.json`; each completion
answer is `supported`, `contradicted`, `inconclusive`, or `not_measured`.

## Time and truth contract

- The **Event settlement boundary** is the end of the KLGA target date in
  `America/New_York`, converted to UTC. `event_end_at` is not an eligibility cutoff.
- `forecast_relevant` requires a trade before both the local-day boundary and its
  Market `closed_at`.
- `resolution_latency` contains trades after the local-day boundary but before the
  contract closes. It is reported separately and never enters the primary estimate.
- `invalid_post_close` is excluded.
- Token truth is the resolved YES/NO value. Weather.com actual high independently
  selects the unique Bucket and must reconcile with Polymarket `resolved_yes`.
- The grouped empirical outcome rate is an estimator; it is not a per-trade latent
  "true probability."

The schema-v1 design probe contained 696,024 fills: 689,051 forecast-relevant,
6,945 resolution-latency, and 28 after contract close. Formal counts must come from
the rebuilt schema-v2 artifact and may not be copied from the probe.

## Estimands

`transaction calibration gap = empirical Token outcome rate - mean Token price`

Three weighting views are retained:

- `fill`: paper-style transaction weighting;
- `event_contract_equal`: duplicate fills collapse within Event/Market/cell, then
  Events receive equal weight; this is primary;
- `usdc`: observed capital exposure, descriptive only.

The report includes 10%, 5%, 2%, and 1% price bins. The 1% × frozen lead-band family
uses Event-cluster bootstrap intervals with Bonferroni family control. A development
candidate additionally needs fill/Event direction agreement, at least four consistent
historical folds, no significantly opposite fold, and the declared sample minimums.

Observed settlement return uses only actual `BUY` Token fills and the canonical taker
fee curve. A negative Token gap is not converted into a synthetic `1-p` opposite-side
execution.

## Immutable artifact flow

1. Refresh the Weather.com cache to retain each complete local day's raw-observation
   maximum, then rebuild actual highs as schema v2. Incomplete local days become
   hash-bound truth exclusions and never enter analysis; hourly representative values
   are not substituted for the raw daily high.
2. Rebuild CLOB schema v2 from the existing raw cache into a new directory.
3. Rebuild hash-bound 5-minute synthetic quotes from that CLOB manifest.
4. Run `scripts.run_market_calibration`; its integrated audit must pass before any
   calibration output is written.

Suggested immutable paths:

```text
data_store/backtest/actuals/KLGA/20250401_20260331_truth_v2.csv
data_store/backtest/clob/KLGA/20250401_20260331_resolution_v2/
data_store/backtest/quotes/KLGA/20250401_20260331_resolution_v2_5min/
data_store/backtest/market_calibration/KLGA/20250401_20260331_v1/
```

The eligible and excluded truth dates must be disjoint and together exactly cover the
requested range. A missing cache file or malformed truth schema still stops the build.
The final artifact contains transaction and resolution-latency panels, calibration
tables, price/lead cells, fold evidence, candidates, observed BUY returns, complete
synthetic-midpoint vector scores, input audit tables, a report, a question ledger,
three HTML plots, hashes, and the frozen protocol.

## Interpretation guardrails

- A significant fill-weighted gap alone may be activity weighting, repeated fills,
  overround, or resolution latency.
- Six-fold agreement is still `oof_development` evidence, not independent validation.
- Synthetic midpoint is a past-trade benchmark, not historical L2.
- `input_truth_exclusions.parquet` records every source-incomplete date removed before
  event resolution reconciliation and transaction-panel construction.
- `executable_edge` remains `not_measured` until a separately frozen study prices the
  intended size from real historical L2 and accounts for capacity and queue risk.
