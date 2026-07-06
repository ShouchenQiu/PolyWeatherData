# Evaluate forecast signal before strategy research

**Status**: accepted

Forecast quality is evaluated in an independent `ForecastSignalStudy` before any
execution calibration, entry rule, position size, fee, or PnL is introduced. The study
accepts only hash-validated `oof_development` ForecastCache, synthetic quotes, raw CLOB,
and settlement truth. It never registers or freezes a trading strategy and cannot
consume `model_test` or `post_test_holdout`.

The comparison sample is one complete Event/tick probability vector: all Markets must
have a past-only synthetic midpoint, all model probabilities must come from the same
latest-visible Forecast snapshot, and truth must uniquely resolve to one ordered
Bucket. Raw midpoint sums remain overround diagnostics; scoring uses the cross-Bucket
normalized synthetic midpoint. Model and benchmark receive identical probability
flooring and samples.

Multiclass Brier, logarithmic score, and normalized ranked probability score are the
primary proper scores. Fixed-bin ECE/MCE, entropy, maximum Bucket probability, top-1
accuracy, true-Bucket probability, and pooled binary Brier are supporting evidence.
Events are weighted equally, with ticks weighted equally within Event. Score
differences use paired event-day bootstrap intervals and are repeated for fixed
lead-time bands. The artifact is descriptive and has no automatic pass gate.

**Considered alternatives**: The execution-calibrated probability is price-informed
and would leak the benchmark into the model side of a pure signal comparison.
Unnormalized midpoint vectors are not probability distributions. CRPS would require
invented numeric endpoints for open-ended temperature Buckets; RPS uses only their
natural order. Strategy PnL mixes forecast quality with liquidity, fee, timing, and
money-management decisions.

**Consequences**: Strategy research can cite an immutable, hash-verified signal
artifact without rewriting an existing Edge Study. Missing complete-vector overlap is
reported as a coverage exclusion rather than filled with stale midpoint data or future
forecasts. A favorable point estimate is evidence, not promotion approval; uncertainty
and lead-band behavior remain visible.

Adaptive advantage-slice discovery is a child artifact of this study. It binds the
parent manifest hash and exact ForecastCache, uses contiguous event-day cross-fitting
for learned thresholds, and limits automatic rules to declared ex-ante features and a
maximum depth of two. Distribution evidence averages all eligible ticks within Event;
strategy-oriented Token evidence uses only the first qualifying tick and one stable
model-favored YES or NO Token.

The resulting theoretical midpoint excess is not executable edge: it omits VWAP,
fees, capacity, and fill risk. Slice tiers are development labels for the next strategy
study, not promotion gates. Because development outcomes have already informed the
slice family, even cross-fitted and multiplicity-adjusted results are not independent
confirmation and cannot consume or stand in for the one-use holdout.
