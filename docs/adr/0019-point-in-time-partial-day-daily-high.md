# Partial-day daily highs require complete point-in-time local-day coverage

**Status**: accepted

Trading forecasts for an in-progress target date combine the station observations
visible by `init_time` with non-extrapolated TFT+MC predictions for later hours.
The union must cover every hour of the station-local calendar day; a missing
right tail rejects that `(init_time, target_date)` instead of treating a partial
maximum as the Event resolution target or using tail fallback.

Hourly IEM evidence uses the representative `H-1 :51` observation aligned to the
`H:00` model slot. Therefore target date D owns aligned slots
`(D 00:00, D+1 00:00]`: the D 00:00 slot belongs to the preceding settlement day,
while D+1 00:00 contains D's final representative observation.

Future-maximum bias is calibrated by trusted forecast-hour count on development
cycles and shrunk toward the complete-day max-bias prior. Each MC daily-high draw
is the maximum of the visible observed high and the bias-corrected future-path
high, then rounded half-up in Fahrenheit before Market Bucket assignment.

**Considered alternatives**: Using only hours present in f01..f48 systematically
understates a right-censored daily maximum. Applying persisted tail values violates
ADR-0016's rule that extrapolated decoder cells do not enter trading calculations.
Reusing one complete-day max-bias scalar ignores that the maximum of 3 forecast
hours and the maximum of 24 forecast hours have different sampling bias.

**Consequences**: Forecast evidence records observed and forecast hour counts and
their time bounds. ForecastCache schema v2 and managed Backtest Studies reject old
complete-day-only caches. Production inference may adopt the same pure builder,
but the initial rollout is limited to governed replay evidence and UI inspection.
