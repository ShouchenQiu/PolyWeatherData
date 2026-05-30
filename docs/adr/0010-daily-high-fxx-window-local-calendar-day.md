# Daily-high fxx window is the station's local calendar day (00:00-24:00 local time)

**Status**: accepted

Polymarket weather events resolve on the daily maximum temperature for a specific
calendar date at a specific station. The pipeline must select only the HRRR fxx hours
whose `valid_time` falls within that event's target date, not a fixed 48-hour horizon from
`init_time`. The chosen boundary is `[target_local_day 00:00, target_local_day 24:00)`
in the station's local timezone (from `StationConfig.timezone`), converted to UTC and
mapped to fxx offsets. `target_date` is the event's local calendar date
(`event_end_date`), not a value inferred from `init_utc.date()`.

**Considered options**:

- *Global 48h max*: take `max(fxx=1..48)` unconditionally. Rejected because it
  conflates hours from multiple calendar days (e.g. a 12:00 UTC init mixes today's
  afternoon with tomorrow's morning), producing a target that does not match the
  event's resolution criterion.

- *Fixed EST offset* (UTC-5 or UTC-4 DST): Simple arithmetic offset. Rejected because
  DST transitions would silently shift the window boundary by one hour twice a year,
  and the approach does not generalise to non-EST stations.

**Consequences**: `ModelPipeline.run()` must accept `target_date` and
`station_config: StationConfig`; it raises `ValueError` when no fxx rows fall within
the computed window for the given `init_time`. `ForecastLoop` derives
`target_date` from each Event's `event_end_date` and separately computes
`day_offset` only for the `ProbabilityStore` key. Backtest replays must pass the
event row's target date directly.
