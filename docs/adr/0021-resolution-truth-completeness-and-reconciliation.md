# Resolution truth requires complete local days and resolved-outcome reconciliation

**Status**: accepted

Weather.com daily-high truth is eligible for immutable backtest artifacts only when
the cache contains every hourly slot in the station-local calendar day. The expected
count is derived from local midnight boundaries, so spring-forward days require 23
hours, ordinary days 24, and fall-back days 25. Missing `valid_time` or any missing
hour makes that target date ineligible; the daily maximum must not be computed from a
partial response. Because site-based stations do not guarantee one report per hour,
an immutable truth artifact partitions its complete requested range into eligible
actual-high rows and hash-bound **truth exclusions**. A truth exclusion records the
station, target date, observed and expected hours, and reason. It is removed from all
formal analysis before reconciliation rather than silently accepted or imputed.

The hourly representative observation is an alignment aid for model and training use;
it is not the Event resolution maximum. Each eligible cache day also retains the
maximum across every raw Weather.com observation in that station-local day. Immutable
actual-high artifacts use this `weather_com_raw_daily_high_f` value and declare
`weather_com_raw_observation_max` as their daily-high source.

Weather.com requests span the requested date through the following date and are then
filtered back to the requested station-local day. This avoids the provider returning
only the pre-transition portion of a spring-forward date. Prefetch treats incomplete
caches as pending and verifies completeness again after retrieval. If retrieval still
returns an incomplete day, the source cache remains unchanged and the artifact builder
records that date as a truth exclusion. Missing cache files, malformed schemas,
overlapping eligible/excluded dates, or a requested date in neither partition remain
hard failures.

Historical Gamma market normalization retains YES/NO terminal outcome prices and a
nullable `resolved_yes` value. When these fields are available, the CLOB semantic audit
must reconcile the bucket selected by `actual_high_f` with the unique resolved YES
Market. Missing resolution fields are reported as `NOT_MEASURED`; disagreement or a
partial/multiple resolved winner is a hard audit failure.

**Consequences**: actual-high artifacts use schema version 2 and declare the
`complete_station_local_day_or_explicit_exclusion` coverage contract. Their manifest
hashes both the actual-high table and truth-exclusion table and proves that the two
tables exactly partition the requested dates. New historical CLOB artifacts use schema
version 2; the loader remains backward-compatible with schema version 1, whose
resolved-outcome reconciliation may be unavailable. Existing immutable artifacts are
not silently overwritten and must be rebuilt after this contract changes.
