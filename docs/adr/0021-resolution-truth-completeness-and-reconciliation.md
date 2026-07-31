# Resolution truth requires complete local days and resolved-outcome reconciliation

**Status**: accepted

Weather.com daily-high truth is eligible for immutable backtest artifacts only when
the cache contains every hourly slot in the station-local calendar day. The expected
count is derived from local midnight boundaries, so spring-forward days require 23
hours, ordinary days 24, and fall-back days 25. Missing `valid_time` or any missing
hour fails closed; the daily maximum must not be computed from a partial response.

Weather.com requests span the requested date through the following date and are then
filtered back to the requested station-local day. This avoids the provider returning
only the pre-transition portion of a spring-forward date. Prefetch treats incomplete
caches as pending and verifies completeness again after retrieval.

Historical Gamma market normalization retains YES/NO terminal outcome prices and a
nullable `resolved_yes` value. When these fields are available, the CLOB semantic audit
must reconcile the bucket selected by `actual_high_f` with the unique resolved YES
Market. Missing resolution fields are reported as `NOT_MEASURED`; disagreement or a
partial/multiple resolved winner is a hard audit failure.

**Consequences**: actual-high artifacts use schema version 2 and declare the
`complete_station_local_day` coverage contract. New historical CLOB artifacts use
schema version 2; the loader remains backward-compatible with schema version 1, whose
resolved-outcome reconciliation may be unavailable. Existing immutable artifacts are
not silently overwritten and must be rebuilt after bad truth caches are repaired.
