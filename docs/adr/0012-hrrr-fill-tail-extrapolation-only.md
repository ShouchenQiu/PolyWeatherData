# HrrrFill uses nearest-major overlap, then bounded extrapolation

**Status**: superseded by ADR-0016

ADR-0005 defines the intended non-MAJOR HRRR decoder-window behavior:
native non-MAJOR coverage first, nearest prior MAJOR overlap second, and bounded
linear extrapolation for the remaining tail.

The prior-prior MAJOR path is removed. For a non-MAJOR init, the nearest
prior MAJOR run already has the latest available 48-hour forecast window that is
valid at that init time. If the nearest MAJOR run cannot cover the final tail
valid_times, an even earlier MAJOR run ends earlier and cannot cover those later
valid_times either. Therefore the prior-prior fetch is redundant for normal HRRR
time geometry and adds network work plus confusing failure modes.

The implemented fill order is:

1. Use the non-MAJOR run for native fxx 0..18 coverage.
2. Use the nearest prior MAJOR run for overlapping tail valid_times.
3. Linearly extrapolate the final uncovered 1..5 hours, setting
   `is_extrapolated=True`.

The code must not fetch the following MAJOR run to fill the tail. That would use
future model information that was not available at the non-MAJOR init time.

## Operational evidence

The 2024-10 through 2026-05 historical download showed that prewarming and
fetching `nearest_major_init - 6h` added remote-store requests, warm failures,
and misleading "tail-gap fallback unavailable" warnings. It did not improve
normal decoder coverage: a prior-prior 48-hour run ends six hours before the
nearest 48-hour run.

After implementation, the existing primary decoder cache was checked across
20 monthly Parquet files: 14,592 init windows and 700,416 rows. For every init,
the number of `is_extrapolated=True` rows equalled `init_hour % 6`; there were
zero geometry mismatches. Tail lengths 0 through 5 each occurred for 2,432
inits, confirming the expected HRRR schedule rather than prior-prior fill.

When an off-hour Zarr forecast store is missing, the hybrid route remains:
S3/GRIB supplies that init's native fxx 0..18, Zarr supplies the nearest MAJOR
overlap, and `HrrrFill` extrapolates only the final bounded tail. If the nearest
MAJOR dependency itself is unavailable, the caller may use its full-window
fallback source; `HrrrFill` does not silently synthesize more than five hours.

## Consequences

- `HrrrFill.fill()` accepts only `non_major_df` and `nearest_major_df`.
- `HrrrZarrSource` and `BaseGribSource` do not fetch
  `nearest_major_init - 6h`.
- Decoder prewarming schedules only the nearest MAJOR dependency for an
  off-hour init.
- Existing decoder-cache rows do not require regeneration solely for this ADR:
  under normal HRRR 48-hour geometry, the removed run could not populate the
  later tail. Existing completeness and NaN audits remain authoritative.
- The following MAJOR run remains forbidden because it introduces future
  information unavailable at the off-hour init.
