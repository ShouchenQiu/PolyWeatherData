# HrrrFill uses nearest-major overlap, then bounded extrapolation

**Status**: accepted, implementation deferred

ADR-0005 defines the intended non-MAJOR HRRR decoder-window behavior:
native non-MAJOR coverage first, nearest prior MAJOR overlap second, and bounded
linear extrapolation for the remaining tail. Current code still carries an older
`prior_major_df` path that attempts to fetch the MAJOR run before the nearest
prior MAJOR run.

That prior-prior MAJOR path should be removed. For a non-MAJOR init, the nearest
prior MAJOR run already has the latest available 48-hour forecast window that is
valid at that init time. If the nearest MAJOR run cannot cover the final tail
valid_times, an even earlier MAJOR run ends earlier and cannot cover those later
valid_times either. Therefore the prior-prior fetch is redundant for normal HRRR
time geometry and adds network work plus confusing failure modes.

The target fill order is:

1. Use the non-MAJOR run for native fxx 0..18 coverage.
2. Use the nearest prior MAJOR run for overlapping tail valid_times.
3. Linearly extrapolate the final uncovered 1..5 hours, setting
   `is_extrapolated=True`.

The code must not fetch the following MAJOR run to fill the tail. That would use
future model information that was not available at the non-MAJOR init time.

**Consequences**: `HrrrFill.fill()` should no longer accept or consult
`prior_major_df`. `HrrrZarrSource` and `BaseGribSource` should stop fetching
`nearest_major_init - 6h` for tail-gap handling. Tests that assert prior-major
tail data is used should be removed or rewritten to assert bounded extrapolation.
Existing already-downloaded decoder cache should be audited before deciding
whether it needs regeneration; this ADR records the desired behavior for the
post-download cleanup slice.
