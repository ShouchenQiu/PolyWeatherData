# HrrrFill resolves non-MAJOR tail gaps with bounded extrapolation

**Status**: superseded by ADR-0016

Non-MAJOR HRRR init times only provide native forecast hours through f18.
For training decoder windows we still need f1..f48. `HrrrFill` builds that
window by aligning rows on `valid_time`:

1. Use the non-MAJOR run where it has native coverage.
2. Use the nearest prior MAJOR-hour run where valid_time overlaps.
3. For the final tail that no MAJOR run covers, linearly extrapolate from the
   last two aligned rows.

The extrapolation cap is 5 hours, which matches the worst-case HRRR off-hour
needed to reach f48: an init at +5 after a MAJOR hour needs f44..f48 filled from
extrapolation. Rows produced this way carry `is_extrapolated=True`.

Unbounded extrapolation remains rejected because it would create synthetic
forecast rows beyond the HRRR schedule. If a source has deeper missing coverage
than the normal off-hour tail, `HrrrFill` raises instead of silently producing
low-confidence decoder data.

**Consequence**: The canonical ParquetStore remains f01-only per ADR-0003. Full
decoder windows are training-only artifacts and must be fetched from a full HRRR
source or decoder cache, not reconstructed from the f01 production store.
