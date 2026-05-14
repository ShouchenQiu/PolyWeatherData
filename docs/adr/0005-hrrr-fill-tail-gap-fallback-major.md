# HrrrFill resolves tail-gap valid_times using the prior-prior MAJOR-hour run

When a non-MAJOR init_time (e.g. 03z) is filled to 48h, the nearest prior MAJOR-hour run
(00z) covers valid_times up to 00z+48h = 48z, leaving up to 6 valid_times uncovered at
the tail (e.g. 49z¨C51z for 03z init_time). HrrrFill resolves these by falling back to
the MAJOR-hour before the nearest one (e.g. previous-day 18z), which has a longer
valid_time reach. A WARNING is logged whenever this fallback is used.

**Considered option (A ¡ª forward-fill)**: Repeat the last available value across the gap.
Rejected because it introduces a flat plateau in the decoder window, which the TFT can
learn to associate with the fill boundary rather than a real forecast signal.

**Considered option (C ¡ª NaN + mask column)**: Mark gap positions as NaN and pass a
`fill_mask` boolean column so the model can learn to down-weight those time steps.
Rejected because it complicates the DataLoader schema and requires retraining to interpret
the mask; the fallback MAJOR approach keeps the column schema clean.

**Consequence**: HrrrFill requires access to at least two prior MAJOR-hour full runs
(fxx 0..48) to handle the worst-case 6-hour tail gap.  These runs are fetched directly
from the network by `BaseGribSource.fetch()` during the non-MAJOR splice; they are
**not** read from the f01-only ParquetStore (see ADR-0003).  Implementations must not
rely on the store as a source of full MAJOR-run data for HrrrFill.
