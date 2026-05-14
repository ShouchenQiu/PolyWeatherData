"""HrrrFill: splice non-MAJOR 18h HRRR runs to 48h using a prior MAJOR-hour run.

Non-MAJOR init_times (03Z, 09Z, 15Z, 21Z) only produce 18h of forecast.
HrrrFill extends these to 48h by aligning on valid_time against the nearest
prior MAJOR-hour run (00Z, 06Z, 12Z, 18Z) which covers 0-48 fxx.

For valid_times in the tail that are not covered by the nearest MAJOR run
(up to a 6h gap in the worst case), HrrrFill falls back to the MAJOR-hour
before that and logs a WARNING.  [ADR-0005]
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

_MAJOR_HOURS = frozenset({0, 6, 12, 18})
_TARGET_FXX_COUNT = 49  # fxx 0..48 inclusive


def _nearest_prior_major(init_time: pd.Timestamp) -> pd.Timestamp:
    """Return the most recent MAJOR-hour init prior to (or equal to) init_time."""
    init_utc = (
        init_time.tz_localize("UTC") if init_time.tzinfo is None
        else init_time.tz_convert("UTC")
    )
    hours_since_major = init_utc.hour % 6
    return init_utc - pd.Timedelta(hours=hours_since_major)


class HrrrFill:
    """Splice a non-MAJOR HRRR run to 48h using prior MAJOR-hour runs.

    Usage
    -----
    fill = HrrrFill()
    full_df = fill.fill(non_major_df, nearest_major_df, prior_major_df)

    The returned DataFrame has fxx 0..48 rows for the non-major init_time,
    using:
        1. non_major_df  for any matching valid_time (highest priority)
        2. nearest_major_df for tail valid_times not in non_major_df
        3. prior_major_df for tail valid_times not in nearest_major_df
           (logs WARNING when this fallback is used)

    All DataFrames must have a ``valid_time`` column (UTC-aware timestamps).
    The ``init_time`` and ``fxx`` columns in the output are recomputed to
    reflect the non_major init_time.
    """

    def fill(
        self,
        non_major_df: pd.DataFrame,
        nearest_major_df: pd.DataFrame,
        prior_major_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fill a non-MAJOR run to 48h and return the spliced DataFrame.

        Parameters
        ----------
        non_major_df:
            DataFrame for the non-MAJOR init_time (fxx 0..18).  Must have
            ``init_time`` and ``valid_time`` columns.
        nearest_major_df:
            DataFrame for the nearest prior MAJOR-hour run (fxx 0..48).
        prior_major_df:
            DataFrame for the MAJOR-hour before the nearest one.  Used when
            nearest_major_df does not cover all tail valid_times.

        Returns
        -------
        DataFrame with fxx 0..48 rows, reindexed to the non-major init_time.
        """
        if non_major_df.empty:
            return non_major_df

        # Normalise init_time from the first row
        raw_init = non_major_df["init_time"].iloc[0]
        init_utc: pd.Timestamp = (
            pd.Timestamp(raw_init, tz="UTC")
            if not isinstance(raw_init, pd.Timestamp)
            else raw_init
        )
        if init_utc.tzinfo is None:
            init_utc = init_utc.tz_localize("UTC")
        else:
            init_utc = init_utc.tz_convert("UTC")

        # Build lookup dicts keyed by valid_time (rounded to second)
        def _vt_index(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            df = df.copy()
            df["_vt"] = pd.to_datetime(df["valid_time"], utc=True)
            return {ts: row for ts, row in zip(df["_vt"], df.to_dict("records"))}

        nm_idx  = _vt_index(non_major_df)
        maj_idx = _vt_index(nearest_major_df)
        pri_idx = _vt_index(prior_major_df) if prior_major_df is not None and not prior_major_df.empty else {}

        data_cols = [
            c for c in non_major_df.columns
            if c not in ("init_time", "fxx", "valid_time", "_vt")
        ]

        rows: list[dict] = []
        used_prior = False

        for fxx in range(_TARGET_FXX_COUNT):
            vt = init_utc + pd.Timedelta(hours=fxx)

            # Explicit priority: non_major > nearest_major > prior_major.
            # used_prior is set when prior_major is the actual source for a row
            # so that the WARNING fires if and only if prior data was used.
            nm_row  = nm_idx.get(vt)
            if nm_row is not None:
                src_row: dict | None = nm_row
            elif maj_idx.get(vt) is not None:
                src_row = maj_idx[vt]
            elif pri_idx.get(vt) is not None:
                src_row = pri_idx[vt]
                used_prior = True
            else:
                src_row = None

            row: dict = {"init_time": str(init_utc), "fxx": fxx, "valid_time": vt}
            for col in data_cols:
                row[col] = src_row.get(col, float("nan")) if src_row else float("nan")
            rows.append(row)

        if used_prior:
            logger.warning(
                "HrrrFill: tail gap for init={} filled using prior MAJOR-hour run",
                init_utc,
            )

        return pd.DataFrame(rows)
