"""Splice non-MAJOR 18h HRRR runs to 48h using the nearest MAJOR run.

Non-MAJOR init_times (03Z, 09Z, 15Z, 21Z) only produce 18h of forecast.
HrrrFill extends these to 48h by aligning on valid_time against the nearest
prior MAJOR-hour run (00Z, 06Z, 12Z, 18Z) which covers 0-48 fxx.

For valid_times in the tail that are not covered by the nearest MAJOR run
(up to 5h in the worst case), HrrrFill linearly extrapolates from the last two
aligned values.  An earlier MAJOR run ends earlier and cannot cover this tail,
so it is deliberately not consulted.  Extrapolation beyond 5h raises
RuntimeError.  Extrapolated rows are flagged with ``is_extrapolated=True``.
[ADR-0012]
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

MAJOR_HOURS = frozenset({0, 6, 12, 18})
_TARGET_FXX_COUNT = 49  # fxx 0..48 inclusive
_MAX_EXTRAP_HOURS = 5
# Metadata columns excluded from data_cols union and not copied from source rows.
_META_COLS = frozenset({"init_time", "fxx", "valid_time", "_vt", "_vt_raw", "is_extrapolated"})


def nearest_prior_major(init_time: pd.Timestamp) -> pd.Timestamp:
    """Return the most recent MAJOR-hour init prior to (or equal to) init_time."""
    init_utc = (
        init_time.tz_localize("UTC") if init_time.tzinfo is None
        else init_time.tz_convert("UTC")
    )
    hours_since_major = init_utc.hour % 6
    return init_utc - pd.Timedelta(hours=hours_since_major)


def _vt_index(df: pd.DataFrame, source_name: str = "source") -> dict:
    """Build a valid_time -> row_dict lookup from a DataFrame.

    Applies .dt.floor("s") to valid_time before indexing to absorb sub-second
    differences.  Sorts by valid_time ascending and keeps the last row when
    duplicates are present (emits WARNING with count and init_time).
    """
    if df.empty:
        return {}
    df = df.copy()
    df["_vt_raw"] = pd.to_datetime(df["valid_time"], utc=True)
    df["_vt"] = df["_vt_raw"].dt.floor("s")
    n_before = len(df)
    df = df.sort_values("_vt").drop_duplicates(subset=["_vt"], keep="last")
    n_after = len(df)
    if n_before != n_after:
        it = df["init_time"].iloc[0] if "init_time" in df.columns else "unknown"
        logger.warning(
            "HrrrFill: {} duplicate valid_times dropped in {} (init={}); kept last",
            n_before - n_after, source_name, it,
        )
    return {ts: row for ts, row in zip(df["_vt"], df.to_dict("records"))}


class HrrrFill:
    """Splice a non-MAJOR HRRR run to 48h using its nearest prior MAJOR run.

    Usage
    -----
    fill = HrrrFill()
    full_df = fill.fill(non_major_df, nearest_major_df)

    The returned DataFrame has fxx 0..48 rows for the non-major init_time,
    using:
        1. non_major_df  for any matching valid_time (highest priority)
        2. nearest_major_df for tail valid_times not in non_major_df
        3. Linear extrapolation for the final uncovered valid_times
           (capped at 5 hours; beyond that raises RuntimeError)

    All DataFrames must have a ``valid_time`` column (UTC-aware timestamps).
    The ``init_time`` and ``fxx`` columns in the output are recomputed to
    reflect the non_major init_time.  Output always includes ``is_extrapolated``.
    """

    def fill(
        self,
        non_major_df: pd.DataFrame,
        nearest_major_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill a non-MAJOR run to 48h and return the spliced DataFrame.

        Parameters
        ----------
        non_major_df:
            DataFrame for the non-MAJOR init_time (fxx 0..18).  Must have
            ``init_time`` and ``valid_time`` columns.
        nearest_major_df:
            DataFrame for the nearest prior MAJOR-hour run (fxx 0..48).

        Returns
        -------
        DataFrame with fxx 0..48 rows, reindexed to the non-major init_time.
        Includes ``is_extrapolated`` column (False for sourced rows, True for
        linearly-extrapolated rows).

        Raises
        ------
        RuntimeError
            When the uncovered tail exceeds _MAX_EXTRAP_HOURS consecutive hours.
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

        nm_idx  = _vt_index(non_major_df,    "non_major")
        maj_idx = _vt_index(nearest_major_df, "nearest_major")

        # Output column set: union of non_major and nearest_major columns,
        # excluding metadata columns that are recomputed per row.
        all_source_cols = set(non_major_df.columns) | set(nearest_major_df.columns)
        data_cols = [c for c in all_source_cols if c not in _META_COLS]

        rows: list[dict] = []

        # State for linear extrapolation
        last_aligned: dict | None = None
        prev_aligned: dict | None = None
        extrap_steps = 0

        for fxx in range(_TARGET_FXX_COUNT):
            vt = init_utc + pd.Timedelta(hours=fxx)

            nm_row = nm_idx.get(vt)
            if nm_row is not None:
                src_row: dict | None = nm_row
                is_extrap = False
                prev_aligned = last_aligned
                last_aligned = src_row
                extrap_steps = 0
            elif maj_idx.get(vt) is not None:
                src_row = maj_idx[vt]
                is_extrap = False
                prev_aligned = last_aligned
                last_aligned = src_row
                extrap_steps = 0
            else:
                # No source covers this valid_time -- linear extrapolation
                extrap_steps += 1
                if extrap_steps > _MAX_EXTRAP_HOURS:
                    raise RuntimeError(
                        f"HrrrFill: tail gap exceeds {_MAX_EXTRAP_HOURS}-hour cap "
                        f"for init={init_utc}; "
                        f"fxx={fxx} has no coverage from the nearest MAJOR run"
                    )
                src_row = None
                is_extrap = True

            row: dict = {
                "init_time": str(init_utc),
                "fxx": fxx,
                "valid_time": vt,
                "is_extrapolated": is_extrap,
            }

            if src_row is not None:
                for col in data_cols:
                    row[col] = src_row.get(col, float("nan"))
            elif last_aligned is not None:
                # Linear extrapolation: slope from last two aligned rows (per 1h step)
                for col in data_cols:
                    last_val = last_aligned.get(col, float("nan"))
                    prev_val = (
                        prev_aligned.get(col, float("nan"))
                        if prev_aligned is not None
                        else last_val
                    )
                    try:
                        slope = float(last_val) - float(prev_val)
                        row[col] = float(last_val) + slope * extrap_steps
                    except (TypeError, ValueError):
                        row[col] = float("nan")
            else:
                for col in data_cols:
                    row[col] = float("nan")

            rows.append(row)

        return pd.DataFrame(rows)
