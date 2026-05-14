import pandas as pd
import pytest

from data.storage.parquet_store import ParquetStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hrrr(init_times: list, fxx_range: range) -> pd.DataFrame:
    rows = []
    for it in init_times:
        for fxx in fxx_range:
            rows.append({
                "init_time": pd.Timestamp(it, tz="UTC"),
                "fxx": fxx,
                "valid_time": pd.Timestamp(it, tz="UTC") + pd.Timedelta(hours=fxx),
                "TMP_2m": float(280 + fxx),
            })
    return pd.DataFrame(rows)


def _make_obs(valid_times: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"valid_time": pd.Timestamp(t, tz="UTC"), "raw_tmpc_tenths": float(i)}
        for i, t in enumerate(valid_times)
    ])


# ---------------------------------------------------------------------------
# ParquetStore -- HRRR dedup
# ---------------------------------------------------------------------------


def test_append_returns_row_count(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-01 00:00"], range(2))   # fxx 0,1 -- only fxx=1 stored (ADR-0003)
    n = store.append("KLGA", df)
    assert n == 1


def test_append_dedup_no_double_write(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-01 00:00"], range(2))   # fxx 0,1 -- fxx=1 written once
    store.append("KLGA", df)
    n2 = store.append("KLGA", df)  # identical data -> 0 new rows
    assert n2 == 0


def test_append_partial_overlap(tmp_dir):
    """Obs (no fxx col) dedup by valid_time: second write of new rows only."""
    store = ParquetStore(tmp_dir / "store")
    obs1 = _make_obs(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"])
    obs2 = _make_obs(["2026-01-01 01:00", "2026-01-01 03:00", "2026-01-01 04:00"])
    store.append("KLGA_obs", obs1)
    n = store.append("KLGA_obs", obs2)
    assert n == 2   # only 03:00 and 04:00 are new


# ---------------------------------------------------------------------------
# ParquetStore -- quarter partitioning
# ---------------------------------------------------------------------------


def test_quarter_key():
    store = ParquetStore("/tmp/_pw_test_q")
    assert store._quarter_key(pd.Timestamp("2026-01-15")) == "2026-Q1"
    assert store._quarter_key(pd.Timestamp("2026-04-01")) == "2026-Q2"
    assert store._quarter_key(pd.Timestamp("2026-07-31")) == "2026-Q3"
    assert store._quarter_key(pd.Timestamp("2026-12-01")) == "2026-Q4"


def test_append_creates_quarter_file(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-02-01 00:00"], range(1, 2))   # fxx=1
    store.append("KLGA", df)
    assert (tmp_dir / "store" / "KLGA" / "2026-Q1.parquet").exists()


def test_append_across_quarter_boundary(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    # Q1 inittime
    df1 = _make_hrrr(["2026-03-31 23:00"], range(2))
    # Q2 inittime
    df2 = _make_hrrr(["2026-04-01 00:00"], range(2))
    store.append("KLGA", df1)
    store.append("KLGA", df2)
    assert (tmp_dir / "store" / "KLGA" / "2026-Q1.parquet").exists()
    assert (tmp_dir / "store" / "KLGA" / "2026-Q2.parquet").exists()


# ---------------------------------------------------------------------------
# ParquetStore -- read with date range filter
# ---------------------------------------------------------------------------


def test_read_returns_filtered_rows(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-10 00:00", "2026-01-20 00:00"], range(2))   # fxx 0,1
    store.append("KLGA", df)

    start = pd.Timestamp("2026-01-10 00:00", tz="UTC")
    end = pd.Timestamp("2026-01-10 23:59", tz="UTC")
    result = store.read("KLGA", start, end)
    # Only rows from init_time 2026-01-10 within the window
    assert len(result) > 0
    for ts in pd.to_datetime(result["init_time"], utc=True):
        assert ts >= start


def test_read_empty_when_no_data(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    result = store.read("KLGA",
                        pd.Timestamp("2026-01-01", tz="UTC"),
                        pd.Timestamp("2026-01-31", tz="UTC"))
    assert result.empty


# ---------------------------------------------------------------------------
# ParquetStore -- obs (valid_time key)
# ---------------------------------------------------------------------------


def test_obs_dedup_by_valid_time(tmp_dir):
    store = ParquetStore(tmp_dir / "store")
    obs = _make_obs(["2026-01-01 00:00", "2026-01-01 01:00"])
    store.append("KLGA_obs", obs)
    n2 = store.append("KLGA_obs", obs)
    assert n2 == 0


# ---------------------------------------------------------------------------
# ParquetStore -- ADR-0003: fxx>1 dropped on write
# ---------------------------------------------------------------------------


def test_append_drops_fxx_gt_1(tmp_dir):
    """Only fxx=1 is stored (ADR-0003 f01-only storage)."""
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-01 00:00"], range(0, 5))   # fxx 0,1,2,3,4
    n = store.append("KLGA", df)
    assert n == 1   # only fxx 1 kept


def test_append_drops_fxx_gt_1_stored_correctly(tmp_dir):
    """Only fxx=1 rows are present in Parquet files after append."""
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-01 00:00"], range(0, 5))
    store.append("KLGA", df)
    stored = store.read(
        "KLGA",
        pd.Timestamp("2025-12-31", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
    )
    assert not stored.empty
    assert (stored["fxx"] == 1).all()


def test_append_all_fxx_gt_1_returns_zero(tmp_dir):
    """append() returns 0 when every row has fxx > 1."""
    store = ParquetStore(tmp_dir / "store")
    df = _make_hrrr(["2026-01-01 00:00"], range(2, 6))   # fxx 2,3,4,5 only
    n = store.append("KLGA", df)
    assert n == 0
