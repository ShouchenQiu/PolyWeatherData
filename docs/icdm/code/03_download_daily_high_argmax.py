"""Download daily max-temperature argmax time for all 12 nodes.

US stations: IEM ASOS minute-level data, fetched in MONTHLY chunks (a full
multi-month request times out).  International stations: Open-Meteo hourly.

Output: docs/icdm/output/daily_high_argmax.parquet
  columns: station, target_date, high_value, argmax_utc, argmax_hour_local
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
OUT_FILE = OUT_DIR / "daily_high_argmax.parquet"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OM_URL = "https://archive-api.open-meteo.com/v1/archive"
PROXY = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
END_DATE = "2026-08-18"

# station -> (iem_id_or_None, lat, lon, tz, series start date)
STATIONS = {
    "KLGA": ("LGA", 40.7769, -73.8740, "America/New_York", "2025-04-01"),
    "KORD": ("ORD", 41.9742, -87.9073, "America/Chicago", "2025-12-04"),
    "KDFW": ("DFW", 32.8998, -97.0403, "America/Chicago", "2025-12-04"),
    "KMIA": ("MIA", 25.7959, -80.2870, "America/New_York", "2025-12-04"),
    "KDEN": ("DEN", 39.8561, -104.6737, "America/Denver", "2025-12-04"),
    "KSEA": ("SEA", 47.4490, -122.3090, "America/Los_Angeles", "2025-12-06"),
    "KATL": ("ATL", 33.6407, -84.4277, "America/New_York", "2025-12-06"),
    "LONDON": (None, 51.5074, -0.1278, "Europe/London", "2025-01-22"),
    "TOKYO": (None, 35.6762, 139.6503, "Asia/Tokyo", "2026-03-11"),
    "SHANGHAI": (None, 31.2304, 121.4737, "Asia/Shanghai", "2026-03-13"),
    "SEOUL": (None, 37.5665, 126.9780, "Asia/Seoul", "2025-12-06"),
    "TORONTO": (None, 43.6532, -79.3832, "America/Toronto", "2025-12-06"),
}


def months_between(start: str, end: str) -> list[tuple[str, str]]:
    """Return [(first_day, last_day)] for each month in [start, end].

    end is capped at min(end, today-5d) because the Open-Meteo archive
    only publishes data up to ~5 days before today; requesting future
    dates returns HTTP 400.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None)
    e = min(e, today - pd.Timedelta(days=5))
    out = []
    cur = s.replace(day=1)
    while cur <= e:
        last = (cur + pd.offsets.MonthEnd(0)).replace(hour=0, minute=0, second=0)
        last = min(last, e)
        out.append((cur.date().isoformat(), last.date().isoformat()))
        cur = (cur + pd.offsets.MonthBegin(1))
    return out


def fetch_iem_month(station_id: str, d1: str, d2: str) -> pd.DataFrame:
    s = pd.Timestamp(d1)
    e = pd.Timestamp(d2) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    params = {
        "station": station_id, "data": "tmpf", "tz": "Etc/UTC",
        "format": "comma", "latlon": "no", "missing": "M",
        "year1": str(s.year), "month1": str(s.month), "day1": str(s.day),
        "hour1": "0", "minute1": "0",
        "year2": str(e.year), "month2": str(e.month), "day2": str(e.day),
        "hour2": str(e.hour), "minute2": str(e.minute),
    }
    r = requests.get(IEM_URL, params=params, timeout=90)
    r.raise_for_status()
    lines = [l for l in r.text.strip().splitlines() if not l.startswith("#")]
    if not lines:
        return pd.DataFrame(columns=["valid", "tmpf"])
    return pd.read_csv(io.StringIO("\n".join(lines)))


def fetch_openmeteo(lat: float, lon: float, d1: str, d2: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(
                OM_URL,
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": d1, "end_date": d2,
                    "hourly": "temperature_2m", "timezone": "UTC",
                },
                timeout=90,
            )
            if r.status_code == 200:
                data = r.json()
                hours = pd.to_datetime(data["hourly"]["time"], utc=True)
                return pd.DataFrame(
                    {"valid": hours, "tmpc": data["hourly"]["temperature_2m"]}
                )
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(3 * (attempt + 1))
    raise last_exc or RuntimeError("open-meteo request failed")


def daily_argmax(df: pd.DataFrame, temp_col: str, tz: str) -> pd.DataFrame:
    d = df.copy()
    d["valid"] = pd.to_datetime(d["valid"], utc=True)
    d["tmp"] = pd.to_numeric(d[temp_col], errors="coerce")
    d = d.dropna(subset=["tmp"])
    # Group by LOCAL calendar date (Polymarket target_date is a local date).
    # Using UTC dates shifts the window by the UTC offset, so the previous
    # evening's temperature can be assigned to the next day's high.
    d["local"] = d["valid"].dt.tz_convert(tz)
    d["date"] = d["local"].dt.date
    rows = []
    for day, g in d.groupby("date"):
        idx = g["tmp"].idxmax()
        argmax_utc = g.loc[idx, "valid"]
        rows.append({
            "target_date": str(day),
            "high_value": float(g.loc[idx, "tmp"]),
            "argmax_utc": argmax_utc,
            "argmax_hour_local": argmax_utc.tz_convert(tz).hour,
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    for station, (iem_id, lat, lon, tz, start) in STATIONS.items():
        frames = []
        try:
            months = months_between(start, END_DATE)
            for i, (d1, d2) in enumerate(months, start=1):
                if iem_id is not None:
                    df = fetch_iem_month(iem_id, d1, d2)
                    temp_col = "tmpf"
                else:
                    df = fetch_openmeteo(lat, lon, d1, d2)
                    temp_col = "tmpc"
                frames.append(df)
                time.sleep(0.8)
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            arg = daily_argmax(raw, temp_col, tz)
            arg["station"] = station
            arg["unit"] = "F" if iem_id is not None else "C"
            all_rows.append(arg)
            print(
                f"[complete] {station}: {len(arg)} days "
                f"({len(months)} months)",
                flush=True,
            )
        except Exception as exc:
            print(f"[error] {station}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1.0)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_parquet(OUT_FILE, index=False)
    print(f"[done] wrote {OUT_FILE} ({len(out)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
