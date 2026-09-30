"""Step 1 (fixed): backfill resolution info for NYC Q2 (2026-04-01..2026-06-30).

The KLGA 20260401_20260630 artifact was downloaded with an older schema
(26 columns, no resolved_yes). This script fetches official settlement
outcomePrices via the Gamma public-search endpoint, querying by EVENT SLUG
(one request per event, ~88 events total, much faster than per-condition).

Output: docs/icdm/output/klga_q2_resolution.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
PROXY = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLOB_ROOT, OUT_DIR, ensure_out_dir  # noqa: E402

ARTIFACT = CLOB_ROOT / "KLGA" / "20260401_20260630"
OUT_FILE = OUT_DIR / "klga_q2_resolution.parquet"
MIN_INTERVAL_S = 1.0


def fetch_event_prices(event_slug: str) -> dict[str, tuple[float, float]]:
    """Return {condition_id: (yes_price, no_price)} for one event."""
    for attempt in range(4):
        try:
            r = requests.get(
                f"{GAMMA}/public-search",
                params={
                    "q": event_slug,
                    "events_status": "closed",
                    "keep_closed_markets": 1,
                    "limit_per_type": 5,
                },
                proxies=PROXY,
                timeout=25,
            )
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            events = r.json().get("events", [])
            # find the event whose slug matches exactly
            for ev in events:
                if str(ev.get("slug", "")) == event_slug:
                    out: dict[str, tuple[float, float]] = {}
                    for m in ev.get("markets", []) or []:
                        cid = str(m.get("conditionId") or "")
                        op = m.get("outcomePrices")
                        if not cid or not op:
                            continue
                        if isinstance(op, str):
                            op = json.loads(op)
                        vals = [float(x) for x in op]
                        if len(vals) >= 2:
                            out[cid] = (vals[0], vals[1])
                    return out
            return {}
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return {}
    """Fetch outcomePrices for one market from Gamma API."""
    for attempt in range(4):
        try:
            r = requests.get(
                f"{GAMMA}/markets/{condition_id}",
                proxies=PROXY,
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                op = data.get("outcomePrices")
                return op if isinstance(op, list) else None
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def main() -> int:
    mk = pd.read_parquet(
        ARTIFACT / "markets.parquet",
        columns=["condition_id", "event_id"],
    )
    ev = pd.read_parquet(
        ARTIFACT / "events.parquet",
        columns=["event_id", "event_slug"],
    )
    ev_slugs = dict(zip(ev["event_id"], ev["event_slug"]))

    print(
        f"[start] backfill {len(ev_slugs)} events ({len(mk)} conditions)",
        flush=True,
    )

    cond_resolution: dict[str, dict] = {}
    for i, (eid, slug) in enumerate(ev_slugs.items(), start=1):
        prices = fetch_event_prices(slug)
        for cid, (yp, np_) in prices.items():
            if yp == 1.0 and np_ == 0.0:
                resolved: bool | None = True
            elif yp == 0.0 and np_ == 1.0:
                resolved = False
            else:
                resolved = None
            cond_resolution[cid] = {
                "condition_id": cid,
                "resolved_yes": resolved,
                "yes_outcome_price": yp,
                "no_outcome_price": np_,
            }
        if i % 20 == 0 or i == len(ev_slugs):
            print(
                f"[progress] {i}/{len(ev_slugs)} "
                f"resolved_conds={len(cond_resolution)}",
                flush=True,
            )
        time.sleep(MIN_INTERVAL_S)

    # merge with all conditions (fill missing as null)
    rows = []
    for cid in mk["condition_id"]:
        if cid in cond_resolution:
            rows.append(cond_resolution[cid])
        else:
            rows.append(
                {
                    "condition_id": cid,
                    "resolved_yes": None,
                    "yes_outcome_price": None,
                    "no_outcome_price": None,
                }
            )
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_FILE, index=False)

    n_true = int((out["resolved_yes"] == True).sum())  # noqa: E712
    n_false = int((out["resolved_yes"] == False).sum())  # noqa: E712
    n_null = int(out["resolved_yes"].isna().sum())
    print(
        f"[complete] resolved True={n_true} False={n_false} null={n_null}",
        flush=True,
    )
    print(f"[done] wrote {OUT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
