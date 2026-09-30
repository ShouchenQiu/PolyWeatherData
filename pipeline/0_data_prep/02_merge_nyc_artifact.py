"""Step 7: merge NYC (KLGA) artifacts into one aligned artifact.

Sources:
  - 20250401_20260331_resolution_v2 (take target_date >= 2025-12-05)
  - 20260401_20260630             (all; backfill resolved_yes from klga_q2_resolution.parquet)
  - 20260701_20260818             (take target_date <= 2026-08-10)

Output: data_store/backtest/clob/KLGA/20251205_20260810/
  events.parquet, markets.parquet, trades.parquet, coverage.parquet, manifest.json

The merged artifact becomes the single NYC node: events covering
[2025-12-05, 2026-08-10]. Raw source artifacts are left untouched.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLOB_ROOT, OUT_DIR, ensure_out_dir  # noqa: E402
CLOB = CLOB_ROOT
KLGA = CLOB / "KLGA"
OUT = KLGA / "20251205_20260810"
RES = OUT_DIR / "klga_q2_resolution.parquet"
START = "2025-12-05"
END = "2026-08-10"

SRC = [
    ("20250401_20260331_resolution_v2", START, None),   # (artifact, lo, hi)
    ("20260401_20260630", None, None),
    ("20260701_20260818", None, END),
]


def filter_events(art: str, lo: str | None, hi: str | None) -> pd.DataFrame:
    ev = pd.read_parquet(KLGA / art / "events.parquet")
    ev["target_date"] = pd.to_datetime(ev["target_date"])
    if lo:
        ev = ev[ev["target_date"] >= pd.to_datetime(lo)]
    if hi:
        ev = ev[ev["target_date"] <= pd.to_datetime(hi)]
    return ev


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    ev_frames, mk_frames, tr_frames = [], [], []
    for art, lo, hi in SRC:
        ev = filter_events(art, lo, hi)
        keep_ev = set(ev["event_id"])
        mk = pd.read_parquet(KLGA / art / "markets.parquet")
        mk = mk[mk["event_id"].isin(keep_ev)]

        if art == "20260401_20260630":
            # Q2 artifact lacks resolution columns; backfill them.
            res = pd.read_parquet(RES)
            mk = mk.merge(
                res[["condition_id", "resolved_yes",
                     "yes_outcome_price", "no_outcome_price"]],
                on="condition_id", how="left",
            )
        tr = pd.read_parquet(KLGA / art / "trades.parquet")
        tr = tr[tr["condition_id"].isin(set(mk["condition_id"]))]

        ev_frames.append(ev)
        mk_frames.append(mk)
        tr_frames.append(tr)
        print(
            f"[merge] {art}: events={len(ev)} conditions={len(mk)} trades={len(tr):,}",
            flush=True,
        )

    events = pd.concat(ev_frames, ignore_index=True)
    markets = pd.concat(mk_frames, ignore_index=True)
    trades = pd.concat(tr_frames, ignore_index=True)

    # sanity: no duplicate ids across sources
    assert events["event_id"].is_unique, "duplicate event_id"
    assert markets["condition_id"].is_unique, "duplicate condition_id"
    assert markets["resolved_yes"].notna().all(), "unresolved conditions remain"

    events.to_parquet(OUT / "events.parquet", index=False)
    markets.to_parquet(OUT / "markets.parquet", index=False)
    trades.to_parquet(OUT / "trades.parquet", index=False)

    # coverage: per-day event presence (reuse the same shape as other artifacts)
    cov = events[["target_date", "event_id"]].copy()
    cov["n_events"] = 1
    cov.to_parquet(OUT / "coverage.parquet", index=False)

    manifest = {
        "data_class": "clob",
        "station_id": "KLGA",
        "requested_range": [START, END],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "events": int(len(events)),
            "markets": int(len(markets)),
            "trades": int(len(trades)),
        },
        "schema_version": "merged-from-3-sources",
        "sources": [a for a, _, _ in SRC],
        "note": (
            "Merged NYC artifact: v2 (>=2025-12-05) + Q2 (resolved backfilled) "
            "+ 7-01 from Q3. Raw sources untouched."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"\n[done] wrote {OUT}: events={len(events)} conditions={len(markets)} "
        f"trades={len(trades):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
