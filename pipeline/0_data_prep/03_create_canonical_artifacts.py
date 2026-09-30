"""Step 9: create canonical windowed artifacts for all non-NYC nodes.

Every node's canonical artifact is clamped to the integrated window
[2025-12-05, 2026-08-10] and stored under the same artifact name:
    data_store/backtest/clob/<STATION>/20251205_20260810/

Tokyo/Shanghai start later (2026-03) - the clamp naturally keeps only
their in-window events. NYC already has 20251205_20260810 (step 7) and
is skipped here.

CRITICAL: the raw source artifacts are NEVER modified or deleted. The
canonical artifact is a read-only view; anything outside the window is
treated as expired, not removed from disk.
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
WINDOW = ("2025-12-05", "2026-08-10")

# city -> (station, source artifact)
SRC: dict[str, tuple[str, str]] = {
    "LONDON": ("LONDON", "20250121_20260810"),
    "CHICAGO": ("CHICAGO", "20251204_20260810"),
    "DALLAS": ("DALLAS", "20250101_20260812"),
    "MIAMI": ("MIAMI", "20250101_20260812"),
    "DENVER": ("DENVER", "20250101_20260812"),
    "SEATTLE": ("SEATTLE", "20250101_20260812"),
    "ATLANTA": ("ATLANTA", "20250101_20260812"),
    "TOKYO": ("TOKYO", "20250101_20260818"),
    "SHANGHAI": ("SHANGHAI", "20250101_20260818"),
    "SEOUL": ("SEOUL", "20250101_20260818"),
    "TORONTO": ("TORONTO", "20250101_20260818"),
}

LO, HI = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])


def main() -> int:
    for city, (station, src_art) in SRC.items():
        src = CLOB / station / src_art
        out = CLOB / station / f"{LO:%Y%m%d}_{HI:%Y%m%d}"

        ev = pd.read_parquet(src / "events.parquet")
        ev["target_date"] = pd.to_datetime(ev["target_date"])
        in_win = (ev["target_date"] >= LO) & (ev["target_date"] <= HI)
        ev_w = ev[in_win].copy()
        keep_ev = set(ev_w["event_id"])

        mk = pd.read_parquet(src / "markets.parquet")
        mk_w = mk[mk["event_id"].isin(keep_ev)].copy()

        tr = pd.read_parquet(src / "trades.parquet")
        keep_cid = set(mk_w["condition_id"])
        tr_w = tr[tr["condition_id"].isin(keep_cid)].copy()

        out.mkdir(parents=True, exist_ok=True)
        ev_w.to_parquet(out / "events.parquet", index=False)
        mk_w.to_parquet(out / "markets.parquet", index=False)
        tr_w.to_parquet(out / "trades.parquet", index=False)

        # coverage: per-day event presence, same shape as other artifacts
        cov = ev_w[["target_date", "event_id"]].copy()
        cov["n_events"] = 1
        cov.to_parquet(out / "coverage.parquet", index=False)

        manifest = {
            "data_class": "clob",
            "station_id": station,
            "requested_range": list(WINDOW),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "row_counts": {
                "events": int(len(ev_w)),
                "markets": int(len(mk_w)),
                "trades": int(len(tr_w)),
            },
            "schema_version": "windowed-canonical",
            "source_artifact": src_art,
            "note": (
                "Canonical windowed view [2025-12-05, 2026-08-10]. "
                "Raw source artifact is preserved untouched; out-of-window "
                "data is treated as expired, not deleted."
            ),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        n_skip_ev = int((~in_win).sum())
        print(
            f"[window] {city}: events={len(ev_w)} (skipped {n_skip_ev} outside "
            f"window) markets={len(mk_w)} trades={len(tr_w):,}",
            flush=True,
        )

    print(f"\n[done] wrote canonical artifacts under {WINDOW}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
