"""Step 6: aligned-dates panel (kept for the experiment page).

The main step-5 panel now uses the integrated window (every node from
2025-12-05, cutoff 2026-08-10). This script reuses step 5 with the same
START_DATE so the aligned panel matches the canonical panel; it remains
for backward compatibility with the Aligned-Dates UI page.

Output: docs/icdm/output/first_layer_panel_argmax_30min_aligned.parquet
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

CODE_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\code")
OUT_DIR = Path(r"d:\NN\PolyWeatherData\docs\icdm\output")
START_DATE = "2025-12-05"

spec = importlib.util.spec_from_file_location(
    "step05", CODE_DIR / "05_rebuild_panel_argmax_30min.py"
)
step05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step05)
step05.START_DATE = START_DATE  # aligned window for every node


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    argmax = pd.read_parquet(step05.ARGMAX_FILE)
    frames = [step05.load_node(node, argmax) for node in step05.NODES]
    panel = pd.concat(frames, ignore_index=True)

    winners = panel[panel["y"] == 1].groupby("event_id").size()
    bad = int((winners != 1).sum())
    print(
        f"\n[sanity] events={panel['event_id'].nunique()} "
        f"conditions={len(panel)} bad_winner={bad}",
        flush=True,
    )

    out = OUT_DIR / "first_layer_panel_argmax_30min_aligned.parquet"
    panel.to_parquet(out, index=False)

    rep = [
        f"Panel aligned to target_date >= {START_DATE}",
        "=" * 60,
    ]
    for node, sub in panel.groupby("city"):
        rep.append(f"\n{node}: events={sub['event_id'].nunique()} conditions={len(sub)}")
    report = "\n".join(rep)
    (OUT_DIR / "panel_argmax_30min_aligned_quality_report.txt").write_text(
        report, encoding="utf-8"
    )
    print("\n" + report, flush=True)
    print(f"\n[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
