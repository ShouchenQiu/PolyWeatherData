"""Build the chronological train / validation / test split for the study.

The paper (Section III-B, "Event Structure and Time Split") splits the sample
by event date so that model choices are completed before the final
time-out-of-sample evaluation:

    train      2025-12-05 .. 2026-05-31
    validation 2026-06-01 .. 2026-06-30
    test       2026-07-01 .. 2026-07-31

The canonical panel (`first_layer_panel_argmax_30min.parquet`) is keyed by
``event_id`` and does not carry the target date, so this step maps every
event to its target date and writes the assignment that later steps consume.

Output:
    docs/icdm/output/train_val_test_split.parquet
        columns: event_id, target_date, split
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLOB_ROOT, OUT_DIR, PANEL, ensure_out_dir  # noqa: E402
from constants import ARTIFACT, NODES, TRAIN_END, VALID_END  # noqa: E402

CLOB = CLOB_ROOT
TEST_END = "2026-07-31"


def load_event_dates() -> pd.DataFrame:
    """Collect (event_id, target_date) for every canonical node."""
    frames = []
    for node, station in NODES.items():
        ev = pd.read_parquet(
            CLOB / station / ARTIFACT / "events.parquet",
            columns=["event_id", "target_date"],
        )
        ev["city"] = node
        frames.append(ev)
        print(f"  [load] {node}: events={ev['event_id'].nunique()}", flush=True)
    ev = pd.concat(frames, ignore_index=True)
    ev["target_date"] = pd.to_datetime(ev["target_date"])
    return ev.drop_duplicates("event_id")


def assign_split(target_date: pd.Series) -> pd.Series:
    """Map a target date to train / val / test (or NaN when out of window)."""
    d = pd.to_datetime(target_date)
    split = pd.Series(pd.NA, index=d.index, dtype="object")
    split[d <= TRAIN_END] = "train"
    split[(d > TRAIN_END) & (d <= VALID_END)] = "val"
    split[(d > VALID_END) & (d <= TEST_END)] = "test"
    return split


def main() -> int:
    ensure_out_dir()

    panel = pd.read_parquet(PANEL, columns=["event_id"])
    events = load_event_dates()

    split = events.copy()
    split["split"] = assign_split(split["target_date"])

    # Restrict to events that actually appear in the panel.
    split = split[split["event_id"].isin(set(panel["event_id"]))].copy()
    split = split[split["split"].notna()].copy()
    split["target_date"] = split["target_date"].dt.strftime("%Y-%m-%d")

    out = split[["event_id", "target_date", "split"]].drop_duplicates("event_id")
    out.to_parquet(OUT_DIR / "train_val_test_split.parquet", index=False)

    counts = out["split"].value_counts()
    print(
        f"\n[split] events={len(out)} "
        f"train={counts.get('train', 0)} "
        f"val={counts.get('val', 0)} "
        f"test={counts.get('test', 0)}",
        flush=True,
    )
    print(f"[write] {OUT_DIR / 'train_val_test_split.parquet'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
