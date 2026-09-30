"""Central path configuration for the ICDM Polymarket weather study.

Every pipeline script imports paths from here instead of hard-coding them.
Two locations are configurable through environment variables, so the same
code runs on a workstation or a server without edits:

    POLYWEATHER_CLOB_ROOT   raw downloaded CLOB artifacts
                            (default: <repo>/data_store/clob)
    POLYWEATHER_OUT_DIR     generated outputs: panels, scores, figures
                            (default: <repo>/output)

Both defaults are relative to the repository root, which is found by walking
up from this file.  Nothing else in the pipeline needs a machine-specific path.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (pipeline/common.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw CLOB artifacts: <root>/<STATION>/<artifact>/{events,markets,trades}.parquet
CLOB_ROOT = Path(
    os.environ.get("POLYWEATHER_CLOB_ROOT", REPO_ROOT / "data_store" / "clob")
)

# Output directory for every intermediate and final artifact.
OUT_DIR = Path(os.environ.get("POLYWEATHER_OUT_DIR", REPO_ROOT / "output"))

# Weather observations (daily-high argmax times) live alongside the outputs.
ARGMAX_FILE = OUT_DIR / "daily_high_argmax.parquet"

# Canonical panel produced by pipeline/1_panel and consumed by every later step.
PANEL = OUT_DIR / "first_layer_panel_argmax_30min.parquet"

# Chronological split assignment (train / val / test) for the panel events.
SPLIT = OUT_DIR / "train_val_test_split.parquet"


def ensure_out_dir() -> Path:
    """Create the output directory if needed and return it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
