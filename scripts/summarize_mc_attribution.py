"""Summarize immutable MC attribution outcome and CLOB evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from training.mc_attribution import MC_ATTRIBUTION_VARIANTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    variants = [item.variant_id for item in MC_ATTRIBUTION_VARIANTS]
    rows: list[dict[str, object]] = []
    for period in ("april", "may_early", "may_late"):
        coverage: tuple[int, int] | None = None
        for variant in variants:
            summary = json.loads((args.root / variant / f"{period}_signal_vs_clob" / "summary.json").read_text())
            sample = summary["sample"]
            current = (int(sample["event_count"]), int(sample["vector_count"]))
            if coverage is None:
                coverage = current
            elif coverage != current:
                raise ValueError(f"{period} variants do not share an identical CLOB panel")
            comparison = summary["comparison"]
            rows.append({
                "evidence": "clob", "period": period, "variant_id": variant,
                "event_days": current[0], "snapshots": current[1],
                "rps": summary["sources"]["model_raw"]["normalized_rps"],
                "rps_vs_midpoint": comparison["normalized_rps"]["model_minus_midpoint"],
                "log_score": summary["sources"]["model_raw"]["log_score"],
                "log_score_vs_midpoint": comparison["log_score"]["model_minus_midpoint"],
                "true_integer_probability": summary["sources"]["model_raw"]["true_bucket_probability"],
            })
    historical = pd.read_csv(args.root / "historical_outcome" / "historical_summary.csv")
    for row in historical.to_dict("records"):
        rows.append({"evidence": "historical_outcome", "period": "2025-06_to_2026-03", **row})
    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / "attribution_summary.csv", index=False)
    historical_rows = table[table["evidence"] == "historical_outcome"]
    historical_rps = historical_rows.set_index("variant_id")["rps"].to_dict()
    report = {
        "schema_version": 1,
        "data_class": "mc_component_attribution_report",
        "variants": variants,
        "rows": len(table),
        "historical_rps": historical_rps,
        "decision": {
            "production_change": "none",
            "retain_sigma_scales": True,
            "retain_partial_day_bias": True,
            "retain_cross_hour_dependence": True,
            "follow_on_candidate": "conditional_cross_hour_residual_dependence",
            "rationale": (
                "Removing any current MC component does not restore May market "
                "skill; removing cross-hour dependence is the largest and most "
                "consistent historical degradation."
            ),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="ascii")
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
