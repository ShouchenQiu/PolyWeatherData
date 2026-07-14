from __future__ import annotations

import pandas as pd

from scripts.run_historical_market_panel import _score_target_range


def test_score_target_range_uses_the_actual_forecast_cache_coverage() -> None:
    snapshots = pd.DataFrame(
        {
            "target_date": ["2025-05-31", "2025-06-01", "2026-03-31"],
        }
    )

    assert _score_target_range(snapshots) == ("2025-05-31", "2026-03-31")
