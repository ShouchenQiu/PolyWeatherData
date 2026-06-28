from __future__ import annotations

import pandas as pd

from training.audit_frames import AuditedFrameCache


def test_audited_frame_cache_normalizes_mixed_timestamp_columns(tmp_path):
    cache = AuditedFrameCache(
        tmp_path / "audit_frames",
        tmp_path / "data_store",
    )
    frame = pd.DataFrame(
        {
            "init_time": [
                "2026-05-31 23:00:00+00:00",
                pd.Timestamp("2026-05-31 23:00:00", tz="UTC"),
            ],
            "valid_time": [
                "2026-05-31 22:00:00+00:00",
                pd.Timestamp("2026-06-01 00:00:00", tz="UTC"),
            ],
            "value": [1.0, 2.0],
        }
    )

    cache.save(
        "test",
        "KLGA",
        "2026-05-31",
        "2026-05-31",
        frame,
        source_fingerprint="fixed",
    )

    persisted = pd.read_parquet(cache.frame_path("test"))
    assert pd.api.types.is_datetime64_any_dtype(persisted["init_time"])
    assert pd.api.types.is_datetime64_any_dtype(persisted["valid_time"])
    assert str(persisted["init_time"].dt.tz) == "UTC"
    assert str(persisted["valid_time"].dt.tz) == "UTC"
    assert persisted["init_time"].nunique() == 1
