from __future__ import annotations

import json
import pickle

import pandas as pd

from scripts.generate_backtest_predictions import main


def test_prediction_cli_uses_audited_frame_and_writes_lineage_manifest(
    tmp_path,
) -> None:
    frame_path = tmp_path / "frame.parquet"
    pd.DataFrame(
        [
            {
                "init_time_str": "2026-05-01 00:00:00+00:00",
                "local_idx": index,
            }
            for index in range(120)
        ]
    ).to_parquet(frame_path, index=False)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    training_dataset = tmp_path / "training.pkl"
    with training_dataset.open("wb") as handle:
        pickle.dump({"dataset": "fixture"}, handle)
    output = tmp_path / "predictions.parquet"

    def fake_predict(**kwargs):
        assert kwargs["checkpoint_path"] == checkpoint
        assert kwargs["training_dataset"] == {"dataset": "fixture"}
        assert len(kwargs["frame"]) == 120
        return pd.DataFrame(
            [
                {
                    "init_time": "2026-05-01T00:00:00Z",
                    "valid_time": "2026-05-01T01:00:00Z",
                    "fxx": 1,
                    "target": 0.0,
                    "p10": -1.0,
                    "p50": 0.0,
                    "p90": 1.0,
                    "base_temp_c": 20.0,
                    "weight": 0.0,
                    "is_extrapolated": False,
                    "extrapolation_step": 0,
                }
            ]
        )

    result = main(
        [
            "--frame",
            str(frame_path),
            "--checkpoint",
            str(checkpoint),
            "--training-dataset",
            str(training_dataset),
            "--output",
            str(output),
        ],
        predict_fn=fake_predict,
    )

    manifest = json.loads(
        (tmp_path / "predictions.parquet.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == 0
    assert len(pd.read_parquet(output)) == 1
    assert manifest["schema_version"] == 2
    assert manifest["row_count"] == 1
    assert manifest["source_sha256"]["checkpoint"]
    assert manifest["sha256"]["predictions.parquet"]
