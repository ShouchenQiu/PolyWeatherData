from __future__ import annotations

from training.config import ExperimentTemplate
from scripts.train_tft import _materialize_experiment


def test_materialized_ab_configs_differ_only_by_history_switch(tmp_path) -> None:
    source = tmp_path / "source.json"
    ExperimentTemplate.from_json(
        "experiments/klga/tft_trend_6h_v2.json"
    ).write_json(source)

    disabled_path = _materialize_experiment(
        run_dir=tmp_path,
        version="ab-a",
        source_path=source,
        history_target="disabled",
    )
    enabled_path = _materialize_experiment(
        run_dir=tmp_path,
        version="ab-b",
        source_path=source,
        history_target="enabled",
    )
    disabled = ExperimentTemplate.from_json(disabled_path)
    enabled = ExperimentTemplate.from_json(enabled_path)

    assert disabled.model.use_history_target is False
    assert enabled.model.use_history_target is True
    assert disabled.trainer == enabled.trainer
    assert disabled.splits == enabled.splits
