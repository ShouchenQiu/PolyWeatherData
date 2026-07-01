"""Training artifact builders for TFT candidate models."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from model.features import TARGET
from training.config import ExperimentTemplate


def write_training_artifacts(
    *,
    output_dir: Path,
    station_id: str,
    version: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    known: list[str],
    unknown: list[str],
    training: Any,
    result: dict[str, Any],
    torch_module: Any,
    lightning_module: Any,
    experiment_config: ExperimentTemplate,
    random_seed: int,
    torch_deterministic: bool,
    precision: str | int,
    num_workers: int,
    settings: Any,
) -> list[Path]:
    """Write candidate evidence files and return paths for registry publish."""
    metadata = build_training_metadata(
        station_id=station_id,
        version=version,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        known=known,
        unknown=unknown,
        training=training,
        result=result,
        torch_module=torch_module,
        lightning_module=lightning_module,
        random_seed=random_seed,
        torch_deterministic=torch_deterministic,
        precision=precision,
        num_workers=num_workers,
        settings=settings,
    )
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )

    experiment_path = output_dir / "experiment_config.json"
    experiment_config.write_json(experiment_path)

    contract_path = output_dir / "model_contract.json"
    contract_path.write_text(
        json.dumps(
            build_model_contract(
                station_id=station_id,
                known=known,
                unknown=unknown,
                metadata=metadata,
                experiment_config=experiment_config,
                settings=settings,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="ascii",
    )

    result_path = output_dir / "train_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="ascii",
    )
    return [metadata_path, experiment_path, contract_path, result_path]


def build_training_metadata(
    *,
    station_id: str,
    version: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    known: list[str],
    unknown: list[str],
    training: Any,
    result: dict[str, Any],
    torch_module: Any,
    lightning_module: Any,
    random_seed: int,
    torch_deterministic: bool,
    precision: str | int,
    num_workers: int,
    settings: Any,
) -> dict[str, Any]:
    return {
        "version": version,
        "station_id": station_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "torch_deterministic": torch_deterministic,
        "torch_version": getattr(torch_module, "__version__", None),
        "lightning_version": getattr(lightning_module, "__version__", None),
        "pytorch_forecasting_version": _module_version("pytorch_forecasting"),
        "cuda_available": bool(torch_module.cuda.is_available()) if hasattr(torch_module, "cuda") else False,
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cudnn_version": _cudnn_version(torch_module),
        "gpu_model": _gpu_model(torch_module),
        "amp_enabled": "mixed" in str(precision),
        "precision": str(precision),
        "accelerator": "auto",
        "num_workers": num_workers,
        "batch_size": settings.tft_batch_size,
        "min_epochs": settings.tft_min_epochs,
        "max_epochs": settings.tft_max_epochs,
        "train_sample_count": _init_sample_count(df_train),
        "val_sample_count": _init_sample_count(df_val),
        "test_sample_count": _init_sample_count(df_test),
        "train_row_count": len(df_train),
        "val_row_count": len(df_val),
        "test_row_count": len(df_test),
        "data_order_hash": _data_order_hash(df_train, df_val, df_test),
        "target_normalization_params": _target_normalizer_params(training),
        "feature_list_and_order": {
            "known_reals": list(known),
            "unknown_reals": list(unknown),
            "target": TARGET,
        },
        "known_reals": list(known),
        "unknown_reals": list(unknown),
        "target": TARGET,
        "split_ranges": {
            "train": [settings.train_start, settings.train_end],
            "val": [settings.val_start, settings.val_end],
            "test": [settings.test_start, settings.test_end],
        },
        "platform": platform.platform(),
        "result": result,
    }


def build_model_contract(
    *,
    station_id: str,
    known: list[str],
    unknown: list[str],
    metadata: dict[str, Any],
    experiment_config: ExperimentTemplate,
    settings: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "station_id": station_id,
        "target": TARGET,
        "known_reals": list(known),
        "unknown_reals": list(unknown),
        "max_encoder_length": settings.tft_max_encoder_length,
        "max_prediction_length": settings.tft_max_prediction_length,
        "quantiles": list(settings.tft_quantiles),
        "data_order_hash": metadata["data_order_hash"],
        "training_config_hash": experiment_config.config_hash(),
        "torch_version": metadata["torch_version"],
        "pytorch_forecasting_version": metadata["pytorch_forecasting_version"],
    }


def _init_sample_count(df: pd.DataFrame) -> int:
    if "init_time_str" in df.columns:
        return int(df["init_time_str"].nunique())
    return 0


def _data_order_hash(*frames: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    cols = ["split", "init_time_str", "local_idx", "valid_time", "fxx", TARGET]
    for frame in frames:
        present = [col for col in cols if col in frame.columns]
        work = frame[present].copy()
        for col in work.columns:
            if pd.api.types.is_datetime64_any_dtype(work[col]):
                work[col] = pd.to_datetime(work[col], utc=True).astype(str)
        payload = work.to_json(orient="records", date_format="iso")
        hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


def _target_normalizer_params(training: Any) -> dict[str, Any]:
    normalizer = getattr(training, "target_normalizer", None)
    if normalizer is None:
        return {"repr": None}
    params: dict[str, Any] = {"class": type(normalizer).__name__, "repr": repr(normalizer)}
    get_parameters = getattr(normalizer, "get_parameters", None)
    if callable(get_parameters):
        try:
            value = get_parameters()
            params["parameters"] = value.tolist() if hasattr(value, "tolist") else value
        except Exception as exc:  # noqa: BLE001
            params["parameters_error"] = str(exc)
    return params


def _module_version(module_name: str) -> str | None:
    import sys

    module = sys.modules.get(module_name)
    return getattr(module, "__version__", None)


def _cudnn_version(torch_module: Any) -> int | None:
    try:
        version_fn = torch_module.backends.cudnn.version
        return int(version_fn()) if callable(version_fn) else None
    except Exception:  # noqa: BLE001
        return None


def _gpu_model(torch_module: Any) -> str | None:
    try:
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return str(torch_module.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return None
    return None
