"""Build aligned quantile prediction frames for offline evidence."""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def build_quantile_prediction_frame(
    *,
    checkpoint_path: Path | str,
    training_dataset: Any,
    frame: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Reload a TFT checkpoint and align its quantiles with one audited split."""
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, QuantileLoss

    dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        frame,
        predict=True,
        stop_randomization=True,
    )
    loader = dataset.to_dataloader(
        train=False,
        batch_size=int(batch_size),
        num_workers=0,
    )
    with _portable_checkpoint(
        checkpoint_path,
        torch=torch,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        logging_metrics=torch.nn.ModuleList(
            [SMAPE(), MAE(), RMSE(), MAPE()]
        ),
    ) as portable_checkpoint:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(portable_checkpoint),
            map_location="cpu",
        )
    prediction = model.predict(
        loader,
        mode="quantiles",
        return_y=True,
        return_index=True,
        trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": True,
        },
    )
    quantiles = prediction.output.detach().cpu().numpy()
    targets = prediction.y[0].detach().cpu().numpy()
    weights = prediction.y[1].detach().cpu().numpy()
    if quantiles.ndim != 3 or quantiles.shape[-1] != 3:
        raise RuntimeError(
            f"unexpected quantile prediction shape: {quantiles.shape}"
        )
    groups = {
        str(key): group.sort_values("local_idx").tail(quantiles.shape[1])
        for key, group in frame.groupby("init_time_str", sort=True)
    }
    index = prediction.index
    group_keys = (
        [str(value) for value in index["init_time_str"].tolist()]
        if index is not None and "init_time_str" in index
        else sorted(groups)
    )
    rows: list[pd.DataFrame] = []
    for position, group_key in enumerate(group_keys):
        metadata = groups.get(group_key)
        if metadata is None or len(metadata) != quantiles.shape[1]:
            raise RuntimeError(
                f"cannot align validation prediction group {group_key}"
            )
        rows.append(
            pd.DataFrame(
                {
                    "init_time": pd.to_datetime(group_key, utc=True),
                    "valid_time": metadata["valid_time"].to_numpy(),
                    "fxx": metadata["fxx"].to_numpy(),
                    "target": targets[position],
                    "p10": quantiles[position, :, 0],
                    "p50": quantiles[position, :, 1],
                    "p90": quantiles[position, :, 2],
                    "base_temp_c": metadata["TMP_2m"].to_numpy(),
                    "weight": weights[position],
                    "is_extrapolated": metadata.get(
                        "is_extrapolated",
                        pd.Series(False, index=metadata.index),
                    ).to_numpy(dtype=bool),
                    "extrapolation_step": metadata.get(
                        "extrapolation_step",
                        pd.Series(0, index=metadata.index),
                    ).to_numpy(dtype=int),
                }
            )
        )
    if not rows:
        raise RuntimeError("quantile prediction produced no aligned rows")
    return pd.concat(rows, ignore_index=True)


@contextmanager
def _portable_checkpoint(
    checkpoint_path: Path | str,
    *,
    torch: Any,
    loss: Any,
    logging_metrics: Any,
):
    """Replace pickled CUDA-bound metric objects without changing weights."""
    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    hyperparameters = payload.get("hyper_parameters", {})
    hyperparameters["loss"] = loss
    hyperparameters["logging_metrics"] = logging_metrics
    special = payload.get("__special_save__", {})
    special["loss"] = loss
    special["logging_metrics"] = logging_metrics
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as handle:
        torch.save(payload, handle.name)
        yield Path(handle.name)
