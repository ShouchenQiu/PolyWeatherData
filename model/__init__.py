"""model package exports.

Exports are loaded lazily so training-only imports such as
``training.dataset`` do not pull the production inference pipeline.
"""
from __future__ import annotations

from typing import Any

__all__ = ["TftModel", "McSampler", "ModelPipeline"]


def __getattr__(name: str) -> Any:
    if name == "TftModel":
        from model.tft import TftModel

        return TftModel
    if name == "McSampler":
        from model.mc import McSampler

        return McSampler
    if name == "ModelPipeline":
        from model.pipeline import ModelPipeline

        return ModelPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
