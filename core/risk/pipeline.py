"""RiskPipeline: AND-gate approval and min-positive sizing across N risk models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.risk.base import BaseRiskModel

if TYPE_CHECKING:
    from core.types import OrderIntent


class RiskPipeline(BaseRiskModel):
    """Composite risk model that requires all sub-models to approve.

    Approval is the logical AND of all sub-model ``approve_signal`` calls.
    Sizing is the minimum positive value returned by all sub-models'
    ``compute_size`` calls.

    Parameters
    ----------
    models:
        Non-empty list of BaseRiskModel instances.

    Raises
    ------
    ValueError
        If ``models`` is empty.
    """

    def __init__(self, models: list[BaseRiskModel]) -> None:
        if not models:
            raise ValueError(
                "RiskPipeline requires at least one sub-model."
            )
        self._models = models

    def approve_signal(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
    ) -> bool:
        """Return True only when every sub-model approves.

        Short-circuits on the first rejection; ``compute_size`` is never
        called inside this method.
        """
        for model in self._models:
            if not model.approve_signal(signal, current_exposure):
                return False
        return True

    def compute_size(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
        positions:        object | None = None,
    ) -> float:
        """Return the minimum positive size across all sub-models.

        Must be called only after ``approve_signal`` returns True.
        """
        sizes = [
            model.compute_size(signal, current_exposure,
                               positions=positions)
            for model in self._models
        ]
        positive = [s for s in sizes if s > 0]
        if not positive:
            return 0.0
        return min(positive)
