"""KellyCriterion risk model stub."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.risk.base import BaseRiskModel

if TYPE_CHECKING:
    from core.types import OrderIntent


class KellyCriterion(BaseRiskModel):
    """Kelly Criterion position-sizing stub.

    Not yet implemented.  Raises NotImplementedError on all method calls.
    Exists to reserve the interface for a future slice.
    """

    def approve_signal(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
    ) -> bool:
        raise NotImplementedError("KellyCriterion is not yet implemented.")

    def compute_size(
        self,
        signal:           "OrderIntent",
        current_exposure: float,
        positions:        object | None = None,
        is_paper:         bool | None = None,
    ) -> float:
        raise NotImplementedError("KellyCriterion is not yet implemented.")
